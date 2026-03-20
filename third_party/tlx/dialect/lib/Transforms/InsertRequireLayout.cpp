#include "IR/Dialect.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-amd-insert-require-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace tlx = ::mlir::triton::tlx;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXINSERTREQUIRELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

// Peel memdesc view ops (index, subslice, trans) to reach the underlying
// allocation that async copies write into.
static Value traceToAlloc(Value memDesc) {
  Value current = memDesc;
  while (true) {
    if (auto indexOp = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = indexOp.getSrc();
    } else if (auto subsliceOp =
                   current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = subsliceOp.getSrc();
    } else if (auto transOp =
                   current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = transOp.getSrc();
    } else {
      break;
    }
  }
  return current;
}

// Recursively search users of a memdesc allocation for AsyncCopyGlobalToLocalOp
// and return its global source's memory order.
static std::optional<SmallVector<unsigned>>
searchAsyncCopyOrder(Value val) {
  for (Operation *user : val.getUsers()) {
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(user)) {
      auto srcTy = dyn_cast<RankedTensorType>(asyncCopy.getSrc().getType());
      if (!srcTy || !srcTy.getEncoding())
        continue;
      if (!dyn_cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding()))
        continue;
      return ttg::getOrderForMemory(srcTy);
    }
    if (isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(user)) {
      if (auto result = searchAsyncCopyOrder(user->getResult(0)))
        return result;
    }
  }
  return std::nullopt;
}

// Find the memory order of the async copy producer that writes into the same
// allocation as the given local_load's source.
static std::optional<SmallVector<unsigned>>
findAsyncCopySourceOrder(ttg::LocalLoadOp localLoadOp) {
  return searchAsyncCopyOrder(traceToAlloc(localLoadOp.getSrc()));
}

// Find the DotOperandEncodingAttr reachable from a local_load's result.
// Only follows ConvertLayoutOp chains; other use patterns are not supported.
static ttg::DotOperandEncodingAttr
findDotOperandEncoding(ttg::LocalLoadOp localLoadOp) {
  SmallVector<Value> worklist = {localLoadOp.getResult()};
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation *user : val.getUsers()) {
      if (user->getNumResults() != 1)
        continue;
      auto resType = dyn_cast<RankedTensorType>(user->getResult(0).getType());
      if (!resType)
        continue;
      if (auto dotOpEnc =
              dyn_cast<ttg::DotOperandEncodingAttr>(resType.getEncoding()))
        return dotOpEnc;
      if (isa<ttg::ConvertLayoutOp>(user))
        worklist.push_back(user->getResult(0));
    }
  }
  return nullptr;
}

// DMA hardware cannot transpose, so when async_copy_global_to_local fills
// shared memory, the shared encoding's order must match the source's order.
// If they differ, recompute the encoding using the producer's order.
static ttg::SwizzledSharedEncodingAttr
adjustEncodingForProducerOrder(ttg::LocalLoadOp localLoadOp,
                               ttg::SwizzledSharedEncodingAttr encoding) {
  auto producerOrderOpt = findAsyncCopySourceOrder(localLoadOp);
  if (!producerOrderOpt)
    return encoding;

  SmallVector<unsigned> producerOrder = *producerOrderOpt;
  SmallVector<unsigned> encodingOrder(encoding.getOrder());
  if (producerOrder == encodingOrder)
    return encoding;

  LLVM_DEBUG({
    DBGS() << "Producer order [";
    llvm::interleaveComma(producerOrder, llvm::dbgs());
    llvm::dbgs() << "] differs from encoding order [";
    llvm::interleaveComma(encodingOrder, llvm::dbgs());
    llvm::dbgs() << "]; recomputing shared encoding\n";
  });

  auto dotOpEnc = findDotOperandEncoding(localLoadOp);
  if (!dotOpEnc) {
    localLoadOp.emitWarning(
        "async copy order mismatch but no DotOperandEncoding found; "
        "keeping original encoding");
    return encoding;
  }

  auto memDescType = cast<ttg::MemDescType>(localLoadOp.getSrc().getType());
  unsigned bitWidth = memDescType.getElementType().getIntOrFloatBitWidth();

  auto newEncoding = ttg::SwizzledSharedEncodingAttr::get(
      encoding.getContext(), dotOpEnc, memDescType.getShape(), producerOrder,
      encoding.getCGALayout(), bitWidth, /*needTrans=*/false);

  LDBG("Recomputed shared encoding: " << newEncoding);
  return newEncoding;
}

LogicalResult insertRequireLayout(ModuleOp m) {
  OpBuilder builder(m.getContext());
  LDBG("insertRequiredLayout\n");
  WalkResult result = m.walk([&](tt::DotOp dotOp) -> WalkResult {
    SetVector<Operation *> backwardSet;
    BackwardSliceOptions options;
    options.inclusive = false;
    options.omitUsesFromAbove = false;
    if (failed(mlir::getBackwardSlice(dotOp.getOperation(), &backwardSet,
                                      options))) {
      return WalkResult::interrupt();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "DotOp\n";
      dotOp.dump();
    });
    for (Operation *op : backwardSet) {
      if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "LocalLoadOp\n";
          localLoadOp.dump();
        });
        bool incompatible = false;
        auto encoding = getSharedEncIfAllUsersAreDotEnc(
                            localLoadOp->getResult(0), incompatible)
                            .value_or(nullptr);
        if (encoding) {
          encoding = adjustEncodingForProducerOrder(localLoadOp, encoding);
          LLVM_DEBUG({
            llvm::dbgs() << "SwizzledSharedEncodingAttr\n";
            encoding.dump();
          });
          builder.setInsertionPoint(localLoadOp);
          auto encodingAttr = mlir::cast<Attribute>(encoding);
          auto loadMemDescTy = op->getOperands()[0];
          if (auto type = dyn_cast<ttg::MemDescType>(loadMemDescTy.getType())) {
            auto newType = ttg::MemDescType::get(
                type.getShape(), type.getElementType(), encodingAttr,
                type.getMemorySpace(), type.getMutableMemory());
            auto converLayoutOp = mlir::triton::tlx::RequireLayoutOp::create(builder, 
                op->getLoc(), newType, loadMemDescTy);
            localLoadOp->setOperand(0, converLayoutOp.getResult());
          }
        } else {
          localLoadOp->emitError(
              "Cannot find appropriate shared encoding for local load op");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

struct TLXInsertRequireLayoutPass
    : public impl::TLXInsertRequireLayoutBase<TLXInsertRequireLayoutPass> {
public:
  using impl::TLXInsertRequireLayoutBase<
      TLXInsertRequireLayoutPass>::TLXInsertRequireLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(tlx::insertRequireLayout(m))) {
      signalPassFailure();
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
