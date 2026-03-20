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
namespace amdpipeliner {
std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value loadedValue);
}

namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXINSERTREQUIRELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

// Trace from a memdesc value back to its root local_alloc, walking through
// MemDescIndexOp and MemDescSubsliceOp chains.
static Value traceToAlloc(Value memDesc) {
  Value current = memDesc;
  while (true) {
    if (auto indexOp = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = indexOp.getSrc();
    } else if (auto subsliceOp =
                   current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = subsliceOp.getSrc();
    } else {
      break;
    }
  }
  return current;
}

// Find the memory order of the global-to-local async copy source that writes
// into the same allocation as the given local_load's source memdesc.  Returns
// std::nullopt when no async copy producer is found.
static std::optional<SmallVector<unsigned>>
findAsyncCopySourceOrder(ttg::LocalLoadOp localLoadOp) {
  Value alloc = traceToAlloc(localLoadOp.getSrc());

  // Recursively search users of `val` for AsyncCopyGlobalToLocalOp.
  std::function<std::optional<SmallVector<unsigned>>(Value)> search;
  search = [&](Value val) -> std::optional<SmallVector<unsigned>> {
    for (Operation *user : val.getUsers()) {
      if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(user)) {
        auto srcTy = dyn_cast<RankedTensorType>(asyncCopy.getSrc().getType());
        if (!srcTy || !srcTy.getEncoding())
          continue;
        return ttg::getOrderForMemory(srcTy);
      }
      if (isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(user)) {
        if (auto result = search(user->getResult(0)))
          return result;
      }
    }
    return std::nullopt;
  };

  return search(alloc);
}

// Find the DotOperandEncodingAttr reachable from a local_load's result by
// tracing through ConvertLayoutOp users.
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

// When the shared memory buffer is filled by async_copy_global_to_local (DMA),
// the shared encoding's order must match the global source's order because the
// DMA hardware cannot transpose.  If the consumer-derived encoding has a
// different order, recompute it using the producer's order.
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
    LDBG("Could not find DotOperandEncodingAttr; keeping original encoding");
    return encoding;
  }

  auto memDescType = cast<ttg::MemDescType>(localLoadOp.getSrc().getType());
  unsigned bitWidth = memDescType.getElementType().getIntOrFloatBitWidth();

  auto newEncoding = ttg::SwizzledSharedEncodingAttr::get(
      encoding.getContext(), dotOpEnc, memDescType.getShape(), producerOrder,
      encoding.getCTALayout(), bitWidth, /*needTrans=*/false);

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
        auto encoding = mlir::amdpipeliner::getSharedEncIfAllUsersAreDotEnc(
                            localLoadOp->getResult(0))
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
            auto converLayoutOp = builder.create<tlx::RequireLayoutOp>(
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
