#include "IR/Dialect.h"
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tlx/dialect/include/Analysis/LayoutPropagation.h"
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
using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace tlx = ::mlir::triton::tlx;
namespace amdgpu = ::mlir::triton::amdgpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXINSERTREQUIRELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

namespace {

// ============================================================================
// Backward dataflow analysis: propagate the required dot-operand encoding and
// rewrite legality from tt.DotOp operands backward through convert_layout
// chains and region-branch carriers to local_load ops.
//
// The analysis tracks both the desired dot encoding and whether rewriting the
// value is still legal. We union convert_layout source/result anchors so mixed
// uses that branch through sibling convert chains share the same legality
// state.
// ============================================================================

class DotRewriteState {
public:
  enum class Kind {
    Uninitialized,
    Required,
    Conflict,
    Illegal,
  };

  DotRewriteState() = default;
  explicit DotRewriteState(Attribute enc)
      : kind(Kind::Required), encoding(enc) {}

  static DotRewriteState getConflict() {
    DotRewriteState state;
    state.kind = Kind::Conflict;
    return state;
  }

  static DotRewriteState getIllegal() {
    DotRewriteState state;
    state.kind = Kind::Illegal;
    return state;
  }

  bool operator==(const DotRewriteState &rhs) const {
    return kind == rhs.kind && encoding == rhs.encoding;
  }

  bool isUninitialized() const { return kind == Kind::Uninitialized; }
  bool isRequired() const { return kind == Kind::Required; }
  bool isConflict() const { return kind == Kind::Conflict; }
  bool isIllegal() const { return kind == Kind::Illegal; }

  Attribute getEncoding() const {
    assert(isRequired() && "expected required dot encoding state");
    return *encoding;
  }

  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    if (isConflict()) {
      os << "<conflict>";
      return;
    }
    if (isIllegal()) {
      os << "<illegal>";
      return;
    }
    if (isRequired()) {
      encoding->print(os);
      return;
    }
    llvm_unreachable("unknown dot rewrite state");
  }

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const DotRewriteState &state) {
    state.print(os);
    return os;
  }

  static DotRewriteState meet(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    if (lhs.isIllegal() || rhs.isIllegal())
      return getIllegal();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    if (lhs.isConflict() || rhs.isConflict())
      return getConflict();
    return getConflict();
  }

  static DotRewriteState join(const DotRewriteState &lhs,
                              const DotRewriteState &rhs) {
    return meet(lhs, rhs);
  }

private:
  Kind kind = Kind::Uninitialized;
  std::optional<Attribute> encoding;
};

class DotRewriteLattice : public Lattice<DotRewriteState> {
public:
  using Lattice::Lattice;
};

static bool isTrackedDotValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

static bool
isTransparentDotUserBeforeConstraintMaterialization(Operation *op,
                                                    unsigned operandIndex) {
  // This is the pre-materialization half of the shared dot-layout policy. The
  // insert pass sees raw tt.dot users and the convert_layout chain that still
  // connects them to local_load. After those converts are rewritten into
  // explicit tlx.require_layout anchors, tlx-propagate-layout enforces the same
  // transparent-carrier policy from the tlx.require_layout anchors instead.
  if (auto dotOp = dyn_cast<tt::DotOp>(op))
    return operandIndex < 2 && operandIndex < dotOp->getNumOperands();

  return isa<ttg::ConvertLayoutOp>(op) || isTransparentLayoutCarrierOp(op);
}

class DotRewriteBackward
    : public SparseBackwardDataFlowAnalysis<DotRewriteLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void initializeEquivalentLatticeAnchor(Operation *top) override {
    top->walk([&](ttg::ConvertLayoutOp cvt) {
      if (!isTrackedDotValue(cvt.getSrc()) ||
          !isTrackedDotValue(cvt.getResult()))
        return;
      unionLatticeAnchors<DotRewriteLattice>(cvt.getSrc(), cvt.getResult());
    });
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotRewriteLattice *> operands,
                 ArrayRef<const DotRewriteLattice *> results) override {
    // Seed from tt.DotOp: propagate the required dot-operand encoding to
    // the values that define operands A and B.
    if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      for (unsigned i = 0; i < 2; ++i) {
        auto type = cast<RankedTensorType>(dotOp.getOperand(i).getType());
        if (auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(type.getEncoding())) {
          ChangeResult changed = operands[i]->meet(DotRewriteState(dotEnc));
          propagateIfChanged(operands[i], changed);
        }
      }
      return success();
    }

    // If a tracked tensor value is used by an unsupported operation, the
    // require_layout rewrite is no longer legal for that entire carrier chain.
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isTrackedDotValue(operand))
        continue;
      if (isTransparentDotUserBeforeConstraintMaterialization(op, index))
        continue;

      DotRewriteState operandState = operands[index]->getValue();
      if (operandState.isUninitialized())
        continue;

      ChangeResult changed =
          operands[index]->meet(DotRewriteState::getIllegal());
      propagateIfChanged(operands[index], changed);
    }

    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {
    if (!isTrackedDotValue(operand.get()))
      return;
    // For RegionBranchTerminatorOpInterface (scf.yield) and
    // RegionBranchOpInterface (scf.for init args), allow the dot encoding
    // to propagate backward so local_load ops feeding loop-carried values
    // produce dot_op layout directly.
    if (isTransparentLayoutCarrierOp(operand.getOwner()))
      return;
    poisonUnhandledCase(operand);
  }

  void visitCallOperand(OpOperand &operand) override {
    poisonUnhandledCase(operand);
  }

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
  void setToExitState(DotRewriteLattice *lattice) override {}

private:
  void poisonUnhandledCase(OpOperand &operand) {
    if (!isTrackedDotValue(operand.get()))
      return;

    auto *lattice = getLatticeElement(operand.get());
    DotRewriteState state = lattice->getValue();
    if (state.isUninitialized())
      return;

    ChangeResult changed = lattice->meet(DotRewriteState::getIllegal());
    propagateIfChanged(lattice, changed);
  }
};

// ============================================================================
// Rewrite helpers
// ============================================================================

static ttg::SwizzledSharedEncodingAttr
computeSharedEncFromDotEnc(ttg::DotOperandEncodingAttr dotEnc,
                           ttg::LocalLoadOp localLoadOp, bool needTrans) {
  auto resultType = cast<RankedTensorType>(localLoadOp.getType());
  auto ctaLayout = ttg::getCGALayout(resultType.getEncoding());
  unsigned bitWidth = resultType.getElementType().getIntOrFloatBitWidth();

  if (needTrans) {
    // For transposed loads, compute encoding using the pre-transpose shape
    // and order so the swizzling is correct for the transposed access pattern.
    auto memTransOp = localLoadOp.getSrc().getDefiningOp<ttg::MemDescTransOp>();
    assert(memTransOp && "needTrans requires memdesc_trans source");
    auto srcMemType = cast<ttg::MemDescType>(memTransOp.getSrc().getType());
    auto srcShape = srcMemType.getShape();
    auto srcOrder = ttg::getOrder(srcMemType);
    return ttg::SwizzledSharedEncodingAttr::get(
        localLoadOp->getContext(), dotEnc, srcShape, srcOrder, ctaLayout,
        bitWidth, /*needTrans=*/true);
  }

  auto order = ttg::getOrderForMemory(resultType);
  return ttg::SwizzledSharedEncodingAttr::get(localLoadOp->getContext(), dotEnc,
                                              resultType.getShape(), order,
                                              ctaLayout, bitWidth,
                                              /*needTrans=*/false);
}

// Walk up the memdesc def-chain through subview / reinterpret ops to
// the source value (typically a `ttg.local_alloc`).
static Value findMemDescRoot(Value memdesc) {
  Value root = memdesc;
  while (root) {
    Operation *def = root.getDefiningOp();
    if (!def)
      break;
    // Treat memdesc views as aliases of the same allocation. This lets TDM
    // anchors and dot-consumer discovery meet on the full buffer even when
    // WMMA consumes a sliced or transposed view.
    if (isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
            ttg::MemDescSubsliceOp, ttg::MemDescTransOp>(def)) {
      root = def->getOperand(0);
      continue;
    }
    break;
  }
  return root;
}

// True if any sibling-subview user of the memdesc value's source alloc is
// a TDM op (load or store). Used by the dot-path walk to hand off TDM-fed
// buffers to the TDM anchor — both walks targeting the same alloc with
// different encodings would otherwise conflict in the propagation lattice.
// Stores are included so a store-anchored alloc isn't also targeted by the
// dot-path walk; the store's hardware verifier requires the default
// padded-shared encoding, which the dot-path swizzled anchor would clobber.
static bool isFedByTDM(Value memdesc) {
  llvm::SetVector<Value> worklist;
  worklist.insert(findMemDescRoot(memdesc));
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (isa<amdgpu::AsyncTDMCopyGlobalToLocalOp,
              amdgpu::AsyncTDMCopyLocalToGlobalOp>(u))
        return true;
      // Follow sibling views from the root allocation so local_load(subslice)
      // and local_load(transpose(subslice)) are recognized as TDM-fed too.
      if (isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
              ttg::MemDescSubsliceOp, ttg::MemDescTransOp>(u))
        worklist.insert(u->getResult(0));
    }
  }
  return false;
}

static void applyRequireLayout(ttg::SwizzledSharedEncodingAttr encoding,
                               ttg::LocalLoadOp localLoadOp,
                               OpBuilder &builder, bool needTrans) {
  auto loadMemDesc = localLoadOp->getOperand(0);

  // For transposed loads, apply the require_layout to the source of
  // memdesc_trans rather than its result so PropagateLayout doesn't need
  // to backward-propagate through the transpose.
  Value targetMemDesc = loadMemDesc;
  if (needTrans) {
    if (auto memTransOp =
            loadMemDesc.getDefiningOp<ttg::MemDescTransOp>())
      targetMemDesc = memTransOp.getSrc();
  }

  if (targetMemDesc.getDefiningOp<tlx::RequireLayoutOp>())
    return;

  // Defer to the TDM anchor for buffers fed by `amdgpu.async_tdm_*`. The
  // TDM walk picks a padded encoding that's compatible with the descriptor
  // (and dot-aware when applicable); inserting a sibling swizzled anchor
  // here would conflict with that constraint and widen the lattice to
  // unknown.
  if (isFedByTDM(targetMemDesc))
    return;

  // Respect user-specified order on the source memdesc.
  if (auto srcType = dyn_cast<ttg::MemDescType>(targetMemDesc.getType())) {
    if (auto srcEnc =
            dyn_cast<ttg::SwizzledSharedEncodingAttr>(srcType.getEncoding())) {
      if (srcEnc.getOrder() != encoding.getOrder()) {
        LDBG("Respecting user-specified order "
             << srcEnc << " instead of derived " << encoding);
        encoding = ttg::SwizzledSharedEncodingAttr::get(
            encoding.getContext(), encoding.getVec(), encoding.getPerPhase(),
            encoding.getMaxPhase(), srcEnc.getOrder(), encoding.getCGALayout());
      }
    }
  }

  if (auto type = dyn_cast<ttg::MemDescType>(targetMemDesc.getType())) {
    auto newType = ttg::MemDescType::get(
        type.getShape(), type.getElementType(), mlir::cast<Attribute>(encoding),
        type.getMemorySpace(), type.getMutableMemory(), type.getAllocShape());
    if (needTrans) {
      // Insert require_layout before the memdesc_trans, then rebuild the
      // memdesc_trans so its result type is inferred from the new source.
      auto memTransOp = loadMemDesc.getDefiningOp<ttg::MemDescTransOp>();
      builder.setInsertionPoint(memTransOp);
      auto requireOp = tlx::RequireLayoutOp::create(
          builder, localLoadOp->getLoc(), newType, targetMemDesc);
      auto newMemTrans = ttg::MemDescTransOp::create(
          builder, memTransOp.getLoc(), requireOp.getResult(),
          memTransOp.getOrder());
      memTransOp.getResult().replaceAllUsesWith(newMemTrans.getResult());
      memTransOp->erase();
    } else {
      builder.setInsertionPoint(localLoadOp);
      auto requireOp = tlx::RequireLayoutOp::create(
          builder, localLoadOp->getLoc(), newType, targetMemDesc);
      localLoadOp->setOperand(0, requireOp.getResult());
    }
  }
}

static void materializeTensorRequireLayout(tt::DotOp dotOp,
                                           unsigned operandIndex,
                                           OpBuilder &builder) {
  Value operand = dotOp.getOperand(operandIndex);
  auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!cvt)
    return;

  auto dstType = dyn_cast<RankedTensorType>(cvt.getType());
  if (!dstType || !isSupportedDotConstraintEncoding(dstType.getEncoding()))
    return;

  builder.setInsertionPoint(cvt);
  auto requireOp = tlx::RequireLayoutOp::create(builder, cvt.getLoc(),
                                                cvt.getType(), cvt.getSrc());
  dotOp->setOperand(operandIndex, requireOp.getResult());
  if (cvt.getResult().use_empty())
    cvt.erase();
}

static void materializeDotUserTensorConstraints(ModuleOp m,
                                                OpBuilder &builder) {
  m.walk([&](tt::DotOp dotOp) {
    for (unsigned i = 0; i < 2; ++i)
      materializeTensorRequireLayout(dotOp, i, builder);
  });
}

// ============================================================================
// AMD TDM descriptor anchors
// ============================================================================
//
// `amdgpu.async_tdm_copy_global_to_local` writes into a user-provided shared
// memory buffer whose required encoding is determined by the descriptor's
// shape and element type — and, when the buffer feeds a `tt.dot`, by the
// dot operand's WMMA encoding. When TLX users allocate the buffer with the
// default `local_alloc(...)` (no explicit `layout=`), the alloc's encoding
// is the generic non-swizzled `SwizzledSharedEncoding(maxPhase=1)` — which
// the TDM op verifier accepts but which produces wrong LDS data on real
// gfx1250 hardware. We anchor a `tlx.require_layout` on the buffer operand
// of every TDM copy so `tlx-propagate-layout` rewrites the source
// `local_alloc` (and any subview / loop-carrier chain) to the
// descriptor-compatible encoding.

namespace {

struct DotConsumerInfo {
  int opIdx;
  unsigned kWidth;
  bool operator==(const DotConsumerInfo &o) const {
    return opIdx == o.opIdx && kWidth == o.kWidth;
  }
};

// Per-memdesc-value lattice tracking the dot-operand consumer info that
// any downstream `ttg.local_load -> tt.dot` chain would impose.
//
//   Uninitialized — no consumer information observed yet.
//   Required(info) — every reachable LocalLoadOp consumer agrees on info.
//   Conflict       — two reachable consumers disagree; the TDM anchor
//                    falls back to the descriptor-default encoding.
class DotConsumerState {
public:
  enum class Kind { Uninitialized, Required, Conflict };

  DotConsumerState() = default;
  explicit DotConsumerState(DotConsumerInfo info)
      : kind(Kind::Required), info(info) {}

  static DotConsumerState getConflict() {
    DotConsumerState s;
    s.kind = Kind::Conflict;
    return s;
  }

  bool isUninitialized() const { return kind == Kind::Uninitialized; }
  bool isRequired() const { return kind == Kind::Required; }
  bool isConflict() const { return kind == Kind::Conflict; }
  DotConsumerInfo getInfo() const {
    assert(isRequired());
    return info;
  }

  bool operator==(const DotConsumerState &o) const {
    return kind == o.kind && info == o.info;
  }

  // Backward propagation meet: uninitialized yields to any concrete
  // state; equal concrete states stay concrete; conflicting concrete
  // states widen to Conflict.
  static DotConsumerState meet(const DotConsumerState &lhs,
                               const DotConsumerState &rhs) {
    if (lhs.isConflict() || rhs.isConflict())
      return getConflict();
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    return getConflict();
  }
  static DotConsumerState join(const DotConsumerState &lhs,
                               const DotConsumerState &rhs) {
    return meet(lhs, rhs);
  }

  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<uninitialized>";
      return;
    }
    if (isConflict()) {
      os << "<conflict>";
      return;
    }
    os << "Required{opIdx=" << info.opIdx << ", kWidth=" << info.kWidth << "}";
  }

  friend raw_ostream &operator<<(raw_ostream &os, const DotConsumerState &s) {
    s.print(os);
    return os;
  }

private:
  Kind kind = Kind::Uninitialized;
  DotConsumerInfo info{};
};

class DotConsumerLattice : public Lattice<DotConsumerState> {
public:
  using Lattice::Lattice;
};

// Sparse backward dataflow that propagates dot-operand consumer info
// from `ttg.local_load` ops up through the memdesc def-chain (subview /
// reinterpret / `tlx.require_layout`) to the source `ttg.local_alloc`,
// and back down to all sibling subviews (including the TDM op's buffer
// operand) via the framework's region-branch / scf.for iter-arg /
// warp_specialize handling.
class DotConsumerBackward
    : public SparseBackwardDataFlowAnalysis<DotConsumerLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<DotConsumerLattice *> operands,
                 ArrayRef<const DotConsumerLattice *> results) override {
    // Seed: a LocalLoadOp whose result tensor carries DotOperandEncoding
    // imposes that encoding on its memdesc operand.
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      if (auto resTy = dyn_cast<RankedTensorType>(load.getResult().getType())) {
        if (auto dotEnc = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
                resTy.getEncoding())) {
          DotConsumerState seed(
              DotConsumerInfo{static_cast<int>(dotEnc.getOpIdx()),
                              static_cast<unsigned>(dotEnc.getKWidth())});
          if (!operands.empty()) {
            ChangeResult changed = operands[0]->meet(seed);
            propagateIfChanged(operands[0], changed);
          }
        }
      }
      return success();
    }

    // Transparent memdesc carriers: meet each result lattice into the
    // corresponding memdesc operand lattice. This propagates from
    // subview / reinterpret / require_layout results back to their
    // source memdesc, which lets the alloc converge to the meet of
    // every sibling subview's dot-consumer state.
    if (isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
            ttg::MemDescSubsliceOp, ttg::MemDescTransOp, tlx::RequireLayoutOp>(
            op)) {
      for (const auto resultLattice : results) {
        for (auto [i, operandLattice] : llvm::enumerate(operands)) {
          if (!isa<ttg::MemDescType>(op->getOpOperand(i).get().getType()))
            continue;
          ChangeResult changed =
              operandLattice->meet(resultLattice->getValue());
          propagateIfChanged(operandLattice, changed);
        }
      }
      return success();
    }

    // Other users of memdesc values (TDM ops, local_store, etc.) impose
    // no dot-consumer requirement — leaving the operand lattice alone.
    return success();
  }

  // Required pure-virtual overrides. This is an info-only lattice (not a
  // legality analysis like DotRewriteBackward, which poisons here): leaving
  // unanalyzed cases as Uninitialized is safe because findDotConsumer
  // returns nullopt for both Uninitialized and Conflict, and the caller
  // falls back to buildDefaultTDMDescriptorEncoding.
  void visitBranchOperand(OpOperand &) override {}
  void visitCallOperand(OpOperand &) override {}
  void setToExitState(DotConsumerLattice *) override {}
  void visitNonControlFlowArguments(RegionSuccessor &,
                                    ArrayRef<BlockArgument>) override {}
};

} // namespace

// Read the dot-consumer info from the propagation lattice at the alloc
// reachable from `buffer`. Returns nullopt for Uninitialized / Conflict
// (the caller falls back to the descriptor-default encoding).
static std::optional<DotConsumerInfo> findDotConsumer(Value buffer,
                                                      DataFlowSolver &solver) {
  Value root = findMemDescRoot(buffer);
  auto *lattice = solver.lookupState<DotConsumerLattice>(root);
  if (!lattice)
    return std::nullopt;
  const auto &state = lattice->getValue();
  if (!state.isRequired())
    return std::nullopt;
  return state.getInfo();
}

// Pick a descriptor-compatible encoding for `buf`. For TDM loads, prefer
// the WMMA-tuned `composePaddedLayout` when the buffer feeds a `tt.dot`
// (correct for both the TDM op and the local_load -> tt.dot lowering).
// For TDM stores, the hardware verifier requires
// `padInterval == innermost block dim`, ruling out the WMMA-tuned form
// (which sets `padInterval = max(innerDim, bankWrapInterval)`); always
// fall back to the descriptor-shape-only default.
//
// Using a dot-tuned encoding for loads is safe because the AMD
// `OptimizeDescriptorEncoding` pass walks TDM ops and propagates this
// encoding back to the descriptor's `TensorDescType`, so the hardware
// (which reads stride from the descriptor) and the alloc (which uses this
// encoding to size the LDS region) agree by construction.
static Attribute chooseTDMBufEncoding(Operation *tdmOp, Value buf,
                                      ttg::MemDescType bufType,
                                      tt::TensorDescType descTy,
                                      bool allowDotAware,
                                      DataFlowSolver &solver) {
  ArrayRef<int64_t> shape = descTy.getBlockType().getShape();
  Type elementType = descTy.getBlockType().getElementType();
  unsigned rank = shape.size();
  SmallVector<unsigned> order(rank);
  for (unsigned i = 0; i < rank; ++i)
    order[i] = rank - 1 - i;
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(buf.getContext(), rank);

  Attribute encoding;
  if (allowDotAware) {
    if (auto info = findDotConsumer(buf, solver)) {
      auto archStr = getAMDArch(tdmOp->getParentOfType<ModuleOp>());
      auto targetInfo = tt::AMD::TargetInfo(archStr.value_or("").str());
      // bufType (MemDescType) carries the alloc's CGA layout; the descriptor
      // type is still un-encoded at this point.
      encoding =
          composePaddedLayout(targetInfo, info->opIdx, info->kWidth,
                              cast<ttg::TensorOrMemDesc>(bufType), order);
    }
  }
  if (!encoding)
    encoding = buildDefaultTDMDescriptorEncoding(buf.getContext(), shape, order,
                                                 cgaLayout, elementType);
  return encoding;
}

// Insert `tlx.require_layout` between `buf` and `tdmOp`'s memdesc operand,
// rewriting it to a descriptor-compatible padded encoding. Idempotent: if
// the buffer is already produced by a `require_layout`, leave it alone.
template <typename TDMOp>
static void anchorTDMRequireLayout(TDMOp tdmOp, Value buf,
                                   MutableOperandRange operandToRewire,
                                   bool allowDotAware, OpBuilder &builder,
                                   DataFlowSolver &solver) {
  if (buf.getDefiningOp<tlx::RequireLayoutOp>())
    return;
  auto bufType = dyn_cast<ttg::MemDescType>(buf.getType());
  if (!bufType)
    return;
  auto descTy = cast<tt::TensorDescType>(tdmOp.getDesc().getType());

  Attribute encoding =
      chooseTDMBufEncoding(tdmOp, buf, bufType, descTy, allowDotAware, solver);
  if (!encoding)
    return;

  builder.setInsertionPoint(tdmOp);
  auto newType = ttg::MemDescType::get(
      bufType.getShape(), bufType.getElementType(), encoding,
      bufType.getMemorySpace(), bufType.getMutableMemory(),
      bufType.getAllocShape());
  auto requireOp =
      tlx::RequireLayoutOp::create(builder, tdmOp.getLoc(), newType, buf);
  operandToRewire.assign(requireOp.getResult());
}

static void materializeTDMConstraints(ModuleOp m, OpBuilder &builder,
                                      DataFlowSolver &solver) {
  m.walk([&](Operation *op) {
    if (auto load = dyn_cast<amdgpu::AsyncTDMCopyGlobalToLocalOp>(op))
      anchorTDMRequireLayout(load, load.getResult(), load.getResultMutable(),
                             /*allowDotAware=*/true, builder, solver);
    else if (auto store = dyn_cast<amdgpu::AsyncTDMCopyLocalToGlobalOp>(op))
      anchorTDMRequireLayout(store, store.getSrc(), store.getSrcMutable(),
                             /*allowDotAware=*/false, builder, solver);
  });
}

} // namespace

// ============================================================================
// Main pass logic
// ============================================================================

// Trace a value backward through scf.for iter_args to find all producing
// local_load ops. Returns false if an unsupported producer is encountered.
static bool traceToLocalLoads(
    Value val, SmallVector<ttg::LocalLoadOp> &loads,
    SmallVector<std::pair<scf::ForOp, unsigned>> &forUpdates) {
  SetVector<Value> worklist;
  DenseSet<Value> visited;
  worklist.insert(val);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    if (auto loadOp = v.getDefiningOp<ttg::LocalLoadOp>()) {
      loads.push_back(loadOp);
      continue;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp)
        return false;
      unsigned idx = blockArg.getArgNumber() - forOp.getNumInductionVars();
      if (idx >= forOp.getNumRegionIterArgs())
        return false;
      forUpdates.emplace_back(forOp, idx);
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      worklist.insert(forOp.getInitArgs()[idx]);
      worklist.insert(yield.getOperand(idx));
      continue;
    }
    if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
      unsigned idx = cast<OpResult>(v).getResultNumber();
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      worklist.insert(yield.getOperand(idx));
      continue;
    }
    return false;
  }
  return !loads.empty();
}

// Canonicalize tt.trans consuming local_load results (directly or via
// scf.for iter_args) into memdesc_trans → local_load, so the transpose
// lives in the memdesc domain where MemoryOpToLLVM lowers it to ds_read_tr
// intrinsics. Runs before the dataflow solver so it sees a plain
// local_load feeding the dot operand chain.
static void hoistTransToMemdesc(ModuleOp m, OpBuilder &builder) {
  // Collect all tt.trans ops eligible for hoisting.
  DenseSet<Operation *> loadsToHoist;
  SmallVector<std::pair<scf::ForOp, unsigned>> forUpdates;
  SmallVector<tt::TransOp> transToErase;

  m.walk([&](tt::TransOp transOp) {
    if (transOp.getOrder() != ArrayRef<int32_t>{1, 0})
      return;

    SmallVector<ttg::LocalLoadOp> loads;
    SmallVector<std::pair<scf::ForOp, unsigned>> updates;
    if (!traceToLocalLoads(transOp.getSrc(), loads, updates))
      return;

    for (auto load : loads)
      loadsToHoist.insert(load);
    for (auto u : updates)
      forUpdates.push_back(u);
    transToErase.push_back(transOp);
  });

  if (transToErase.empty())
    return;

  // Phase 1: Insert memdesc_trans before each identified local_load.
  for (auto *op : loadsToHoist) {
    auto localLoadOp = cast<ttg::LocalLoadOp>(op);
    builder.setInsertionPoint(localLoadOp);
    auto memTransOp = ttg::MemDescTransOp::create(
        builder, localLoadOp.getLoc(), localLoadOp.getSrc(),
        ArrayRef<int32_t>({1, 0}));
    localLoadOp.getSrcMutable().assign(memTransOp.getResult());
    auto oldType = localLoadOp.getType();
    auto shape = oldType.getShape();
    auto newType = RankedTensorType::get({shape[1], shape[0]},
                                         oldType.getElementType(),
                                         oldType.getEncoding());
    localLoadOp.getResult().setType(newType);
  }

  // Phase 2: Update scf.for iter_arg and result types.
  DenseSet<std::pair<Operation *, unsigned>> updatedForArgs;
  for (auto [forOp, idx] : forUpdates) {
    if (!updatedForArgs.insert({forOp.getOperation(), idx}).second)
      continue;
    auto oldType =
        cast<RankedTensorType>(forOp.getRegionIterArg(idx).getType());
    auto shape = oldType.getShape();
    auto newType = RankedTensorType::get({shape[1], shape[0]},
                                         oldType.getElementType(),
                                         oldType.getEncoding());
    forOp.getRegionIterArg(idx).setType(newType);
    forOp.getResult(idx).setType(newType);
  }

  // Phase 3: Erase tt.trans (replace uses with now-transposed source).
  for (auto transOp : transToErase) {
    transOp.getResult().replaceAllUsesWith(transOp.getSrc());
    transOp->erase();
  }
}

LogicalResult insertRequireLayout(ModuleOp m) {
  OpBuilder builder(m.getContext());
  LDBG("insertRequireLayout");

  // Canonicalize local_load → tt.trans into memdesc_trans → local_load before
  // the dataflow solver runs so it sees the standard local_load → dot path.
  hoistTransToMemdesc(m, builder);

  // --- Run backward dataflow analysis ---
  // SparseBackwardDataFlowAnalysis requires a SymbolTableCollection even though
  // this analysis does not query symbol tables directly.
  SymbolTableCollection symbolTable;
  DataFlowSolver solver;
  loadBaselineAnalyses(solver);
  solver.load<DotRewriteBackward>(symbolTable);
  // Memdesc-level dot-consumer analysis used by the TDM anchor below to
  // pick the WMMA-tuned padded encoding when a downstream `local_load`
  // requires it. Conflict-widens-to-default; the framework handles
  // scf.for iter-args, region branches, and warp_specialize captures.
  solver.load<DotConsumerBackward>(symbolTable);
  if (failed(solver.initializeAndRun(m)))
    return failure();

  // InsertRequireLayout owns constraint synthesis only:
  // 1. Discover dot-fed local_load ops and add the missing memdesc-side
  //    tlx.require_layout constraints for shared memory.
  // 2. Rewrite matched dot-path ttg.convert_layout ops into explicit tensor
  //    tlx.require_layout constraints.
  // 3. Leave tensor/register propagation, region-branch retagging, and final
  //    convert cleanup to tlx-propagate-layout and downstream cleanup passes.
  m.walk([&](ttg::LocalLoadOp localLoadOp) {
    auto *lattice =
        solver.lookupState<DotRewriteLattice>(localLoadOp.getResult());
    if (!lattice || lattice->getValue().isUninitialized())
      return;

    if (lattice->getValue().isIllegal() || lattice->getValue().isConflict()) {
      LDBG("Skipping local_load rewrite due to state: " << lattice->getValue());
      localLoadOp->emitRemark()
          << "dot operand layout constraint cannot be folded into local_load "
             "because the value has incompatible users or conflicting dot "
             "requirements";
      return;
    }

    auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
        lattice->getValue().getEncoding());
    if (!dotEnc)
      return;

    LDBG("local_load needs dot encoding: " << dotEnc);

    // Detect if this local_load was hoisted from a tt.trans (its source is
    // a memdesc_trans). If so, compute the shared encoding with needTrans.
    bool needTrans =
        localLoadOp.getSrc().getDefiningOp<ttg::MemDescTransOp>() != nullptr;

    // Insert RequireLayoutOp for memdesc swizzling.
    auto sharedEnc = computeSharedEncFromDotEnc(dotEnc, localLoadOp, needTrans);
    applyRequireLayout(sharedEnc, localLoadOp, builder, needTrans);
  });

  materializeDotUserTensorConstraints(m, builder);

  // Anchor `tlx.require_layout` on AMD TDM copy buffer operands so
  // tlx-propagate-layout rewrites the source `local_alloc` to a
  // descriptor-compatible encoding.
  materializeTDMConstraints(m, builder, solver);

  return success();
}

struct TLXInsertRequireLayoutPass
    : public impl::TLXInsertRequireLayoutBase<TLXInsertRequireLayoutPass> {
public:
  using impl::TLXInsertRequireLayoutBase<
      TLXInsertRequireLayoutPass>::TLXInsertRequireLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(tlx::insertRequireLayout(m)))
      signalPassFailure();
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
