# TLX Flash-Attention Causal Optimizations

Optimization log for the TLX causal Flash-Attention forward kernel, targeting
**gfx950 (MI350)**. All work lives in
`third_party/tlx/tutorials/amd-fa-pipelined_test.py`.

The goal was to close the gap between the causal kernel and the non-causal
`async_prefetch` baseline. Causal started at **67–73%** of non-causal and now
reaches **95–98%**.

## How to run

```bash
source ~/nod/tlx/venv/tlx_venv/bin/activate

# Benchmark (TFLOPS)
python third_party/tlx/tutorials/amd-fa-pipelined_test.py \
    -b 1 -hq 64 -sq 8192 16384 -d 64 128 -causal true \
    --kernel async_prefetch_causal async_prefetch_persistent_causal \
             async_prefetch_persistent_nomirror_causal async_prefetch_dynamic_causal

# Correctness
pytest third_party/tlx/tutorials/amd-fa-pipelined_test.py::test_fa_correctness \
    -k async_prefetch_causal -s --tb=short
```

## Available kernel modes

Each mode = a registry key → host wrapper → `@triton.jit` kernel. All live in
`third_party/tlx/tutorials/amd-fa-pipelined_test.py`; `causal=False` falls back
to the non-causal `async_prefetch` for every mode.

| Registry key | Host wrapper | JIT kernel | Description | Status |
|---|---|---|---|---|
| `async_prefetch` | `flash_attn_async_prefetch` | `_attn_fwd_async_prefetch` | Non-causal reference / fallback | baseline |
| `async_prefetch_causal` | `flash_attn_async_prefetch_causal` | `_attn_fwd_async_prefetch_causal` | Peeled mask + **static mirror-pair** balance + XCD L2 remap | **default / champion (D=128)** |
| `async_prefetch_persistent_causal` | `flash_attn_async_prefetch_persistent_causal` | `_attn_fwd_persistent_causal` | Persistent, XCD-grouped, mirror-balanced | champion (D=64) |
| `async_prefetch_persistent_nomirror_causal` | `flash_attn_async_prefetch_persistent_nomirror_causal` | `_attn_fwd_persistent_nomirror_causal` | Persistent, no pairing, m-major + heavy-first | experiment (worse) |
| `async_prefetch_persistent_balanced_causal` | `flash_attn_async_prefetch_persistent_balanced_causal` | `_attn_fwd_persistent_balanced_causal` | Persistent, **constant-cost fold bundling** (generalized mirror) | general + matches mirror |
| `async_prefetch_persistent_partition_causal` | `flash_attn_async_prefetch_persistent_partition_causal` | `_attn_fwd_persistent_partition_causal` | Persistent, **cumulative-cost partition** (no bundling, variable tiles/program) | general but ~12–19% slower |
| `async_prefetch_dynamic_causal` | `flash_attn_async_prefetch_dynamic_causal` | `_attn_fwd_dynamic_causal` | FCFS dynamic work-stealing via per-XCD atomic queue | experiment (worse) |
| `streamk_causal` | `flash_attn_streamk_causal` | `_attn_fwd_streamk_causal` | **StreamK** K-block split + partial-softmax reduction | wins low-parallelism (≤1 wave) |
| `causal_auto` | `flash_attn_causal_auto` | (dispatch) | Auto-pick StreamK (low-parallelism) vs mirror (else) | **best across regimes** |

### Per-mode run / repro

Each command benchmarks one mode (TFLOPS) across D=64/128 and N=8192/16384;
append `-causal false` to also bench the non-causal fallback. Swap
`python ... ` for the pytest line to check correctness instead.

```bash
source ~/nod/tlx/venv/tlx_venv/bin/activate
BENCH="python third_party/tlx/tutorials/amd-fa-pipelined_test.py -b 1 -hq 64 -sq 8192 16384 -d 64 128 -causal true --kernel"

# Non-causal reference / fallback
$BENCH async_prefetch -causal false

# Mirror static (default / champion D=128)
$BENCH async_prefetch_causal

# Mirror persistent (champion D=64)
$BENCH async_prefetch_persistent_causal

# No-mirror persistent (m-major, heavy-first)
$BENCH async_prefetch_persistent_nomirror_causal

# Constant-cost fold bundling (general + matches mirror)
$BENCH async_prefetch_persistent_balanced_causal

# Cumulative-cost partition (no bundling; general but slower)
$BENCH async_prefetch_persistent_partition_causal

# StreamK split + reduction (use a low-parallelism shape: few heads / short N)
python third_party/tlx/tutorials/amd-fa-pipelined_test.py -b 1 -hq 8 -sq 1024 2048 4096 -d 128 -causal true --kernel streamk_causal async_prefetch_causal

# Auto-dispatch (best across regimes)
$BENCH causal_auto

# FCFS dynamic work-stealing (per-XCD atomic queue)
$BENCH async_prefetch_dynamic_causal

# Correctness for a given mode (all N incl. partial-block fallback):
pytest third_party/tlx/tutorials/amd-fa-pipelined_test.py::test_fa_correctness \
    -k async_prefetch_dynamic_causal -s --tb=short
```

## Consolidated performance (all modes)

TFLOPS, gfx950, B=1, H=64, bf16, causal. Single apples-to-apples run
(cache cleared, `do_bench` warmup=25/rep=100). Higher is better.

| Config | Torch SDPA | mirror static | mirror persistent | no-mirror persistent | FCFS dynamic |
|---|---|---|---|---|---|
| D=64,  N=8192  | 324.6 | **684.9** | 700.7 | 630.3 | 666.0 |
| D=64,  N=16384 | 391.9 | 727.6 | **745.3** | 660.5 | 694.0 |
| D=128, N=8192  | 416.7 | **801.9** | 792.6 | 739.0 | 779.1 |
| D=128, N=16384 | 464.5 | **854.7** | 850.6 | 738.0 | 776.5 |

(Run-to-run noise is ~±1%; mirror persistent edges D=64, mirror static edges
D=128 — both within a couple percent. No-mirror and FCFS dynamic are
consistently behind.)

## Causal vs non-causal (final, mirror static default)

| Config | orig baseline | optimized | non-causal | % of non-causal |
|---|---|---|---|---|
| D=64,  N=8192  | 351 | 683 | 715 | 95% |
| D=64,  N=16384 | 439 | 726 | 739 | 98% |
| D=128, N=8192  | 479 | 802 | 840 | 95% |
| D=128, N=16384 | 551 | 851 | 867 | 98% |

Was 67–73% of non-causal; now 95–98%. All 36 correctness cases pass; the
causal=False path matches `async_prefetch` exactly.

Total stacked gains, in order of impact:
1. Mirror-pair load balancing (biggest: eliminates the causal tail-wave)
2. XCD L2 remap (+5–9%)
3. Peel mask out of the hot loop (+11–16%)
4. BLOCK_N=128 for D=64 (+7–8% on D=64)
5. EVEN_N boundary-mask drop (~0.5%)

---

## Framing / intuition

- Causal does ~half the effective work. The benchmark divides FLOPs by 2
  (`valid_el = N(N+1)/2`), so **causal TFLOP is naturally lower than non-causal**
  even for a "perfect" kernel: fixed overheads (prologue/epilogue, diagonal
  masking, load imbalance) amortize over half the FLOPs.
- Wall-clock parity vs non-causal at equal N is the truer signal than TFLOP.
- Main causal-specific costs: (1) mask in every inner iter [FIXED by peeling],
  (2) load imbalance — `pid_m=0` does 1 K-block, `pid_m=max` does N/BLOCK_N
  blocks, so the final wave is all-heavy tiles; (3) diagonal-block masking.

## Baseline (async_prefetch, gfx950)

| Config | nocausal | causal |
|---|---|---|
| D=64,  N=8192  | 715.9 | 351.3 |
| D=64,  N=16384 | 738.7 | 440.1 |
| D=128, N=8192  | 838.6 | 477.9 |
| D=128, N=16384 | 868.8 | 549.9 |

## Step 1 — Peel causal mask out of the hot loop  (landed)

New kernel `_attn_fwd_async_prefetch_causal` / wrapper
`flash_attn_async_prefetch_causal` (registry key `async_prefetch_causal`).

Idea: for query block `pid_m`, K blocks fully below the diagonal
(`start_n + BLOCK_N <= pid_m*BLOCK_M`) need no mask. Only the `BLOCK_M//BLOCK_N`
diagonal blocks do. Split into:
- unmasked steady-state loop (no `tl.where`, FMA-friendly softmax)
- short masked diagonal tail loop

The async double-buffered prefetch chain stays continuous across the boundary
(slot = `global_block_idx % 2`), so no pipeline bubble. `diag_start = pid_m*BLOCK_M`
is a multiple of BLOCK_N (`BLOCK_M%BLOCK_N==0`) so the split is exact.

| Config | baseline | peeled | gain |
|---|---|---|---|
| D=64,  8k  | 351 | 405 | +15% |
| D=64,  16k | 436 | 483 | +11% |
| D=128, 8k  | 479 | 556 | +16% |
| D=128, 16k | 552 | 631 | +14% |

## Step 2 — Mirror-paired load balancing  (landed, BIG win)

Refactored per-tile compute into device fn `_attn_causal_tile`. Each program now
processes a *light* tile `pid_m` and its *heavy* mirror `NUM_M_BLOCKS-1-pid_m`
(guard `if pid_mirror > pid_m`), so combined K-block count ≈ `NUM_M_BLOCKS+1` for
every program → uniform work, no causal tail-wave imbalance. Grid first dim
halved to `(num_m+1)//2`.

Gotcha: calling the tile fn twice with `tlx.local_alloc` *inside* it crashed the
compiler at N=500 D=128 (partial block) — `iota_range Begin<=End` assertion.
FIX: hoist `local_alloc` to the outer kernel, allocate K/V double-buffers ONCE,
pass them into both tile calls. Safe because tile A drains all async loads before
tile B starts (last masked iter issues no prefetch).

| Config | baseline | peeled | +mirror | nocausal |
|---|---|---|---|---|
| D=64,  8k  | 351 | 405 | 610 | 716 |
| D=64,  16k | 436 | 483 | 648 | 739 |
| D=128, 8k  | 478 | 556 | 754 | 839 |
| D=128, 16k | 547 | 631 | 776 | 869 |

Causal D=128 now ~89% of non-causal.

## Step 3 — XCD L2 remap  (landed, BIG win)

gfx950 has 8 XCDs. HW round-robins a head's m-tiles across all 8 XCDs by flat
block id → each XCD touches every head's K/V → poor L2 reuse. `_remap_xcd`
groups contiguous flat pids onto the same XCD so each XCD owns whole heads
(K/V stays L2-resident). Requires a **flat 1D grid** + decode
`pid_m = pid % GRID_M`, `pid_hz = pid // GRID_M`.

Gotcha: the modulo decode crashes the compiler (`iota_range Begin<=End`) at
`GRID_M==1` (small N like 512). FIX: constexpr special-case `GRID_M==1`
(`pid_m≡0`) and a `NUM_XCDS==1` 2D-grid path for the non-remapped fallback.
NUM_XCDS defaults to 8.

| Config | +mirror | +remap | nocausal | % of nc |
|---|---|---|---|---|
| D=64,  8k  | 610 | 637 | 716 | 89% |
| D=64,  16k | 646 | 674 | 739 | 91% |
| D=128, 8k  | 758 | 801 | 839 | 95% |
| D=128, 16k | 779 | 854 | 869 | 98% |

EVEN_N (drop the redundant boundary `kn<N_CTX` where N is a clean multiple) gave
only a marginal ~0.5% bump; kept.

## Step 4 — BLOCK_N=128 for D<=64  (landed)

Now that the causal mask is peeled, D=64 prefers BLOCK_N=128 (more compute per
LDS barrier). D=64 8k 637→685, 16k 675→727. D=128 must stay BLOCK_N=64 (LDS
budget for double-buffered K+V). Wired into the wrapper default.

## Profiling note

Inspected AMDGCN (`~/.triton/cache`): vgpr_count=512 (maxed by BLOCK_M=256, fp32
acc [256,128]), only 7 VGPR spills (minor), 256 MFMA. Register pressure is the
occupancy limiter but spills are negligible. Given we already reach 95–98% of
non-causal, the remaining gap is the *inherent* diagonal-triangle
over-computation (~half of the `BLOCK_M//BLOCK_N` diagonal blocks per tile are
masked-away), a hard floor without finer diagonal tiling.

## Step 5 — Persistent-style alternative (vs mirror-pair)  (landed as 2nd kernel)

New kernel `_attn_fwd_persistent_causal` / wrapper
`flash_attn_async_prefetch_persistent_causal` (registry
`async_prefetch_persistent_causal`). Launches exactly NUM_SMS=256 resident
programs (gfx950 CU count). Design that actually works:
- **XCD-grouped**: head `hz` assigned to XCD `hz % NUM_XCDS`; each program only
  touches heads owned by its XCD → preserves the K/V L2 locality that a naive
  strided persistent destroys.
- **Mirror-balanced**: within an XCD, work units are mirror pairs
  `(p, NUM_M_BLOCKS-1-p)`, balanced like the mirror kernel.
- **Full-utilization flatten**: flatten `(head_on_xcd, pair)` and stride across
  the `NUM_LOCAL = NUM_SMS/NUM_XCDS` programs. Crucial — the first attempt
  (pair-loop nested in head-loop) left half the XCD's programs idle when
  `GRID_M < NUM_LOCAL` (N=8192), tanking it to ~520 TF.
- **Heavy-first**: not needed once balanced; mirror pairing already balances.

Lessons:
- Naive strided persistent (`tile in range(pid, total, NUM_SMS)`) = 444–529 TF —
  scatters each head across all 8 XCDs, killing L2. L2 locality dominates here,
  so any persistent scheme MUST keep heads XCD-local.
- Partial-block N (`N%BLOCK_N!=0`) hits the same compiler iota_range crash;
  wrapper falls back to the mirror kernel for those (not perf-critical).

| Config | mirror+remap | persistent | winner |
|---|---|---|---|
| D=64,  8k  | 686 | 700 | persistent +2% |
| D=64,  16k | 726 | 746 | persistent +3% |
| D=128, 8k  | 802 | 789 | mirror +1.6% |
| D=128, 16k | 851 | 847 | ~tie (mirror) |

Verdict: persistent wins D=64, mirror wins D=128 slightly. Both kept; mirror is
the simpler/robust default (`async_prefetch_causal`), persistent is the D=64
champion.

## Step 6 — NaN-prop max & disable-vector-combine  (tried, not adopted)

- `DISABLE_LLVM_OPT="disable-vector-combine"` (compile-time env, cleared cache):
  no meaningful change — within ±1% run-to-run noise on both kernels (e.g. D128
  16k mirror 843→854, D64 8k 686→684). Not adopted.
- NaN-propagating max (custom `tl.reduce` with `tl.maximum(..., propagate_nan=ALL)`,
  mirroring the Gluon kernel): no perf upside AND it *destabilized compilation* —
  perturbed the existing `iota_range Begin<=End` compiler bug so it began firing
  on more configs, including (nondeterministically) the large-N benchmark.
  Reverted entirely. Our masking sets entries to `-inf` which plain `max` handles
  correctly, and the diagonal always retains ≥1 valid entry, so NaN propagation
  buys nothing here. Verdict: skip on this kernel until the underlying iota bug
  is fixed.

## Step 7 — Persistent WITHOUT mirror (empirical test)  (confirmed worse)

New experimental kernel `_attn_fwd_persistent_nomirror_causal` (registry
`async_prefetch_persistent_nomirror_causal`): same XCD-grouping (L2), but each
work unit is a single m-tile (no pairing). To give averaging the best shot,
tiles are flattened **m-major** (`unit = m*heads_per_xcd + local_head`) so each
program's round-robin set spreads m across the full range (step
`NUM_LOCAL//heads_per_xcd`) instead of locking onto a fixed light/heavy band.

| Config | persistent mirror | persistent no-mirror | gap |
|---|---|---|---|
| D=64,  8k  | 702 | 620 | -12% |
| D=64,  16k | 747 | 648 | -13% |
| D=128, 8k  | 789 | 731 | -7% |
| D=128, 16k | 851 | 682 | -20% |

Conclusion: in our regime (few tiles/program, tile counts evenly divisible by
NUM_LOCAL, heads pinned to XCDs for L2) averaging cannot converge — a heavy band
of m-tiles still piles onto some programs, leaving a tail. Mirror's *static*
per-unit balancing is required. Persistent is a conceptual superset of mirror
only if it has enough well-mixed tiles per program to average; here it does not,
so the explicit mirror balancer stays. (Worst at D128 16k: num_m=64, heaviest
tiles cluster → -20%.)

## Step 8 — Heavy-first (bottom-of-mask) scheduling  (helps no-mirror only)

Added `REVERSE` flag to the no-mirror persistent kernel: schedule heaviest
m-tiles (bottom of causal mask) first (`pid_m = NUM_M_BLOCKS-1-m_idx`).

| Config | light-first | heavy-first | gain |
|---|---|---|---|
| D=64,  8k  | 621 | 631 | +1.6% |
| D=64,  16k | 654 | 661 | +1% |
| D=128, 8k  | 727 | 737 | +1.4% |
| D=128, 16k | 687 | 737 | +7% |

Heavy-first helps the *imbalanced* no-mirror kernel (biggest at the worst case
D128 16k, +7%): even though static per-program assignment makes makespan
theoretically order-independent, on real HW running the long tiles first
overlaps them across all programs early and leaves short light tails, cutting
end-of-kernel divergence. Set `REVERSE=True` default for the no-mirror kernel.

BUT it still does not beat the mirror variants (heavy-first no-mirror
632/659/740/733 vs mirror 685/727/801/854). And for the *balanced* mirror
kernels heavy-first is **moot**: every mirror pair `{p, num_m-1-p}` already has
equal weight (~num_m+1 blocks), so there is no heavy/light ordering to exploit.
Conclusion: heavy-first is a partial substitute for balancing, not a substitute
for mirror. Mirror stays the champion.

## Step 9 — Dynamic FCFS work-stealing (per-XCD atomic queue)  (slower)

Question: instead of a *fixed* tile→workgroup mapping, make it
first-come-first-serve — whichever WG is free grabs the next available output
tile. Implemented as `async_prefetch_dynamic_causal` (`_attn_fwd_dynamic_causal`):
- **Per-XCD atomic counters** (`Counter[NUM_XCDS]`): a program only pulls tiles
  for heads owned by its XCD (`hz % NUM_XCDS`), so K/V stay L2-resident in that
  XCD slice. A single global counter would balance perfectly but scatter heads
  across XCDs → L2 loss (the naive-strided ~520TF regime), so per-XCD it is.
- **Heavy-first order**: tile `idx` decodes to `pid_m = NUM_M_BLOCKS-1-m_idx`
  (LPT — claim the long bottom-of-mask tiles first).
- No mirror pairing needed: a WG that drew a heavy tile finishes later and pulls
  fewer subsequent tiles, so load self-balances; tail shrinks to ≤1 tile.

Two TLX gotchas hit & fixed:
1. **Atomic scope**: default `tl.atomic_add` was *not* cross-workgroup coherent
   here → counter read stale 0 → infinite GPU hang. Fix: explicit
   `sem="relaxed", scope="gpu"` (verified: 256 WGs claim 1000 tiles exactly once).
2. **while-loop OOB**: a natural `while idx < N:` work-queue **memory-faulted** —
   TLX only lowers the async-prefetch pipeline over `scf.for` (`tl.range`), not
   `scf.while`. Fix: express the claim loop as a *bounded* `tl.range` (upper
   bound = generous slack over avg tiles/program; programs spin the atomic and
   skip via a guard once done). Correct for all shapes.

| Config | mirror static | mirror persistent | FCFS dynamic | gap |
|---|---|---|---|---|
| D=64,  8k  | 683 | 700 | 661 | -3..6% |
| D=64,  16k | 728 | 745 | 692 | -5..7% |
| D=128, 8k  | 802 | 787 | 774 | -2..4% |
| D=128, 16k | 856 | 847 | 775 | -8..9% |

Tightening the loop cap (3× avg vs full queue) to kill spin-atomic overhead made
~no difference, so the gap is **intrinsic to dynamic scheduling**, not loop
overhead: (a) atomic contention on the hot path (per-XCD counter is a serialized
point), and (b) non-deterministic tile *arrival order* hurts L2/temporal reuse
vs the static schedule's deterministic head grouping.

Conclusion: FCFS dynamic is feasible and correct, but **slower** here. Dynamic
work-stealing only pays off when per-tile cost is *irregular/unpredictable*;
causal-FA tile cost is a deterministic function of `pid_m`, so static mirror
already balances perfectly and dynamic only adds contention + locality cost.
Mirror stays the champion. (Kept the kernel in the registry for the
irregular-shape case and as a reference.)

## Step 10 — General + mirror-fast: constant-cost fold bundling  (landed)

Goal: a scheme **as general as the no-mirror persistent kernel** (persistent,
no hardcoded `p<->N-1-p` pairing) but **as fast as static mirror**. New kernel
`async_prefetch_persistent_balanced_causal` (`_attn_fwd_persistent_balanced_causal`).

The investigation pinned down *why* the general schemes lagged, and the answer
generalizes mirror cleanly:

1. **Snake (boustrophedon) balancing** — pairing-free. The no-mirror kernel gives
   each program a fixed round-robin residue; since tile cost is monotone in the
   flattened index, that residue is a systematic load bias. Walking the work list
   in a snake (even rounds 0..P-1, odd rounds reversed) makes each program's
   *total* work symmetric about the mean — provably balanced to within one tile
   (verified: per-program block sums are identical). Result: 670/696/783/771 —
   recovers most of the no-mirror gap but still **2–10% behind mirror**.

2. **Why snake still loses (the key insight)**: per-program *total* balance is
   necessary but **not sufficient**. Snake makes every program do its heavy
   tiles early and light tiles late, so the kernel ends in an overhead-bound
   "all-light" tail — full prologue/epilogue per tile for only 1–4 K-blocks, MFMA
   utilisation sagging over time. Worst where the tail is longest (D128 N=16k,
   num_m=64 → -10%). What static mirror really buys is **constant work *per
   iteration*** (light `p` + heavy `N-1-p` = `num_m+1` blocks every step), which
   keeps fixed overheads amortised throughout. Constant-cost-per-iteration ⟺
   bundling a heavy tile with a light one ⟺ pairing. So a *strictly* pairing-free
   scheme cannot fully match mirror — the few-% gap is intrinsic.

   (Head-pinning each program to one head for L2 locality was also tried and did
   *not* help — slightly worse than spreading heads, since an XCD's L2 holds
   enough and spreading gives better wave-level balance.)

3. **Constant-cost fold bundling** — the generalization that wins. Each iteration
   processes the heaviest-remaining and lightest-remaining m-tile *together*
   (`p` and `num_m-1-p`), so every iteration costs ≈ `num_m+1` K-blocks: no light
   tail, uniform utilisation. This is a *fold* of the cost-sorted tile list; for
   the linear causal cost it reduces to the mirror pairing, but the construction
   only needs a *monotone* cost — a greedy-bundle (accumulate light tiles until a
   bundle hits the target cost) extends it to arbitrary/non-linear profiles that
   the fixed `p<->N-1-p` rule cannot express. Persistent + XCD-grouped, bundles
   flattened `(head, fold-pair)` and strided across programs.

| Config | mirror static | snake (pairing-free) | fold bundling | 
|---|---|---|---|
| D=64,  8k  | 685 | 670 | 699 |
| D=64,  16k | 726 | 696 | 747 |
| D=128, 8k  | 802 | 783 | 790 |
| D=128, 16k | 851 | 771 | 850 |

Conclusion: **constant-cost bundling is the general principle behind mirror.**
The fold kernel matches static/persistent mirror across the board while being
expressed as a general cost-driven bundling (persistent, any monotone cost) —
strictly more general than the hardcoded static-mirror rule. It is in fact
structurally equivalent to `async_prefetch_persistent_causal` (for causal,
fold == mirror pairing), which confirms the persistent-mirror kernel was already
the right general+fast answer; this kernel just states the principle explicitly.
The pairing-free snake remains the best *strictly* pairing-free option but is
intrinsically a few % short. All 12 correctness cases pass.

## Step 11 — Is mirror "hardcoded"? Cumulative-cost partition (no bundling)  (experiment, worse)

Mirror/fold bundles tiles in pairs ("exactly 2/program"). Is there something
*more general and simpler* that drops bundling entirely? New kernel
`async_prefetch_persistent_partition_causal` (`_attn_fwd_persistent_partition_causal`):
lay out an XCD's tiles in cost order and cut that 1-D list into NUM_LOCAL
contiguous chunks of **equal cumulative cost** — light programs get many cheap
m-rows, heavy programs get few expensive ones. Boundaries are precomputed on the
host from `cost(m) = (m+1) + ALPHA` (any cost model works → fully general); no
pairing, no fixed tiles/program, and heavy & light chunks run concurrently so
there's no global light tail.

| Config | mirror | partition (best ALPHA) |
|---|---|---|
| D=64,  8k  | 672 | 592  (-12%) |
| D=128, 16k | 852 | 693  (-19%) |

Swept ALPHA ∈ {0..16} and tried both head-pinned and head-spread layouts — none
got within ~12%. The reason is fundamental and is the real answer to "isn't
mirror hardcoded?":

**A program's runtime ≈ (K-block compute) + (per-tile fixed overhead × tile
count) — two independent cost dimensions.** To equalize makespan you must
equalize *both* the K-block sum *and* the tile count across programs. A
contiguous cumulative partition only equalizes one weighted combination
(`blocks + ALPHA·count`): for a triangular profile, heavy programs unavoidably
get few tiles and light programs many, and a single ALPHA knob cannot flatten
both dimensions at once (small ALPHA balances blocks but light programs drown in
per-tile overhead; large ALPHA balances count but heavy programs do too much
compute).

**Bundling a heavy tile with a light tile is the unique simple construction that
makes every unit identical in BOTH dimensions simultaneously** (constant count =
2, constant blocks = num_m+1), so both balance for free. So mirror's "exactly 2"
is not arbitrary — it is the *minimal constant-cost bundle*, and constant-cost
bundling (Step 10, greedy-bundle for arbitrary costs) is the principled general
form. The bundling is the point, not an artifact. Kept the partition kernel as a
documented negative result. All 12 correctness cases pass.

## Step 12 — Learnings from ROCm/aiter Lean Attention (StreamK)

Studied `aiter/ops/triton/_triton_kernels/attention/lean_atten.py` (ROCm/aiter),
a production AMD Triton **StreamK** ("Lean") attention. It independently confirms
most of our design and points at the one technique we're missing.

**What it validates (we landed on the same things):**
- **Ping-pong tile order == our mirror/fold.** `find_group_pingpong` orders a
  head's m-tiles as `where(i%2==0, i//2, num_m-1-i//2)` → 0, N-1, 1, N-2, … i.e.
  interleave light+heavy. It keeps a `find_group_sequential` variant but uses
  ping-pong — exactly our Step 2/10 conclusion, from a production kernel.
- **XCD L2 remap == ours.** Its `remap_xcd` (tall/short XCDs, contiguous pids per
  XCD) is character-for-character our Step 3, and Lean computes its load-balance
  split **per XCD** ("relative to 1 XCD") — matching our "balance *within* an XCD,
  never globally" finding (Step 5: global scatter = ~520 TF).
- **Bounded `static_range` + `if iter < end` guard**, not a `while` loop —
  the same persistent-loop shape we were forced into (TLX pipelines `scf.for`,
  not `scf.while`; our Step 9 dynamic kernel).
- **exp2 + NaN-safe masking** for all-`-inf` rows (`where(m_ij==-inf, …)`), the
  base-2 softmax we use; the NaN guard we skipped (Step 6) becomes *necessary*
  once tiles are split (a partial tile can see only masked K-blocks).

**What we're missing — StreamK K-block-level splitting (the real generalisation):**
Step 11 concluded a tile is "atomic" so cumulative-cost partition can't balance
the heaviest tile. StreamK dissolves exactly that limit: it flattens *all* causal
K-block work into one 1-D stream of "lean tiles" and gives each of the fixed
`total_programs` WGs a **contiguous equal-length slice** (`max_tiles_per_wg` for
the first `high_load_wgs`, one fewer for the rest) — perfect balance to within
one K-block, regardless of cost profile / ragged batch / partial blocks. The
price is that a WG may own a *partial* output tile, so it:
  1. carries partial online-softmax state `(m, l, acc)`;
  2. non-"host" WGs write partials to scratch `Mp/Lp/Op` + set a `lock`;
  3. the "host" WG (owns the tile's first lean tile) spins on locks and merges
     partials with the standard `m_new=max; rescale by exp2` reduction.

This is the principled answer to "balance arbitrary shapes": split at K-block
granularity + a cross-WG fixup, instead of bundling whole tiles.

**When it matters (and when it doesn't for us):**
- StreamK wins in **low-parallelism** regimes — decode (`num_m==1`), short
  sequences, or whenever `num_m * heads < #CUs` — where mirror/fold has *nothing
  to balance* (≤1 tile per head) and our kernels would tank or fall back. There
  StreamK splits a single head's K across many CUs.
- For our **large-prefill** benchmark (N=8k/16k, 64 heads) there is already
  abundant tile parallelism, so the StreamK reduction overhead (extra global
  Op/Mp/Lp traffic + lock spins) is pure cost — mirror/fold should stay ahead.

**Takeaway / future work:** add a StreamK variant aimed at the low-parallelism
regime (decode / short-seq / small grid), keeping mirror/fold for large prefill;
it needs the partial-state reduction (scratch buffers + locks + `.wt`/`.cv`
cache modifiers + `debug_barrier`) and NaN-safe masking for split tiles.

## Step 13 — StreamK split + auto-dispatch  (landed)

Implemented the Step 12 plan: a StreamK causal kernel
`streamk_causal` (`_attn_fwd_streamk_causal`) plus an occupancy-based
auto-dispatcher `causal_auto`.

Design (plain Triton, not TLX async — like aiter's lean_atten):
- Flatten an XCD's causal K-block work into a 1-D stream (`tiles_per_head =
  MASKED_BLOCKS·num_m(num_m+1)/2`, ping-pong tile order via
  `_find_group_pingpong`) and cut it into equal contiguous slices, one per
  resident program (`max_tiles_per_wg` / `high_load_wgs`) — balanced to ±1
  K-block.
- A program may own a *partial* tile → keeps partial `(m,l,acc)`; non-host
  programs write `Mp/Lp/Op` + raise a `lock` (`.wt` stores + `debug_barrier`),
  the host program (owns a tile's first K-block) spins on locks (`.cv` volatile
  loads) and merges partials with the max-rescale-by-exp2 reduction. NaN-safe
  masking (Step 12) for partials that see only masked K-blocks.
- Heads pinned to XCDs (`HEADS_PER_XCD`) via `_remap_xcd` for L2 locality.
- **Deadlock constraint**: TOTAL_PROGRAMS must be co-resident (≤ #CUs), else the
  host spin-wait hangs. Launch `(cu//NUM_XCDS)·NUM_XCDS` programs.
- Bounded loops only (`while iter < cta_end`, plain Triton supports it here;
  TLX's async-only `scf.for` limitation from Step 9 doesn't apply — no tlx).

Results (D=128, causal):
| Config | mirror | StreamK | auto picks |
|---|---|---|---|
| H=8,  N=512   | 12.0 | **16.2** | StreamK (+35%) |
| H=8,  N=1024  | 34.7 | **57.1** | StreamK (+65%) |
| H=8,  N=2048  | 94.3 | **187.1**| StreamK (+98%) |
| H=8,  N=4096  | 227.7| **401.7**| StreamK (+76%) |
| H=8,  N=16384 | **854** | 595 | mirror |
| H=64, N=2048  | **578** | 449 | mirror |
| H=64, N=8192  | **801** | 526 | mirror |
| H=64, N=16384 | **856** | 499 | mirror |

Clear complementary split: **StreamK wins up to ~2× when output tiles < CUs**
(decode / short-seq / few heads — exactly where whole-tile mirror/fold leaves
CUs idle); **mirror wins ~1.6× with abundant tile parallelism** (bigger BLOCK_M,
no reduction traffic, no lock spins). The auto-dispatcher (`est_tiles = B·H·
ceil(N/256) < #CUs → StreamK`) picked the winner in every benchmarked config.
All correctness cases pass (streamk + auto, 12/12 each).

Caveats / scope: causal MHA, non-ragged, `N % BLOCK_M == 0` and
`B*H % NUM_XCDS == 0` (else both fall back to mirror); GQA / ragged batch / true
decode (BLOCK_M=1) not wired up (aiter's kernel covers those).

## Future work

- Extend StreamK to GQA, ragged batch, and true single-token decode (BLOCK_M=1).
- Tune the `causal_auto` threshold (currently 1 wave) per arch; sweep the
  StreamK/mirror crossover.
- Finer-grained diagonal tiling (split the diagonal block into a half-size
  sub-tile) could recover the last few % but adds complexity/branches.
- Greedy-bundle (vs the linear fold) for non-linear/irregular cost profiles.
- Fix the compiler `iota_range` crash for the modulo decode at GRID_M==1
  upstream so the constexpr special-case isn't needed.
- Re-tune num_warps/waves_per_eu now that scheduling is balanced.
