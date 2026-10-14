# TLX tutorials

Search existing tutorial implementations for the exact operation and gfx950 target.
Read `third_party/tlx/tutorials/` and checked benchmark callers. Do not call a tutorial based
only on its file name. Keep the selected path, source commit, tile/warp parameters, and any
relationship to the public operation in the proposal rationale. An alias to an ops implementation
is valid for reproducing that variant but is not an independent algorithm.

For a16w16 GEMM, inspect `gfx9_gemm/a16w16/bench.py`: its VERSION_MAP, shape validation,
`load_matmul_module`, and `launch_tutorial_matmul` define the launch contract. Versions progress
from naive to warp_pipeline and beyond_hotloop. Early versions need larger/divisible tile shapes. Later versions may require K multiples of 128. Search supported versions and existing launch
knobs, preserving B layout and including output allocation. Compare the two default seed versions
before expanding to earlier versions or inter-wave implementations. Read the inter-wave
`matmul_kernel.py`: it imports the public ops kernel as a compatibility alias. Label that identity.
Check split-M, stream-K, reduction work, and scratch costs through their existing callers.

For dense attention, inspect `amd_fa_pipelined.py` and `amd_fa_cluster.py`. The pipelined wrapper
accepts a config with BLOCK_M, BLOCK_N, num_warps and PREFETCH. The seed checks simple and
prefetch modes. check valid LDS footprints and head dimensions before changing tiles/warps.
The wrapper signature is attention(q,k,v,sm_scale,causal,config=None). Do not copy arguments
from the public ops API. Inspect return layout and dtype. Include any required conversions.
Search cluster or async-pipeline variants only when their target features and shape limits match.

There is no known independent packed-varlen tutorial seed in the initial checkout. Search source
for a genuine packed THD/cumulative-length caller before proposing one. A dense kernel invoked
per sequence can be considered only if it belongs to the tutorial backend, respects GQA and
bottom-right masking, and includes all slicing, launches, and output packing in the timed call.
A loop over many short sequences may expose host overhead. That is part of the eager contract.
Do not import `tlx.ops.flash_attn_varlen` and rename it a tutorial. If no compatible implementation
exists, return no proposals. Unsupported is useful coverage information.

Use feedback to stop investigating invalid or slower variants. Do not edit tutorial source,
create new algorithms, or time calls in the discovery agent. The artifact adapter may select
existing variants and launch parameters. A tuning proposal must cite the existing source caller
that establishes its validity. The harness checks independent inputs and checks the selected
winner in a fresh measurement process.
