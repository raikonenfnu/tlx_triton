# Nightly deployment

Use a dedicated benchmark checkout and virtual environment. Build TLX from source with the
repository's build instructions. Configure AITER in a separate environment when necessary.
Follow [AITER setup](../backends/aiter/setup.md) and supply its runtime override through `runtimes` in the deployment configuration.
Install and authenticate `codex` or `claude` for the selected provider and `gh` for publication.
Check the installed CLI flags against `coding_agents.py`. Do not put tokens in the JSON file.

Create an absolute-path configuration outside the source checkout:

```json
{
  "python": "/path/to/venv/tlx.venv/bin/python",
  "pythonpath": "/path/to/triton/python:/path/to/torch/site-packages",
  "targets": ["/path/to/triton/third_party/tlx/tools/agents/kernel_scout/targets/amd/gfx950/varlen_f16_fa.json"],
  "state": "/path/to/scout-state",
  "repository": "raikonenfnu/tlx_triton",
  "gpu": 0,
  "provider": "codex",
  "aiter_search": false,
  "rounds": 2,
  "max_candidates": 6,
  "agent_timeout": 300,
  "worker_timeout": 600,
  "context": "/path/to/megabrain/context.md"
}
```

The repository is configurable. check that the configured account can create/update issues
there. A classic token can fail on facebookexperimental even when public reads work.
Use a persistent, project-scoped context pack. Regenerate it when project knowledge changes.
Each discovery invocation receives this context explicitly. The controller owns memory updates.
The deployment runner does not publish Megabrain records automatically.

Preview each job first:

```bash
python deploy/run.py --config /path/to/scout.json --mode weekly
python deploy/run.py --config /path/to/scout.json --mode nightly
```

Run the first weekly job manually with `--execute`, inspect correctness, search coverage, and
reproducer commands, then review the resulting issue. Enable cron only after that run passes.
Add entries with `crontab -e`. The application does not install a schedule itself:

```cron
CRON_TZ=UTC
30 0 * * 1 /path/to/python /path/to/kernel_scout/deploy/run.py --config /path/to/scout.json --mode weekly --execute >> /path/to/scout-weekly.log 2>&1
30 1 * * * /path/to/python /path/to/kernel_scout/deploy/run.py --config /path/to/scout.json --mode nightly --execute >> /path/to/scout-nightly.log 2>&1
```

Check your cron implementation's CRON_TZ support, or use a host configured for UTC.
Supply PATH for codex/claude/gh, authentication configuration, and the required library environment
in the service account. The jobs take a nonblocking per-user physical-GPU lock and state lock. An overlapping job fails visibly. If weekly work exceeds one hour, move the nightly time later.
Monitor nonzero exits and logs. No report is published after a failed measurement.

Weekly measures all registered backends and freezes controls. Nightly measures backends with
`cadence="nightly"`, then reuses matching weekly controls. A changed target/case selection,
Torch/driver runtime, device, backend registry, or dependency identity requires a fresh campaign.
The runner does not silently replace an existing weekly baseline.
AITER uses public operators by default, without LLM discovery.
Set `aiter_search` to true only for an explicit comparison of alternative AITER implementations.

For fresh source, provide an optional `refresh_command` argument list that invokes YOUR deployment
script to fetch/update a clean dedicated checkout and rebuild its environment. That script must
stop on dirty source or failed builds. Configure `refresh_timeout` in seconds. The controller
records the actual measured source/compiler hashes. It does not fetch, rebase, or install packages.
Keep old pinned source checkouts and run directories for frozen-baseline reproduction.

The runner measures first, reads a pointer written only after report completion, and then invokes
publication. Publication checks a new weekly issue before linking and closing earlier owned
issues. Nightly updates one dated comment. Re-running publish is idempotent. Unrelated issues
are excluded by campaign ownership markers. Raw logs, adapters, and reproducer requests remain
local. Archive run directories separately if reports need to be reproduced on another machine.
