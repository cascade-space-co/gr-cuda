# Continuous integration

Two workflows. `ci.yml` lints on every push on an ordinary runner. `gpu-qa.yml`
runs the QA suite on a real GPU, and is the subject of this document.

Measurements here were taken 2026-09-09 unless noted. Treat them as an order of
magnitude, not a contract.

## What runs, and when

| Event | GPU QA runs? |
| --- | --- |
| PR from a branch in this repo | yes, automatically |
| PR from a fork | only with the `run-gpu-qa` label |
| Push to `cascade/main` or `cascade/main-gr-3.10.13` | yes |
| Manual dispatch from the Actions tab | yes |

Documentation-only changes do not trigger it: `paths-ignore` covers `**/*.md`,
`docs/**` and `LICENSE`. It is a deny-list rather than an allow-list on purpose,
so a file type nobody anticipated still gets tested. Note that if `gpu-qa` is
ever made a required check, a run skipped this way reports nothing at all, and a
docs-only PR would never become mergeable.

Our own PRs are not gated on a label because a gate you have to remember to
open is a gate that gets left closed, and a forgotten label means a CUDA
regression merges unexercised. Fork PRs are gated because GPU minutes bill to
the organisation and a stranger should not be able to spend them unasked; a
stranger also cannot self-apply the label, and GitHub's first-time-contributor
approval sits in front of that again.

This is a cost control, not a security boundary. The job holds no secrets, its
token is read-only, and it uses `pull_request` rather than
`pull_request_target`, so fork code never sees credentials.

To stop a run, use the Cancel button in the Actions tab. Removing the label does
not stop one — `unlabeled` is deliberately not a trigger, because a run entering
the concurrency group cancels the in-flight one whether or not its own job then
runs, which would mean removing any label from one of our PRs kills a GPU run
with nothing to restart it.

One dynamic worth knowing while the cache is cold: `cancel-in-progress` means a
superseded run contributes nothing, and the cache is not saved until roughly
twenty minutes in. So with an empty cache, two pushes less than a cold build
apart will keep it empty. Harmless in steady state, where a warm run reaches the
save step in seconds and there is an entry to fall back on regardless.

## The runner

`gpu-t4` is a **GitHub-hosted larger runner**, not a self-hosted machine. The
`runs-on: <custom label>` syntax looks like self-hosted, but the VM is
provisioned per job by GitHub's Hosted Compute Agent and destroyed afterwards.
Nothing persists between jobs, and there is no host to re-image.

| | |
| --- | --- |
| GPU | Tesla T4, compute capability 7.5, 16 GB |
| Driver | 590.48.01, supports CUDA 13.1 |
| Machine | 4 vCPU, 27 GB RAM, ~162 GB free, Ubuntu 24.04 |
| Image | NVIDIA GPU-Optimized for AI and HPC |
| Cost | $0.052/min, billed to the org, outside included minutes |

The image ships the driver but **no `nvcc` on `PATH`**; CUDA comes from conda.
Miniforge is preinstalled at `/opt/miniforge` and is **not writable by the
runner user**, which is why both the environment and `CONDA_PKGS_DIRS` live
under `/home/runner`.

Runner concurrency is two jobs at once, shared with gr-cascade.

## GNU Radio is built from source

conda-forge's newest gnuradio is 3.10.12.0, and `include/gnuradio/cuda/cuda_buffer.h`
carries a `#warning` explaining that ≤3.10.12 has a fan-out deadlock that hangs
this suite outright. So the conda package alone is unusable and CI builds
gnuradio from source, pinned to an exact commit in `GNURADIO_REF`.

Pinned rather than tracking a branch so that a red run means gr-cuda changed,
not gnuradio. Bumping it is a deliberate one-line PR that costs one cold
rebuild.

`environment.yml` still lists `gnuradio` and `gnuradio-build-deps`. That is not
redundant — they are installed for their dependency closure (boost, fftw, volk,
spdlog, pybind11) and then gnuradio itself is removed with
`mamba remove --no-prune-deps` before the source build. `--no-prune-deps` is the
load-bearing flag: it unlinks gnuradio while orphaning rather than
garbage-collecting the closure the source build needs. This mirrors what
`bootstrap.sh` does on europa.

**When gnuradio ≥3.10.13 reaches conda-forge**, all of this goes away: pin it in
`environment.yml`, delete `.github/build-gnuradio.sh` and the cache steps, and a
run drops to a few minutes. There is a `TODO(GR-3.10.13)` marking the spot.

## The environment cache

The conda environment with source-built gnuradio is ~5.8 GB on disk, ~2.2 GB
stored. GitHub's cap is 10 GB per repository, with least-recently-used eviction.

```
cold   ~21 min   builds gnuradio
warm   ~2 min    43s to restore the environment
```

Two things about it are easy to get wrong:

**The key covers the build recipe.** It hashes `environment.yml`,
`.github/build-gnuradio.sh` and `GNURADIO_REF`. The cached artifact is gnuradio
*as configured by that script*, so if the recipe were not in the key, changing a
`-DENABLE_*` flag would silently reuse an environment built with the old flags —
and keep doing so until the entry evicted, at which point CI's behaviour would
change on its own with no commit to blame.

This is also why the build lives in its own script rather than inline. An
earlier version hashed the whole workflow file, which was correct but far too
broad: editing a comment forced a 21-minute rebuild and stored another 2.2 GB,
and a handful of edits filled the cap with near-identical copies.

**The save happens before gr-cuda is built.** `cache/restore` and `cache/save`
are separate steps for this reason. The single `actions/cache` action saves in a
post-job step, by which point gr-cuda has been installed into the same prefix —
so every run would store an environment carrying that commit's gr-cuda and
restore it over the next one. It would look like a working cache while serving a
stale build.

## Traps

Things that look wrong, or that an obvious cleanup would break.

**Do not build gnuradio with `-DENABLE_TESTING=OFF`.** It is the obvious saving
and it breaks every downstream OOT. gnuradio templates the build-time value into
the installed `GnuradioConfig.cmake`, which appends Boost's
`unit_test_framework` to its required components only when testing is on.
With it off, anything calling `GR_ADD_CPP_TEST` — which this repo does — fails at
generate time with `Target "cuda_qa_seq.cc" links to: Boost::unit_test_framework
but the target was not found`. gnuradio's own tests are built here but never run.

**QA runs from a flat copy of `python/cuda`.** `python/cuda` is a package
directory named `cuda` with an `__init__.py`, so pytest names the test modules
`cuda.qa_*` and imports the parent package — which shadows NVIDIA's `cuda`
namespace. cupy then fails at `from cuda import pathfinder` and every file errors
during collection. Copying the files flat, without `__init__.py`, leaves no
`cuda` package to resolve to. `--import-mode=importlib` does *not* fix this; the
module naming causes it, not `sys.path`.

The five files that import siblings by bare name (`qa_add_cupy`, `qa_fanin_sync`,
`qa_multiply_const_cupy`, `qa_parallel_sync`, `qa_stress_sync`) work because that
is the fallback branch of their `try: from .add_cupy import ... except
ImportError:`. The suite still tests the installed build — the tests import
gr-cuda as `from gnuradio import cuda`, not from the source tree.

A consequence worth knowing: **plain `pytest` from a checkout does not work**.
Running a single file directly (`python python/cuda/qa_fft.py`) is unaffected.

**`--timeout-method=thread`.** It is pytest-timeout's default and the robust
choice: the deadline is enforced from a watchdog thread, so it fires even when
the main thread is wedged inside a C call that never returns to the interpreter.
The `signal` method raises through SIGALRM, which Python only delivers at a
bytecode boundary, so it cannot interrupt that case at all.

A wedged flowgraph is *not* that case, and it is worth recording why, because it
is easy to assume otherwise: `gr.top_block.wait()` runs the blocking wait on a
separate thread through `GR_PYTHON_BLOCKING_CODE` (`PyEval_SaveThread`, so the
GIL is released) while the main thread polls a `threading.Event` at 10 Hz. Either
method would fire there. `thread` is chosen for the cases where neither of those
holds.

The thread method takes the whole process with it rather than raising into the
test, so a hang aborts the run instead of failing one test. That is the right
trade for a gate, and pytest-timeout dumps every thread's stack before exiting,
so the run still tells you which test wedged. The invocation is wrapped in
`timeout(1)` as a last backstop for the case where nothing Python can run.

**No pytest-timeout settings in `pyproject.toml`.** `--strict-config` makes an
unrecognised ini key an error, and `timeout` is only recognised with the plugin
installed. In shared config it would break `pytest` for anyone whose environment
predates `environment.yml` — which, when this was written, was every environment
on europa. CI passes `--timeout` on the command line instead.

**CUDA is capped below 13.1.** europa's driver is 580.178.04 (supports CUDA
13.0) and the runner's is 590.48.01 (supports 13.1), so a 13.0 toolkit is under
both ceilings and one pin serves both. A toolkit newer than the driver fails
*silently*: launches are unchecked, so kernels return zeros and a test reports a
numeric mismatch rather than a CUDA error. **If a CUDA test fails with all-zero
output, check the toolkit and the architecture before anything else.**

**`-DENABLE_IBV=OFF` is explicit.** Its default is `find_library(ibverbs)`,
which would make CI's compiled surface a function of whatever the runner image
ships — an image adding rdma-core would silently start building blocks with no
NIC to exercise them. IBV compile coverage is deliberately out of scope.

## What CI does not cover

- **Blackwell codegen.** The runner is `sm_75`; europa is `sm_120`.
  `CMAKE_CUDA_ARCHITECTURES=native` compiles for the T4 and only the T4.
- **Performance.** A T4 behind PCIe 3 is far off the numbers in the README.
- **Race coverage is a different sample, not a subset.** 4 vCPU behind PCIe 3
  produces a different interleaving distribution than 32 cores behind PCIe 5.
  CI will catch races europa never hits and miss races europa hits routinely.
  Green on both is meaningfully stronger than green on either.
- **The IBV blocks.** Not compiled; the runner has no Mellanox NIC.
- **gr-cuda's C++ tests.** Built, never executed. Running them needs a targeted
  `ctest` invocation, because plain `ctest` also fires the Python tests through
  gnuradio's `GR_ADD_TEST`, which fails headless.
- **gnuradio's own suite.** Built, never run.

## Maintenance

**Bumping the gnuradio pin.** Change `GNURADIO_REF` in `gpu-qa.yml`. One cold
rebuild follows.

**Forcing a rebuild.** Bump `CACHE_VERSION`.

**Reproducing CI's gnuradio locally.** `.github/build-gnuradio.sh` is
self-contained; activate a conda environment built from `environment.yml`, set
`GNURADIO_REF`, and run it.

**A run that is slower than expected** is almost always a cache miss. The
`Environment size` step prints whether it hit, and the size against the 10 GB
cap.
