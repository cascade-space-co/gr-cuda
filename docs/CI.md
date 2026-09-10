# Continuous integration

`ci.yml` lints on every push. `gpu-qa.yml` builds gr-cuda and runs its tests on a
GPU runner.

## What runs, and when

| Event | GPU QA |
| --- | --- |
| Pull request from a branch in this repo | yes |
| Pull request from a fork | only with the `run-gpu-qa` label |
| Push to `cascade/main` or `cascade/main-gr-3.10.13` | yes |

GPU minutes are billed per minute, so a fork's pull request needs a maintainer to
apply `run-gpu-qa` before it can spend them; an outside contributor cannot apply
a label themselves. Cascade pull requests are not gated, because a label you have
to remember to add is one that gets forgotten.

To stop a run, cancel it from the Actions tab. Removing the label does not stop
one.

## GNU Radio is built from source

`cuda_buffer` derives from `buffer_double_mapped` and overrides
`on_transfer_type_set()`. Both extension points were added to GNU Radio after
3.10.12.0 — the subclassable constructor and its `defer_alloc_t` tag in
[`58365a1`](https://github.com/gnuradio/gnuradio/commit/58365a106601656109ff34c422f7ffbf666747a0),
the hook in
[`8c39821`](https://github.com/gnuradio/gnuradio/commit/8c39821dc5828fff11f775da8c6b00510a5c7d7a)
— so gr-cuda does not compile against any released version.

CI therefore builds GNU Radio from source at the commit in `GNURADIO_REF`, pinned
so that a red run means gr-cuda changed rather than GNU Radio. conda-forge's
package is installed first and then removed, purely to pull in the dependency
closure the source build needs.

`.github/ci-setup.sh` does all of that. When a new enough GNU Radio is available
as a conda package, that script and the caching around it can be deleted and a
run drops to a few minutes.

## The cache

The environment takes about 20 minutes to build and is roughly 2 GB stored,
against a 10 GB per-repository cap. A warm run takes about three minutes.

The key is a hash of `.github/ci-setup.sh` plus `GNURADIO_REF` — everything that
determines what the environment contains. Editing anything else, including this
file, does not invalidate it. `CACHE_VERSION` forces a rebuild by hand.

It is saved immediately after being built rather than at the end of the job, so a
run that gets cancelled part-way still banks the work.

## Things that will bite you

**Do not build GNU Radio with `-DENABLE_TESTING=OFF`.** It is the obvious saving
and it breaks every out-of-tree module downstream: GNU Radio templates that value
into the installed `GnuradioConfig.cmake`, which only requests Boost's
`unit_test_framework` when testing is on. Without it, anything calling
`GR_ADD_CPP_TEST` fails to configure.

**CUDA is capped below 13.1** to stay under the driver. A toolkit newer than the
driver fails silently: kernel launches are unchecked, so they return zeros and a
test reports a numeric mismatch rather than a CUDA error. If a CUDA test fails
with all-zero output, check the toolkit and the architecture first.

**`ENABLE_IBV` is pinned off.** Its default comes from `find_library(ibverbs)`,
so a runner image that started shipping rdma-core would quietly turn the IBV
blocks on with no hardware to exercise them.

## What CI does not cover

- **Other GPU architectures.** `CMAKE_CUDA_ARCHITECTURES=native` compiles for the
  runner's card and nothing else.
- **Performance.** The runner is far slower than the cards this is developed on.
- **Race coverage is a different sample, not a superset.** The runner has four
  cores; the timing-sensitive tests explore a different interleaving distribution
  there than on a workstation. Green in both places means more than green in one.
- **The IBV blocks**, which are not compiled.
- **GNU Radio's own test suite**, which is built but not run.
