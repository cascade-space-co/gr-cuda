# Contributing to gr-cuda

Thanks for contributing! This guide covers the development setup, code style, and
the project-specific gotchas worth knowing before you add a block.

## Development setup

Build and test as described in the [README](README.md#build-and-install):

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -Wno-dev
ninja && ninja test
```

`ninja test` runs the full `ctest` suite (C++ Boost.UTF tests in `lib/qa_*.cc` and
Python `gr_unittest` tests in `python/cuda/qa_*.py`). Run a subset with, e.g.,
`ctest -R seq --output-on-failure`.

## Code style and pre-commit

Formatting and lint are enforced by [pre-commit](https://pre-commit.com/). Install
the hooks once, then they run on every commit:

```bash
pre-commit install
pre-commit run --all-files   # run manually across the tree
```

The hooks (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)):

- **ruff** / **ruff-format** — Python lint + formatting.
- **clang-format** — C++/CUDA formatting (`.clang-format`).
- **gersemi** — CMake formatting (`.gersemirc`). It needs the GNU Radio CMake macro
  definitions, so it is a local hook:
  `gersemi -i --definitions=$CONDA_PREFIX/lib/cmake/gnuradio <files>`.
- **check-bindtool-hashes** — verifies the pybind11 binding files are in sync with
  their headers (see below).

## Adding a new block

`gr_modtool` scaffolds the source files fine, and it correctly edits the
`install(...)`-style lists (the public header in `include/` and the GRC YAML in
`grc/`) and the Python QA list. But for **every** block — IBV or not — it
mis-handles the two `list(APPEND ...)` insertions (the `lib/` build sources and
the pybind binding list), and for feature-gated blocks it can't pick the right
section. See the [caveat below](#why-does-gr_modtool-mis-edit-the-cmakelists) for
why. Recommended workflow:

1. Scaffold the block and let `gr_modtool` make the edits it gets right:

   ```bash
   gr_modtool add -t <blocktype> <blockname>
   ```

2. Fix the two `list(...)` insertions it gets wrong, **by hand**:
   - confirm `<name>_impl.cc` ended up in the `cuda_sources` list in
     `lib/CMakeLists.txt` (it tends to land in `test_cuda_sources` or get stuffed
     into `add_library(...)` instead);
   - add `<name>_python.cc` to `cuda_python_files` in
     `python/cuda/bindings/CMakeLists.txt` (it is silently skipped).
3. Generate the pybind11 binding and refresh its header hash:

   ```bash
   gr_modtool bind <blockname>
   ```

4. Add tests: a C++ `lib/qa_<name>.cc` (append it to `test_cuda_sources` in
   `lib/CMakeLists.txt`) and/or a Python `python/cuda/qa_<name>.py` (register it
   with a `GR_ADD_TEST(...)` block in `python/cuda/CMakeLists.txt`).
5. For optional/feature-gated blocks only, move all of the block's CMake entries
   into the matching conditional (e.g. `if(ENABLE_IBV)`); `gr_modtool` always puts
   them in the unconditional section.
6. `pre-commit run --all-files` and `ninja test`.

### Why does `gr_modtool` mis-edit the CMakeLists?

`gr_modtool`'s CMake editor is pure regex text-substitution (it replaces only the
**first** match of a command and is blind to `if()/endif()` blocks). Two project
conventions trip it up:

- **`list_expansion: favour-expansion`** in [`.gersemirc`](.gersemirc) reflows
  `list(APPEND <var> ...)` onto multiple lines:

  ```cmake
  list(
      APPEND cuda_sources
      foo_impl.cc
  )
  ```

  `gr_modtool` anchors on `list(APPEND <var>` with `(` and `APPEND` *adjacent*, so
  it never matches the reflowed form. As a result, when adding a block it either
  falls back to stuffing the `.cc` into `add_library(...)`, lands it in the wrong
  list (e.g. `test_cuda_sources`), or — for pybind — **silently adds nothing**, so
  the binding is never compiled. This is chosen deliberately (per-line list items
  produce far fewer merge conflicts across branches), so prefer fixing CMake by
  hand over reverting the formatting.

- **Optional / hardware-gated blocks** live inside `if(ENABLE_IBV)` sections (in all
  of `lib/`, `include/`, `python/cuda/bindings/`, and `grc/` CMakeLists). `install(...)`
  lists (headers, GRC YAML) *are* matched, but always in the **first, unconditional**
  block — `gr_modtool` has no concept of "this block is optional." If your block is
  gated behind a feature flag, move its entries into the matching `if(...)` block
  manually.

## Submitting changes

- Keep commits focused; write a clear commit message explaining the *why*.
- Run `pre-commit run --all-files` and `ninja test` before opening a PR.
- New blocks should ship with QA coverage and a GRC YAML definition.
- See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the GPU stream contract that all
  blocks must follow.

## License

By contributing, you agree that your contributions are licensed under
GPL-3.0-or-later, consistent with the rest of the project.
