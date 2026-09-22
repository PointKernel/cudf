---
name: improve-compilation-cudf
description: Reduce libcudf C++/CUDA build time, redundant template instantiations, compiler memory use, and binary size, using cuDF PR precedents and controlled measurement. Use for source-level compilation and code-size work, not runtime tuning.
---

Use this skill when the user asks to:
- speed up compilation of a libcudf translation unit (TU) or the whole library
- remove unused, duplicated, or redundant template / kernel instantiations
- reduce compiler peak memory or the size of `libcudf.so`, its `.nv_fatbin`, or object files
- review or write a PR whose main claim is a build-time or binary-size improvement

Companion skills: `build-test-cudf` (building), `perf-compare-cudf` (nvbench comparison),
`review-cudf` (general review rules). Detailed material lives in this directory:
[measurement protocol](references/measurement.md), [implementation patterns](references/implementation.md),
and [the PR reference grouped by mechanism](references/pr-history.md). Read only the parts
that match the measured bottleneck.

# Principles

- Start from measured compiler work and the current call graph, not from a technique.
- Preserve supported types, public behavior, GPU architecture coverage, stream and
  memory-resource contracts, and error behavior.
- Historical PRs provide candidate techniques and expected magnitudes, not proof that the
  same change helps today. Separate a PR author's historical numbers from measurements
  reproduced now, and an issue estimate or unmerged prototype from a merged result.
- Preserve the chosen baseline: do not merge another branch, switch toolchains, or enable
  compiler caches as a side effect of applying a build recipe.
- Drafting or using this skill does not authorize commits, pushes, or PRs.

# Metrics

Report these separately. They pull in different directions, so state which one a change
targets:

| Metric | What it is | Notes |
|---|---|---|
| Slowest TU | Compile time of the single longest TU | Bottleneck indicator used by most 2026 PRs (#21804, #23285, #23320, #23322, #23330, #23343, #23448). It bounds the build under high parallelism but is not the full dependency critical path or measured wall time. |
| Parallel wall time | `ninja` elapsed at a fixed job count | The number CI's build-metrics report shows. |
| Aggregate work | Sum of per-TU compile times | Reviewers ask for it when a split may only redistribute work (#23322). |
| Compiler peak memory | RSS of the worst nvcc / cicc / ptxas process | CI builds have OOM'd on concurrent large TUs; #10756 cut peak RSS 14.6 GB to 2.4 GB. |
| Incremental rebuild scope | Number of TUs a header edit recompiles | Header splits and `.hpp` conversions target this (#20166). |
| Object / library size | Affected `.cu.o` bytes, `libcudf.so`, `.nv_fatbin`, packaged artifact | Fatbin is about 70% of `libcudf.so` (#23419). Duplicate kernel instantiations across TUs inflate it. |

Splitting a TU shortens the slowest TU but can duplicate device code and grow the binary
(#23322 doubled its object size; #23733 re-merged files an earlier PR had split).
Co-locating callers of one kernel family shrinks the binary but can recreate a long TU.
Measure both effects; neither is a universal rule.

# Process

## 0. Discover prior work

Prior PRs are the precedent reviewers hold a change to, and the list grows every release.
The PR reference in this directory is a starting point; refresh it for the target files:

```bash
R=NVIDIA/cudf
for q in "build time" "compile time" "binary size" "instantiation" "fatbin" "libcudf.so" "noinline"; do
  gh pr list --repo $R --state merged --search "$q" --limit 100 \
    --json number,title,mergedAt -q '.[] | "\(.number)\t\(.mergedAt[:10])\t\(.title)"'
done | sort -u -t$'\t' -k1,1n
gh pr list --repo $R --state open --search "build time OR binary size OR instantiation" --limit 50
gh pr list --repo $R --state merged --search "<path or symbol of the target TU>" --limit 50
```

Then read the tracking issues, which hold ranked slow-TU lists and measured runtime
trade-offs:

- #21973 "cudf C++ Compilation Time Optimization Report" (compile time; closed PR #21974
  carried the experiments and a "header cost only" list of TUs that do not benefit from
  splitting)
- #23419 "Reduce libcudf fatbin size by deduplicating equivalent CUDA kernel instantiations"
  (binary size)

For any PR touching the same files as the planned change, read its body, thread, and
inline review comments (`gh pr view N --comments`, `gh api --paginate repos/NVIDIA/cudf/pulls/N/comments`)
for numbers, setup, and choices made deliberately for compile time (co-located files,
custom kernels, duplicated headers, inline decisions), so they are not undone without
re-measuring. Check whether the historical optimization already landed or was superseded.

## 1. Establish the actual problem

1. Inspect the branch, working-tree changes, current build configuration, and repository
   instructions. Do not resume unrelated experiments found in another worktree.
2. Identify which metric the user wants improved and say so.
3. Rank current offenders (step 3). A tiny `.cu` can instantiate a huge template graph;
   source line count is not a cost estimate. Distinguish repeated header parsing from
   instantiation and optimizer cost before choosing a fix.
4. Map the expensive operation's instantiation dimensions: value / index / output types,
   aggregation kind, nullability, nestedness, comparator and hash type, join kind, window
   kind, NaN and keep policy, sort stability, iterator type, launch policy. Trace each
   dimension to callers and guards. Write down which combinations are unreachable,
   equivalent, duplicated, or necessary.
5. Capture nvbench "before" results for the affected benchmark now, before editing.

## 2. Measurement environment

Compile-time numbers are only meaningful with the compiler cache off and a fixed nvcc
thread count. Use a dedicated build directory so the devcontainer wrappers
(`build-cudf-cpp`, `configure-cudf-cpp`) cannot silently re-enable sccache:

```bash
cmake -S cpp -B cpp/build/buildtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER="" -DCMAKE_CXX_COMPILER_LAUNCHER="" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CUDA_FLAGS="--threads=1" \
  -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
  -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
grep -q 'COMPILER_LAUNCHER:STRING=/' cpp/build/buildtime/CMakeCache.txt && echo "sccache still enabled"
grep -cE 'sccache|ccache' cpp/build/buildtime/build.ninja   # must be 0
```

- Base and candidate must share toolchain, dependency revisions, architectures, build
  mode, flags, and job count. Do not change CMake options between them.
- Single-arch (`NATIVE`) is fine for iteration. Size claims about `libcudf.so` must either
  be re-measured on the RAPIDS arch list or be labeled a single-architecture prototype.
- A cached build used only to obtain a binary is not compile-time evidence.
- Full protocol, paired runs, and pitfalls: [measurement.md](references/measurement.md).

## 3. Rank offenders

Per-TU compile time and object size come from ninja's log. Run the parser from the build
directory: it reads object sizes relative to the working directory and reports zero bytes
from anywhere else.

```bash
ninja -C cpp/build/buildtime cudf -j"$(nproc --ignore=2)"
mkdir -p reports/base
( cd cpp/build/buildtime &&
  python3 ../../scripts/sort_ninja_log.py .ninja_log --fmt csv  > ../../../reports/base/compile.csv &&
  python3 ../../scripts/sort_ninja_log.py .ninja_log --fmt html > ../../../reports/base/compile.html &&
  cp .ninja_log ../../../reports/base/ninja.log )
sort -t, -k1 -nr reports/base/compile.csv | head -25            # columns: time_ms,size_bytes,file
```

- Snapshot right after the `cudf` target finishes; the parser resets when timestamps go
  backwards, so building tests afterwards pollutes the log.
- CI publishes the same report per PR at
  `https://downloads.rapids.ai/ci/cudf/pull-request/<PR>/<sha>/cuda<ver>_<arch>.ninja_log.html`
  with parallel build time and `libcudf.so` size (`build.sh --build_metrics` locally).
- Peak memory for one TU: re-run its command from `compile_commands.json` under
  `/usr/bin/time -v` and read "Maximum resident set size".

Binary size, whole library plus affected objects:

```bash
stat --format='%s %n' cpp/build/buildtime/libcudf.so
readelf --section-headers --wide cpp/build/buildtime/libcudf.so | awk '$2 == ".nv_fatbin" { print strtonum("0x" $6) }'
stat --format='%s %n' cpp/build/buildtime/CMakeFiles/cudf.dir/src/<dir>/<file>.cu.o
```

Kernels instantiated identically in more than one TU (the #23419 / #23706 / #23733 class):

```bash
cuobjdump --dump-sass cpp/build/buildtime/libcudf.so \
  | awk '/^code for /{arch=$3} /Function :/{fn=$3; count[arch"\t"fn]++} END{for (k in count) if (count[k]>1) print count[k]"\t"k}' \
  | sort -nr | head -40
nm --defined-only --size-sort --demangle -C cpp/build/buildtime/libcudf.so | tail -40
```

Compare kernel identities within the same architecture; repeated images across supported
architectures are intentional. PR #24258 adds `cpp/scripts/report_libcudf_binary_size.sh`
wrapping these; use it if merged.

## 4. Choose a mechanism that fits the evidence

| Observed cause | Candidate change | Precedent |
|---|---|---|
| Dispatch does not use the selected C++ type | Remove the dispatch; keep exceptional preprocessing such as dictionary key normalization | #23282 (68 s to 24 s) |
| Cartesian product of independent dispatch dimensions | Stage dispatches, derive intermediate types (`common_type` pairwise), normalize index / offset representation with `indexalator` / `offsets_iterator` | #10756, #6457, #6727 |
| Unsupported type or aggregation combinations instantiate heavy code | `if constexpr` / `requires` guards, or a local dispatcher whose id-to-type map collapses unsupported ids to a sentinel handled by `CUDF_UNREACHABLE` | #23330 (640 s to 127 s), #17753, #11489 |
| A policy flag duplicates an expensive algorithm | Runtime flag, `nullate::DYNAMIC`, optional iterator instead of pair iterators, `is_valid_nocheck` behind a runtime `has_nulls` | #6835, #9623, #9324, #21312 |
| Same implementation compiled in callers and in explicit-instantiation TUs | Pick one ownership scheme: delete the redundant TUs, or make callers include the declaration-only header | #23323 (removed 1,455 s TU), #23331 (73 s to 18 s, 1.4 MB off .so) |
| Equivalent kernels emitted from several TUs | Out-of-line non-template helper defined once (`make_offsets_child_column(device_span)`, `make_strings_column(device_uvector)`), or co-locate the wrappers that instantiate one kernel family | #23706 (38 MB off .so), #23733, #23420 |
| One large TU dominates the parallel build | Split along a compile-time axis into declaration `.hpp` + `_impl.cuh` + small per-instantiation `.cu`; keep heavy definitions out of common headers | #21804 (1,993 s to 262 s), #23322, #23343, #17089, #10671, #18948 |
| Recursive row operators expand inside CUB / cuco / thrust kernels | Transform into a `device_uvector` then a trivial reduction; precompute hashes or window bounds; call the comparator through a `__noinline__` device-pointer indirection; trivial `CUDF_KERNEL` instead of a CUB transform with a heavy functor | #12900, #23320, #23322, #23448 (2,899 s to 55 s), #23343, #21793 |
| Device code recomputes metadata known on the host | Precompute compact metadata on the host and pass it in, preserving alignment and lifetime | #17234 |
| Excessive parsing, or host-only code compiled as CUDA | Split fat headers, `.cuh` to `.hpp` where no device code, PIMPL so detail TUs become `.cpp`, `gather.hpp` over `gather.cuh`, dedicated CUB headers, `#pragma once`, object library for test utilities | #20166, #9299, #20491, #21804, #21349, #18925, #18131 |
| Anonymous namespace in a header | Remove it; every includer gets its own copy of the symbols | #22418 |

Choose one small, reviewable change with an explicit hypothesis. Worked examples for each
row are in [implementation.md](references/implementation.md). Constrained dispatch
example (#23330):

```cpp
struct unsupported_type {};
template <cudf::type_id Id>
struct dispatch_supported_type {
  using type = cuda::std::conditional_t<Id == type_id::LIST or Id == type_id::STRUCT or Id == type_id::DICTIONARY32,
                                        unsupported_type, cudf::id_to_type<Id>>;
};
cudf::type_dispatcher<dispatch_supported_type>(type, functor{}, args...);   // functor: if constexpr unsupported -> CUDF_UNREACHABLE
```

Existing dispatch helpers: `dispatch_storage_type` (decimals to rep), `row::primitive::
dispatch_primitive_type` (numeric only), `cudf::detail::dispatch_bool` and `dispatch_enum`
in `cpp/include/cudf/detail/utilities/dispatchers.hpp` (#20927) to write the dispatched
set down in one place. nvcc time scales with instantiation requests even when the code is
discarded later, so pruning unreachable branches pays even if they never execute (#21973).

Split structure that reviewers accept: `foo.hpp` declarations, `foo_impl.cuh` or
`foo.cuh` template definitions, one 20 to 50 line `foo_<variant>.cu` per instantiation
holding a single `template ... f<ConcreteArgs>(...);` or one non-template overload, shared
aliases in one `helpers.cuh`, every file added alphabetically to `add_library(cudf ...)`
in `cpp/CMakeLists.txt`, and "no logic change" stated when true. Hoist non-template pieces
into plain functions (`CUDF_HIDDEN` if internal) compiled once.

## 5. Correctness boundaries

- **Reachability first.** Prove an excluded combination is rejected or handled before the
  affected kernel. A runtime early return does not prevent instantiation, and a runtime
  `if (is_primitive_row_op_compatible(...))` inside an instantiation TU still compiles both
  branches (#18896). Do not remove a public specialization because no internal caller uses
  it. Preserve empty-input validation and error behavior.
- **Restrict locally.** Shared-memory aggregation support is narrower than global-memory
  support; follow the decomposition and compatibility checks on the current branch before
  narrowing a dispatcher, and do not copy an old aggregation list. #21973 recorded the same
  constrained-dispatch idea improving shared-memory groupby 45 to 59% and regressing the
  global-memory variant 37 to 54%. Preserve dictionary, decimal, nested, count, and
  overflow semantics wherever supported.
- **Make ownership explicit.** Document the host entry point, the TU that owns each
  definition, explicit specializations, CMake sources, and consumers. Under CUDA
  whole-program compilation keep a kernel launch with its device definition and expose a
  host launch wrapper across TUs (#16603, #24258). `extern template` is not a general
  device-linking solution; verify which instantiations it suppresses and what is still
  parsed and inlined. Check exact template parameters, including cooperative-group size
  (#19518 re-land).
- **Prefer existing precompiled overloads.** A resident gather map or strings-size buffer
  may already fit a span overload (#23420, #23706). Do not materialize a lazy iterator just
  to reach one without measuring the extra allocation, pass, and traffic; materializing at
  dispatch boundaries regressed tdigest 25 to 49% (#21973).
- **Keep comparisons exact.** Preserve null and NaN equality, signed-zero and hash
  consistency, dictionary key normalization, nested child nulls, stability and keep
  policy, decimal scale, chrono units, and window boundary rules. Erase a representation
  only where these stay equivalent. Avoid per-element device dispatch in bandwidth-bound
  reductions (10 to 430% slower in #21973).
- **Treat inlining as an experiment.** Blanket Release `__noinline__` on row operators
  regressed sort and search (#12900, #21197). Debug-only `#ifndef NDEBUG` noinline fixed
  hangs (#21197, #22675); a targeted unconditional noinline on `n_table_comparator` was
  accepted after Release benchmarks (#22699); the streaming groupby comparator uses a
  device-pointer indirection (#23448); AST needed force-inlining for runtime (#9530). Do
  not generalize any of these, and do not alter CUB policies to mask a regression. Use the
  `__noinline__` spelling in CUDA code.
- **Account for scratch and lifetime.** Cached hashes, bounds, and device-resident
  comparators must survive asynchronous use on the correct stream and honor resource
  ownership. Include preprocessing passes and temporary memory in runtime comparisons.
  #23322 kept ordered nested hashing inline because materialization regressed that path.
- **Keep headers honest.** Include what is used directly, no reliance on transitive
  includes, no anonymous namespaces in shared headers (#22418), `.hpp` unless the header
  defines device code, dedicated `<cub/device/...>` headers over `<cub/cub.cuh>`.
- **CCCL algorithm swaps change codegen.** `thrust::tabulate` to `transform` disables
  CUB's vectorized transform kernel (#21793 review); `count_if` / `unique_copy` to
  transform-then-reduce cost 20 to 50% on the non-nested `unique` path (#12900). Benchmark.

## 6. Re-measure incrementally

Do not rebuild from scratch. Re-run configure only if `CMakeLists.txt` changed, delete
the affected objects (all replacement TUs on the candidate side, all removed objects on the
base side), rebuild, and diff against the baseline log:

```bash
cmake -S cpp -B cpp/build/buildtime          # only if CMakeLists.txt changed; ninja also re-runs it automatically
rm -f cpp/build/buildtime/CMakeFiles/cudf.dir/src/<dir>/<changed_or_new>.cu.o
ninja -C cpp/build/buildtime cudf -j"$(nproc --ignore=2)"
mkdir -p reports/candidate
( cd cpp/build/buildtime &&
  python3 ../../scripts/sort_ninja_log.py .ninja_log --fmt csv > ../../../reports/candidate/compile.csv &&
  python3 ../../scripts/sort_ninja_log.py .ninja_log --fmt html --cmp_log ../../../reports/base/ninja.log > ../../../reports/candidate/compare.html )
grep -E '<dir>/' reports/candidate/compile.csv
```

Record for the affected set: slowest TU, summed time, summed object bytes, and, for size
work, `libcudf.so` plus `.nv_fatbin` and the SASS duplicate count, all before and after.
Repeat paired runs when variation could explain the result. Confirm the exported symbol set
is unchanged for size work:
`nm -D --defined-only libcudf.so | awk '{print $3}' | sort` on both builds.

## 7. Validate behavior and runtime

- Run the gtests covering the touched code with
  `ctest --test-dir cpp/build/buildtime -R <PATTERN>`, including the removed dispatch
  dimensions and fallback paths.
- Benchmark whenever device instructions, launch structure, dispatch, materialization, or
  inlining change. Idle GPU, fixed clocks where possible, small and large sizes, the real
  null / nested / type variants, and preprocessing included. Compare to the "before" JSON
  from step 1 and report how many cases were compared. `perf-compare-cudf` describes the
  workflow.
- Investigate any repeatable regression, including small percentages with large absolute
  cost; a fixed threshold is a reporting aid, not proof of noise. If a case regresses,
  restore the original code for that path or drop the change.

## 8. Write the PR

```
## Description
<One sentence: which TU / kernel set, what technique, part of #21973 or #23419 if so.>
<Diagnosis: which instantiation dimension or inlined functor caused the cost, and why the
 removed combinations are unreachable or equivalent.>
<What changed, file by file for splits; ownership of definitions; "No functional change" when true.>

Measured on <CUDA x.y>, <arch(s)>, Release, compiler cache disabled, <N> jobs:
- Slowest CUDA compile: <before> s -> <after> s
- Summed compile time for affected TUs: <before> s -> <after> s
- Affected object size: <before> MB -> <after> MB
- libcudf.so / .nv_fatbin: <before> -> <after> bytes (<delta>)   # size PRs, state arch list
- Benchmarks: <suite>, <N> cases, <max delta>% on <GPU>; tests: <N> passing
- Unresolved limitations / trade-offs: <...>
```

A fuller table-based template is in [measurement.md](references/measurement.md).
Reviewer expectations seen repeatedly: both slowest-TU and aggregate numbers when
splitting; object or `.so` growth explained; a code comment wherever a shape was chosen
for compile time so it is not refactored away; trailing return types on device lambdas;
`cuda::stream_ref` or `rmm::cuda_stream_view` plus `rmm::device_async_resource_ref` on new
signatures; `pre-commit run --all-files` clean.

# Anti-patterns

- Splitting a file from the "header cost only" list in #21973; the time is include
  parsing and splitting duplicates it.
- Declaring victory on the slowest TU while summed object size doubles, without saying so.
- Adding `extern template` in a header while the consumer still includes the definitions.
- Templating two near-identical kernels on comparator type when that drags both comparator
  header graphs into each TU; measured duplication was cheaper in #18896, but bring numbers.
- Applying a technique because it worked elsewhere, without mapping this TU's dimensions.
- Re-running the full baseline after every edit; incremental rebuild of touched objects is
  the measurement.
