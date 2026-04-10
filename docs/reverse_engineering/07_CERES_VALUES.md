# Round 3 — Ceres Parameter Values & Residual Arity

**Objective**: Capture actual Ceres::Problem parameter block values, cost function arities, and confirm structural properties of the 3 Problems during a cold-start render.

**Status**: INCOMPLETE — Shim injection successful but zero function calls captured.

---

## Method

**Approach**: DYLD_INSERT_LIBRARIES macOS function interposition via C++ shim library.

- **Shim file**: `/tmp/ceres_full_shim.cpp` (176 lines)
- **Compilation**: `arch -x86_64 clang++ -arch x86_64 -shared -fPIC -std=c++17 -o /tmp/ceres_full_shim.dylib /tmp/ceres_full_shim.cpp`
- **Hooked functions** (Itanium ABI C++ names):
  - `_ZN5ceres7Problem17AddParameterBlockEPdi` (1-arg variant: void* ptr, int size)
  - `_ZN5ceres7Problem17AddParameterBlockEPdii` (2-arg variant: void* ptr, int size, parameterization)
  - `_ZN5ceres7Problem16AddResidualBlockEPNS_12CostFunctionEPNS_12LossFunctionEPd` (1–5 pb variants)
  - `_ZN5ceres7Problem21SetParameterLowerBoundEPd` / `SetParameterUpperBound`
  - `_ZN5ceres6Solver5SolveERKNS_10SolveOptionsEPNS_7ProblemEPNS_8SummaryE`
- **Output file**: `/tmp/ceres_trace_full.log` (line-buffered FILE*)
- **Execution**:
  ```bash
  DYLD_INSERT_LIBRARIES=/tmp/ceres_full_shim.dylib arch -x86_64 \
    /Users/ryaker/Documents/Light_Work/lri_process \
    /Volumes/Base\ Photos/Light/2019-08-28/L16_04574.lri \
    /tmp/output_round3.hdr
  ```
- **Runtime**: ~120 seconds; process hung after render completion (did not exit cleanly).
- **Fallback**: frida unavailable (not in brew, pip install blocked by system policy).

---

## Render Status

**Completed**: YES

- Output file: `/tmp/output_round3.hdr` created, size 50,331,648 bytes
- File type: Radiance HDR (magic bytes `#?RADIANCE`)
- Interpretation: Render executed successfully to completion despite zero logged Ceres calls

---

## Log Sample

```
=== Ceres shim initialized (pid 43036) ===
```

**Total lines**: 1 (only initialization message)

**Function calls captured**:
- AddParameterBlock (1-arg): 0 calls
- AddParameterBlock (2-arg): 0 calls
- AddResidualBlock (all arities): 0 calls
- SetParameterLowerBound: 0 calls
- SetParameterUpperBound: 0 calls
- Solver::Solve: 0 calls

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Shim initialization logged | ✅ YES |
| Render completion verified | ✅ YES (50MB HDR) |
| Parameter blocks captured | 0 |
| Residual blocks captured | 0 |
| Solver calls captured | 0 |
| Cost function pointers logged | 0 |
| Bound-setting calls captured | 0 |
| Execution time | ~120 seconds |
| Log file size | 53 bytes |

---

## Interpretation & Limitations

### What We Learned
1. **Shim loading confirmed**: Constructor executed and logged initialization message to disk, proving `DYLD_INSERT_LIBRARIES` successfully injected the shim into the process.
2. **Render succeeded**: Valid HDR output was produced, indicating the render pipeline executed to completion.
3. **Mangled names verified**: Symbol names were confirmed present in libceres.dylib via `nm` prior to compilation.

### Why Zero Calls Were Captured
Three hypotheses:

1. **Functions not called in this code path**: The render process may use a different entry point into Ceres (e.g., factory-provided Problem instances already constructed, or an alternate API not hooked).

2. **Inlining or compile-time optimization**: If libceres.dylib was compiled with aggressive optimization, hot-path AddParameterBlock / AddResidualBlock calls might be inlined or tail-called, bypassing the function pointers we hooked.

3. **Different calling convention or function variant**: The actual functions called might use a different Itanium mangling or signature (e.g., different parameter order, const qualifiers, overload resolution).

### Fallback Approaches Not Available
- **frida**: Not installed; `brew install frida` failed, `pip3 install frida` blocked by system policy.
- **lldb**: User constraint: "do not try to improvise with lldb again" (breakpoints have prohibitive overhead and interfere with Ceres' timer-based logic).
- **Source-level recompilation**: libceres.dylib and libcp.dylib are closed binaries; no source available.

---

## Conclusion

**Mission goal NOT achieved**: Unable to capture actual parameter block values, cost function arities, or confirm structural properties of the 3 Problems.

**Deliverable status**: Shim injection confirmed working; render execution confirmed complete; but the target Ceres API calls remain unobserved.

**Recommendation for next round**: 
- Investigate whether Problems are pre-constructed and passed in as arguments to higher-level functions (unlikely to be hooked without widening the API surface).
- Consider alternative dynamic tracing: `dtrace` (if macOS allows), `strace` on Linux cross-compile, or source-level rebuild of libceres if feasible.
- Revisit symbol table to confirm whether target functions exist in the actual runtime image (not just libceres.dylib static library).

---

**Report Date**: 2026-04-09
**Methodology**: DYLD_INSERT_LIBRARIES with C++ function interposition
**Status**: Failed to capture target function calls; render completed successfully
