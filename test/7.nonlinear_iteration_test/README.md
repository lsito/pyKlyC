# 7. Nonlinear iteration comparison

Use the same workflow as tests 1-6:

1. Run **nonlinear_iteration_test.ipynb**: edit one visible case, build the Python
   objects, solve, inspect the plots, save `inputs.mat` and `python_results.mat`.
2. Run **nonlinear_iteration_test.mlx** in MATLAB. The matching sectioned `.m`
   is its readable source. MATLAB loads only raw inputs, performs its own
   initialization and nonlinear solve, then saves `matlab_results.mat`.
3. Run **compare_results.ipynb**: compare saved arrays and iteration decisions,
   check the efficiency-residual formula, and overlay the results.

The default is the five-cavity 1 GHz fixture, 180 kV / 16 A modulated entrance
beam, Nt=64, Ns=3, Nz=64, factor=0.1, Nlimit=30. Inputs are deliberately visible
in the notebook; there is no helper runner, animation framework or case matrix.
The default is not a full reproduction of the manual's resolution/settings.
Change one input at a time and rerun all three notebooks. Each export writes
a new run ID and overwrites this folder's input/Python files; an older MATLAB
result is rejected. One selected case is stored at a time.

## MATLAB source and adapters

The Gaussian branch is copied from KlyC_v6.mlapp, as in test 5. Its GUI values
are supplied by a small struct. Generated maps are kept in memory instead of
being saved/reloaded as TXT. Original constants and arithmetic are retained.

The readable script directly contains KLJCmainsv22.m initialization lines
45-209, nonlinear calculations 210-534, and decisions 601-626. The source is
visible in the notebook, not dynamically evaluated. Documented test changes:

- Remove the original stray `N` in `Nesz=zeros(length(zi),2)N;`.
- Join the split `.*` operator at original lines 322-323.
- Use serial `for` instead of `parfor` for prefill/global space charge.
- Replace the field-map TXT load with the independently generated maps.
- Record histories and preserve the SC DC field before MATLAB adds Emz0.
- Omit original GUI and ramp-power reporting lines 535-600; the case is an
  ordinary, fixed-power solve. Calculation and stopping decisions are retained.

No original KlyC source file or Python solver is changed. The `.mlx` is generated
from the `.m` with MATLAB; regenerate it after editing the readable script.
Run All in the saved Live Script; add this folder to the MATLAB path if running
it without an active editor tab. A missing input error gives the expected path.
The original enabled-prefill dN=1 case is explicitly excluded.

## What agreement means

The comparison checks initialization, complex circuit solutions, final beam
and field arrays, particle histories, source accumulators and iteration history.
Only declared vector orientations and MATLAB's one-based Y indices are adapted.
Finite values use rtol=1e-8, atol=1e-10; iteration/index decisions are exact.
Unwritten trajectory infinities must match. Near-zero beam efficiency makes
its relative residual ill-conditioned, so each engine's residual is checked
against its own efficiency history and SC-switch reset; efficiency itself is
still compared across languages.

Field names: Esz/Esz0 = SC; Emz/Emz0 = cavity; Esmz/Esmz0 = total tracking field.
The current original-source model leaves Emz and Emz0 zero. Agreement tests
translation fidelity, not physical cavity-induced gain. No field-feedback
repair is inserted in either implementation.
