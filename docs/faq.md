# FAQ and troubleshooting

Answers to the questions that come up most often when running
`doubleml-pipeline` for the first time. Error and warning texts are quoted so
that you can search this page for them. For the concepts see
[Double Machine Learning in 10 minutes](concepts.md); for every parameter see
the [API Reference](api_reference.md).

## Running the pipeline

### How long does a run take, and how do I make a first run fast?

The cost is dominated by nuisance-model fits:

*   **Phase 1** fits one candidate per combination in
    `ml_models_y × ml_models_t × n_folds_list` (two learners on each side and
    one fold setting = 4 candidates). With the default
    `hparam_policy="global"`, FLAML searches hyperparameters **once per
    treatment and learner** for `time_budget` seconds; every fold then fits
    the winner with fixed hyperparameters.
*   **Phase 3** re-estimates the `top_n` shortlisted candidates `n_reps` times
    (Phase 1's result is reused as repetition 0).
*   **Phase 0** (saturation only) adds `2 × n_seeds × k_folds` fits per
    treatment on the selection sample (100 at the defaults).

All candidate and repetition jobs run in parallel on `n_jobs` cores (`-1` =
all cores). For a first run on a laptop, start small and scale up once the
configuration is right:

```python
from doubleml_pipeline.workflow import CausalWorkflowOrchestrator

orchestrator = CausalWorkflowOrchestrator(
    # ... your columns ...
    ml_models_y=["lgbm"],
    ml_models_t=["lgbm"],
    n_folds_list=[5],
    time_budget=10,
    output_dir="./smoke_test",
)
```

and call `orchestrator.run_full_pipeline(df=df, scaler=scaler, top_n=1,
n_reps=3)`.

### Do I need a GPU?

No. `n_jobs` controls the number of **CPU** cores; the nuisance learners
(LightGBM, XGBoost, …) and FLAML run on the CPU. A GPU is not used by Phases
1–4.

### How much data do I need?

There is no hard minimum, but a few thresholds in the pipeline are worth
knowing:

*   With fewer than 1,000 rows and CATE covariates the runner warns
    *"Sample size is small (n < 1000). Set degree of continuous covariates to 2
    or less."*
*   Phase 0 switches its nuisance learners to `RidgeCV` when the selection
    sample is smaller than `small_sample_threshold` (default 2,000 rows).
*   The cluster-key rule warns when a key has fewer than 30 clusters; cluster
    standard errors with very few clusters are noisy.

Keeping the panel at the region × product level (rather than aggregating to a
national total) is the usual way to get enough rows.

### Can I estimate several treatments in one run?

You can pass several columns in `d_cols` (with one `x_cols_list` entry per
treatment), and Phases 1–4 estimate each of them. **Optimization is only
performed for a single treatment**: with several treatments the pipeline warns
*"Optimization with multiple treatments is not yet supported."* and runs no
optimizer. The recommended pattern is one orchestrator (and `output_dir`) per
treatment.

## Errors

### `ValueError: The input DataFrame index has N duplicated label(s) ...`

Rows are matched by index label throughout the pipeline, so the index must be
unique. Call `df = df.reset_index(drop=True)` before `run_full_pipeline`.

### `ValueError` naming one of my columns (`y_col`, `d_cols`, `x_cols_list`, …)

Every column you name is checked against `df.columns` before any fitting
starts, and the message includes a *did you mean* hint for near misses. Fix the
typo in the configuration; the scaler performs the same check for its scaling
lists.

### `ValueError: Treatment '...' is configured for 'standardize' normalization, which is incompatible with a saturation family`

A saturated treatment must keep `f(0) = 0`, so it may only use a **divisive**
normalization: list it in `median_normalize_cols` or
`population_median_normalize_cols`, or leave it unscaled. `standardize_cols`,
`population_standardize_cols` and `min_max_cols` shift the treatment and are
rejected by Phase 0.

### `ValueError: optimize=True with simple_optimization=False requires a saturation function`

The detailed (saturation-aware) optimizer needs a saturation curve; without
one the budget allocation is a linear program with a corner solution. Either
add `saturation_config={"saturation": {"family": "tanh"}}` or keep
`simple_optimization=True`.

### `ValueError: Treatment column '...' is marked as 'percentage', but contains values outside [0.0, 1.0]`

A `percentage` treatment must be a share between 0 and 1 (a 15% discount is
`0.15`, not `15`). A `percentage` treatment also requires `sales_col`.

### `TRefMismatchError` at the start of Phase 1

Phase 0 stores the reference point `T_ref` of the saturation curve; Phase 1
recomputes it from the treatment column it receives and stops when the two do
not agree (`diag6_t_ref_consistency.csv` is written first). This indicates
that the treatment column was modified between the phases — for example by a
scaler change — and is not expected in normal use.

## Warnings

### "simple_optimization is forced to True because the saturation-aware optimizer is only implemented for treatment_type='spend'"

You enabled a saturation family for a `percentage`, `impressions` or `price`
treatment. Phases 0–3 still use the saturation curve; only the Phase 4
optimizer is the simple one, because the saturation-aware allocator budgets the
treatment column directly and that is only money when the treatment *is* spend.
Nothing to fix unless you expected the detailed optimizer.

### "Treatment column '...' has values in [0.0, 1.0] and name containing discount/pct/rate, but is set to 'spend'"

The default treatment type is `spend`. For a discount rate set
`treatment_types={"discount_rate": "percentage"}` together with
`treatment_spend_cols` (the column holding its monetary cost) and `sales_col`,
otherwise ROI is computed against the rate itself.

### "Skipping ROI output and optimization for treatment '...'"

A non-`spend` treatment needs both `treatment_spend_cols` and `sales_col` for
amount-based ROI; one of them is missing. See
[Treatment types and ROI charts](advanced.md#treatment-types-and-roi-charts).

### "Sample size is small (n < 1000). Set degree of continuous covariates to 2 or less."

The CATE basis can become unstable on small samples. Keep the default
`degree=2` (or lower it to 1) and use few CATE covariates.

## Reading the results

### Where is the HTML report?

In the root of `output_dir`, named `<treatment>_<outcome>[_<family>].html` —
for example `discount_rate_sales_revenue.html`, or
`tv_spend_sales_revenue_tanh.html` when saturation is enabled. Several
treatments are joined with `-`.

### The confidence band explodes in some weeks or segments

That is usually a **positivity** (overlap) problem: in those weeks or segments
the data contains almost no variation in the treatment given the covariates,
so the effect there is extrapolated. Check the treatment distribution for those
rows, extend the period, or add a geo / item breakdown. The User Guide shows
the typical chart and a KDE check.

### The ROI is negative or below the threshold

A negative incremental KPI means the model attributes a *decrease* of the
outcome to the treatment; an ROI below `opt_threshold_roi` means each unit of
spend returns less than the threshold. Before acting, check the robustness
value against the benchmark covariate (`006_sensitivity`), the heterogeneity
across regions and products, and whether a confounder is missing — a strong
seasonal driver left out of `x_cols_list` is the most common cause of
implausible signs.

### `selected_family = None` / "falling back to linear optimization"

Phase 0 ran but the data did not support a saturation curve better than a
straight line (`decision_reason` is `1se_fail`, `grid_edge_upper` or
`basis_ill_conditioned` in `summary_alpha_selection.json`). Phases 1–3 then
use the raw treatment and Phase 4 runs the simple optimizer. Nothing is broken;
see [Saturation DML](saturation.md) for the diagnostics and for how to fix the
half-saturation point from prior knowledge.
