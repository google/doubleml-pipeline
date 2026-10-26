# Advanced configuration & internals

This page collects the configuration details and implementation notes that go
beyond the [README](../README.md): how treatment types drive the ROI charts,
the two optimization modes, the automatic Cluster DML keys, the hyperparameter
policy and parallel plan of Phases 1–3, the formulas behind the sensitivity
analysis, and the HTML report file name. Saturation DML (Phase 0) has its own
page: [saturation.md](saturation.md). Every parameter mentioned here is
documented in the [API Reference](api_reference.md).

## Treatment types and ROI charts

The ROI chart folders under `roi_output/<treatment>/`
(`003_roi_by_month`, `004_roi_by_geo`, `005_roi_by_item`) are
conditionally generated depending on your
configured `treatment_types` (default is `"spend"`), along with
`treatment_spend_cols` and `sales_col`, as summarized below:

| `treatment_types` | `treatment_spend_cols` & `sales_col` | ROI Charts | Pipeline Behavior & Reason |
| :--- | :--- | :--- | :--- |
| `"spend"` *(default)* | Not required | **Not Generated** | Not generated specifically because they are redundant with the output generated in Phase 4. |
| `"percentage"`, `"impressions"`, or `"price"` | **Provided** | **Generated** | Explicitly generates the ROI charts (`003_roi_by_month`, `004_roi_by_geo`, `005_roi_by_item`). |
| `"percentage"`, `"impressions"`, or `"price"` | **Missing** | **Skipped (Warning)** | Pipeline logs a warning and skips ROI charts because monetary spend mapping is missing. |

> [!NOTE]
> When setting `treatment_types` to `"price"`, "price" should refer to the
> difference between the current price and a benchmark price (for example,
> `price = maximum price - current price`).

## Optimization modes

Optimization is limited to a **single treatment**.
Two modes are available:

| `optimize` | `simple_optimization` | `saturation` block | Behavior |
| :--- | :--- | :--- | :--- |
| `True` | `True` *(derived when no family is set)* | Not required | Linear (simple) optimization, unchanged from previous releases. |
| `True` | `False` *(derived when a family is set)* | **Required** | Saturation-aware optimization (see [saturation.md](saturation.md#saturation-aware-optimization)). |
| `True` | `False` *(passed explicitly)* | Missing | **`ValueError` at construction.** Detailed optimization is only defined under a saturation model. |
| `True` | `True` *(forced for non-`spend` treatment types)* | Optional | Simple optimization. The saturation-aware optimizer is only implemented for `treatment_types = "spend"`. |

> [!IMPORTANT]
> Saturation-aware optimization requires `treatment_types = "spend"`. For
> `"percentage"`, `"impressions"` and `"price"` the orchestrator **forces
> `simple_optimization = True`** at construction (with a warning when the
> derived or explicit value was `False`), because the saturation allocator
> budgets the treatment column directly and that is only money when the
> treatment *is* spend. Phases 0–3 still use saturation for these
> treatments; only the Phase 4 optimizer is the simple one. The value in
> force is reported in `metadata.json["simple_optimization"]` and in the
> header of the HTML report (`Simple Optimization: True/False`).

> [!NOTE]
> `simple_optimization` defaults to `None`, meaning "derive it from whether
> `saturation_config['saturation']['family']` is set". An explicit
> `True`/`False` always wins for `spend` treatments, which is the only way
> to reach the third row.

## Cluster DML keys (chosen automatically, every run)

The cross-fitting folds and the cluster-robust standard errors use a **2-way
Cluster DML** whose keys are picked by the cluster-key determination rule —
this runs whether or not `saturation_config` is given. The candidates are
`item_col`, `geo_col` (each needs at least two distinct values) and a
contiguous **date block** derived from `date_col` (never raw dates); the
two candidates with the **most clusters** are adopted, one if only one is
usable, none → observation-level inference. `date_col`, `item_col` and
`geo_col` keep their visualization role in every chart and summary CSV
regardless of which keys were adopted. The choice is logged (at the start
of Phase 1 in a normal run, in the Phase 0 header otherwise):

```
Cluster DML keys (determination rule applied automatically; no saturation_config needed): 2-way: cluster_date_block (7 clusters), geo (6 clusters); not adopted: item (2 clusters); date block = monthly blocks (weekly data)
```

and recorded in `metadata.json["cluster_dml"]` (`keys`, `n_way`,
`cluster_counts`, `not_adopted`, `date_block_granularity`,
`visualization_cols`, `keys_used`, …); the HTML report shows the same
information in its header. See
[`docs/output_csv_schema.md`](output_csv_schema.md).

## Hyperparameter policy (`hparam_policy`, default `"global"`)

Phases 1–3 fit a nuisance learner in every cross-fitting fold of every
candidate and every repetition. With `hparam_policy="global"` the FLAML search
(`time_budget` seconds, `eval_method="auto"`, `early_stop=True`) runs
**once per (treatment, learner)** at the start of Phase 1 — on the sample
Phases 1–3 are fitted on (`final_fit`), with target `Y` for the Y-side
learners and the PLR treatment `D` (`f̄(T; α̂)` when Phase 0 adopted a
saturation family, `T` otherwise) for the T-side learners — and every fold
then fits the winner's concrete estimator with those fixed
hyperparameters (`models.fixed_hparam_regressor.FixedHparamRegressor`). The DML
cross-fitting, CATE, repetitions, median aggregation and the combined
error (now fed by the same validation RMSE measured inside each fold's
training data with the method FLAML chose) are unchanged. The searches
are persisted to `step1_exploration_<d_col>/hparams_<d_col>_<learner>.json`
(one file per learner; a learner used on both sides has a `Y` and a `T`
entry). `hparam_policy="per_fold"` restores the previous behaviour (a
FLAML search inside every fold). Independently of the policy, Phase 1
reuses its results as repetition 0 of Phase 3, Phase 3 runs all
`(candidate, repetition)` jobs as one parallel job list, and the parallel
sections allocate `workers = min(tasks, cores)` processes with
`cores // workers` learner threads each (logged as `workers = W,
threads/worker = h`). The sensitivity analysis fits the benchmark's short
model once per treatment and reports group-specific benchmark points
(`benchmark_cf_d`, `benchmark_cf_y`, plus the new `benchmark_rho` and
`benchmark_scope` columns of `sensitivity_metrics.csv`). Phase 0 is not
affected by `hparam_policy`.

## Sensitivity analysis details

Set `run_sensitivity=True` (and `sensitivity_scope`, default `["total"]`,
optionally with `"geo"` / `"item"` for group-level rows) to run the
omitted-variable-bias (OVB) checks on the best model. The bias bound is
`rho_U * scale * sqrt(cf_y * cf_d / (1 - cf_d))` with the residual-based
`scale = sqrt(sigma2 * nu2)` (DoubleML's `sensitivity_elements`;
recomputed from the out-of-fold residuals when unavailable; per-group
moments for geo / item GATE rows) and the robustness value is
`RV = (sqrt(f^4 + 4 f^2) - f^2) / 2` with `f = (|theta| - 1.96 se) / scale`
(Cinelli & Hazlett 2020), so the RV sits on the zero contour of the 95%
lower bound; `sensitivity_metrics.csv` carries `n_obs`, `f_q`, `sigma2`,
`nu2`, `scale` so the numbers can be verified. With a saturation family
adopted in Phase 0 the analysis runs on the same PLR fitted with
`D = fbar(T; alpha_hat)` (no extra fit), the benchmark covariate is chosen
by correlation with `fbar(T)`, and the bounds are conditional on
`alpha_hat` (`treatment_variable = "f_bar(T)"` in the CSV).

## HTML report

When `generate_html=True` the interactive report is written to the root of
`output_dir` as `<treatment>_<outcome>_<family>.html`, for example
`tv_spend-digital_spend_sales_power.html`. Multiple treatments are joined with a
hyphen, and the `_<family>` segment is omitted when saturation is not enabled
(`tv_spend_sales.html`).
