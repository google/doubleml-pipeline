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
> A `"price"` treatment is estimated as the **gap** between a benchmark price
> and the observed price, not as the price itself; the pipeline builds the
> gap (and the price investment the ROI is computed against) from the
> `benchmark` entry of `treatment_types`. See
> [Price treatments](#price-treatments-benchmark-gap-and-price-investment).

## Price treatments: benchmark, gap and price investment

The incremental KPI of a row is `θ(X) × D`, i.e. the effect of the treatment
against the counterfactual `D = 0`. With a raw price as `D` that
counterfactual is a price of zero — far outside the data — and the
"incremental" KPI is a linear extrapolation of the size of the sales
themselves. A `"price"` treatment is therefore estimated as the gap

`D_{g,t} = b_{g,t} − p_{g,t}`

between a **benchmark price** `b` and the observed price `p`, so that `D = 0`
means "sold at the benchmark price". Vocabulary used below:

| Term | Definition |
| :--- | :--- |
| group `g` | the rows with the same `item_col` and `geo_col` values (one of them when only one is set; all rows when neither is set, with a warning) |
| `p_{g,t}` | the price of group `g` at date `t` (the treatment column as given) |
| `b_{g,t}` | the benchmark price of that row |
| `D_{g,t}` | `b_{g,t} − p_{g,t}`, the gap — the treatment Phases 0–4 estimate |
| `U_{g,t}` | `sales_{g,t} / p_{g,t}`, the quantity sold (`sales_col`) |
| `C_{g,t}` | `U_{g,t} × max(D_{g,t}, 0)`, the price investment (the ROI cost) |

### Setting the benchmark

The benchmark is configured inside `treatment_types`; a `"price"` entry may
be the plain string or a dict with a `benchmark` (and, for `rolling_max`, a
`window`) key. The key is the **treatment column as listed in `d_cols`**, not
the word `price` (for a price column `UnitPriceSM`:
`{"UnitPriceSM": {"type": "price", "benchmark": "q_95"}}`); a key that is not
in `d_cols` raises `ValueError` at construction, because it would otherwise
be ignored and the column estimated as a plain `spend` treatment. A lag
`<d>_l<n>` listed in `d_cols` needs its own entry (see *Lags* below).

```python
treatment_types = {
    "tv_spend": "spend",
    "price": {"type": "price", "benchmark": "q_95"},
}
```

| `benchmark` | Meaning |
| :--- | :--- |
| *(omitted, or the plain string `"price"`)* | **Automatic**: a column whose smallest non-NaN value is `<= 0` cannot be a price and is taken to be a gap already (`as_is`); otherwise the group maximum `q_100` is used. The decision is announced with a `UserWarning` and recorded in `metadata.json`. |
| `"as_is"` | The column already is `D` (for example `list_price − price` computed by the caller); nothing is transformed. |
| `"q_k"`, `k` an integer in `1…100` | The `k`-th percentile of the prices of the group, taken as the **smallest observed price at or above the percentile** (`np.quantile(..., method="higher")`), so `b` is always a price that occurred and every group has at least one row with `D = 0`. `q_100` is the group maximum; it is the only quantile with `D >= 0` on every row. |
| `"rolling_max"` with `"window": "<n>W"` or `"<n>D"` | `b_{g,t} = max{ p_{g,s} : t − W < s <= t, (g, s) observed }` — the maximum price of the group over the calendar window of length `W` (`n` weeks or days) **ending at and including the row's own date**, so `D >= 0` always. The first observed date of a group has `D = 0` (its window holds that date only); missing dates are simply absent from the window. Requires `date_col`. |
| a number | A constant benchmark price (must be `> 0`). |
| any other string | The name of a column holding the benchmark price (a list price, for example). It must exist in `df`; a column named like one of the keywords above (`as_is`, `rolling_max`, `q_<k>`) is rejected. |

### What the pipeline does with a price treatment

At the start of `run_full_pipeline`, right after the column check and before
anything else:

1.  `<d>_benchmark` (`b`) and `<d>_gap` (`D`) are added to a copy of `df` (the
    caller's frame is not modified) and the treatment is **renamed** to
    `<d>_gap` in `d_cols`, `treatment_types` and `treatment_spend_cols`: every
    output (`estimated_incremental_KPI_<d>_gap`, the charts, the summaries,
    `metadata.json["treatment_cols"]`) carries the gap's name. `as_is` leaves
    the treatment unchanged.
2.  `<d>_benchmark` is appended to the treatment's `x_cols` (logged; not for a
    constant benchmark, which carries no information). It is **not** added to
    `covariates_for_cate`. The price level is thereby controlled for, which
    is what makes the absolute gap comparable across items. Never put the
    price column itself in `x_cols`: with `D = b − p`, `p` determines `D`
    exactly and the treatment residual vanishes.
3.  The **price investment** `<d>_investment` is created and becomes the
    treatment's `treatment_spend_cols` entry:
    *   with `treatment_spend_cols[d]` given by the caller, that column with
        its value replaced by `0` on the rows where `D <= 0`;
    *   otherwise, when `sales_col` is set and the benchmark is not `as_is`,
        `C = sales_col / p × max(D, 0)` (rows with NaN sales get `0`; a price
        `<= 0` raises `ValueError`, because the quantity is undefined). This
        presumes `sales_col` to be the sales **amount** of the same item, geo
        and date at price `p`; if your `sales_col` is a quantity, pass
        `treatment_spend_cols` instead;
    *   otherwise no investment column exists and Phase 4 skips the ROI
        output and the optimization of the treatment with the usual warning.
4.  **Lags.** A treatment `<d>_l<n>` (as built by `create_lagged_features`)
    is a lag of a price column: its benchmark is computed from the base
    column `<d>` (which must be in `df`; all lags of one column must agree on
    the benchmark) and its gap `<d>_gap_l<n>` is the gap of the same group
    `n` periods earlier — benchmark and price both taken at the earlier date —
    with `0` where that row does not exist. The investment is shifted the
    same way (unless `treatment_spend_cols` names a column for the lag).
    Requires `date_col`. Each lag needs its own `treatment_types` entry
    (e.g. `{"price": {...}, "price_l1": "price"}`); a lag without one whose
    base column is a `price` treatment raises `ValueError` at construction.
5.  If the price column is listed in one of the scaler's column lists
    (`standardize_cols`, …), `<d>_gap` and `<d>_benchmark` are added to the
    same list, so the treatment is scaled the way the caller intended.
6.  The benchmark values are written to `price_benchmark/<d>_benchmark.csv`
    (one row per group, or per group and date for `rolling_max` and column
    benchmarks) and `metadata.json["price_benchmarks"]` records, per gap
    column, the specification and how it was resolved (`auto` / `explicit`),
    the benchmark column, the group columns, the numbers of rows with
    `D = 0` and `D < 0`, and the investment column with its source
    (`user` / `auto` / `none`) and formula. The header of the HTML report
    repeats the benchmark (and window) next to the treatment type and the
    origin of the investment column next to the spend column.

### Rows with `D < 0`

They occur only with `as_is`, a column or constant benchmark, or `q_k` with
`k < 100` (a price above the benchmark). Such a row **keeps its negative gap
in the estimation**: its `estimated_incremental_KPI` is `θ(X) × D` (or
`θ(X) × f̄(D)` under a saturation family), whose sign is that of `θ(X)`. With
a saturation family the curve is extended to negative values as an odd
function, `f(D) = −f(−D)`, in Phase 0 (α selection) and Phases 1–3 alike.
In the ROI output and the optimization the row counts as **zero
investment** (`C = 0`). The number of such rows is reported with a
`UserWarning` and in `metadata.json`.

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
> force is reported in `metadata.json["simple_optimization"]` (the user's
> selection in `simple_optimization_requested`, the derived value in
> `simple_optimization_derived` and the forcing in
> `simple_optimization_forced`) and in the header of the HTML report
> (`Simple Optimization: user selection: … | final decision: …`, where the
> final decision says whether the simple optimizer ran — also `True` through
> the Phase 4 linear fallback, which is then named).

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
