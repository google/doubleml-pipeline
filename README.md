# DoubleML Pipeline

A robust Python library for automated causal inference utilizing Double Machine
Learning (DoubleML) and FLAML AutoML.

**By Google's APAC Marketing Effectiveness Taskforce**

*Note: `doubleml-pipeline` refers to and builds upon the underlying `DoubleML`
Python package published by [doubleml.org](https://docs.doubleml.org).*

--------------------------------------------------------------------------------

> [!IMPORTANT] **Caveat on File Generation:** For verification purposes,
> currently `doubleml_pipeline` generates CSV files and charts explicitly across
> the process.

## Why DoubleML Pipeline?

With **DoubleML Pipeline**, you can estimate the true incremental impact and ROI
of your marketing interventions including pricing and promotions by controlling
for complex, high-dimensional confounders.

Causal inference is challenging due to confounding bias, where standard
regression models often fail to isolate true causal relationships. Furthermore,
unlike standard supervised learning, Causal Machine Learning (CausalML) lacks
observational ground truth data for unobserved counterfactuals, making
**verifiability** a critical requirement for decision-making.

Double Machine Learning (DML) was selected as our base framework because it
solves these challenges through rigorous theoretical and statistical
guarantees—such as Neyman orthogonality and cross-fitting to eliminate
bias—allowing causal estimates to be mathematically verified and trusted.
**DoubleML Pipeline** operationalizes this framework by combining DML with
automated machine learning (AutoML) to make causal estimation both
scientifically robust and highly automated.

### Key Features:

*   **Automated Nuisance Modeling**: Automates nuisance parameter estimation
    using FLAML AutoML. It dynamically searches and tunes candidates (like
    LightGBM, XGBoost) under a specified time budget, removing the guesswork of
    model selection.
*   **Safe Business Unit Scaling**: Normalizes and scales geographical and
    population data securely, ensuring that descaled results represent accurate
    real-world business metrics (KPI, ROI) without introducing mathematical
    singularities.
*   **Treatment Effect Heterogeneity (CATE)**: Automatically fits continuous
    splines or binary bases to estimate how treatment effects vary across
    different contexts (Conditional Average Treatment Effects).
*   **Sensitivity Analysis (OVB)**: Runs robust omitted variable bias (OVB)
    checks to evaluate how sensitive your causal estimates are to unobserved
    confounders. The bias bound is
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
*   **Bayesian MMM Prior Calibration**: Extracts fitted distribution
    parameters (Normal, LogNormal, Beta) for treatment contribution (%) and ROI
    (e.g., discount) as prior knowledge. Because CATE estimated by DML
    fluctuates depending on conditions (e.g., seasonality), using only the mean
    of the distribution as input to Bayesian MMM priors is a consideration.
    Additionally, it is important to be aware that there may be a gap between
    ATE MMM estimates and CATE DML estimates due to differences in the
    estimands.
*   **Publication-Ready Visualizations**: Generates comprehensive time series
    line plots, stacked contribution bar charts, cross-sectional ROI
    comparisons, and prior distribution shapes.

--------------------------------------------------------------------------------

## Getting Started

### Prerequisites

Before running the pipeline, ensure you have:

*   **Python 3.10** or later.

### Installation

You can install the library directly from PyPI:

```bash
pip install doubleml-pipeline
```

Alternatively, to install from source:

```bash
git clone https://github.com/google/doubleml-pipeline.git
cd doubleml-pipeline
pip install .
```

### Quick Start Example

Here is how to initialize and run the causal pipeline in a standard Python
environment, explicitly configuring all scaling, modeling, and financial
parameters.

> [!TIP]
> **Ready-to-Use Simulation Datasets:** Don't have a dataset on hand? We have
> prepared realistic marketing simulation datasets and end-to-end sample
> notebooks in the [`examples/`](examples/) directory. Feel free to use them to
> explore the pipeline and test its features right away!

```python
import pandas as pd
# Note: Module name uses underscores (_) instead of hyphens (-)
from doubleml_pipeline.preprocessing.scalers import CausalDataScaler
from doubleml_pipeline.workflow import CausalWorkflowOrchestrator

# 1. Load your dataset
df = pd.read_csv("path/to/your/marketing_data.csv")

# 2. Initialize the scaler with explicit scaling strategies per column
scaler = CausalDataScaler(
    standardize_cols=["holiday_index", "seasonality_factor"],
    population_standardize_cols=["competitor_spend"],
    population_median_normalize_cols=["sales_revenue", "tv_spend"],
    min_max_cols=["store_count"],
    median_normalize_cols=["discount_rate"],
    population_col="geo_population",
)

# 3. Initialize the orchestrator with all explicit configuration parameters
orchestrator = CausalWorkflowOrchestrator(
    # Primary target and treatments (single treatment to support optimization)
    y_col="sales_revenue",
    d_cols=["discount_rate"],
    x_cols_list=[
        ["holiday_index", "seasonality_factor", "competitor_spend"],
    ],
    # Nuisance estimation models & cross-validation setup
    ml_models_y=["lgbm", "xgboost"],
    ml_models_t=["lgbm", "xgboost"],
    n_folds_list=[5],
    output_dir="./causal_results",
    n_jobs=-1,
    # Conditional Average Treatment Effect (CATE) specification
    covariates_for_cate_list=[
        ["seasonality_factor"],
    ],
    cate_structure_list=[
        {"type": "auto_additive", "df": 3, "degree": 2, "include_intercept": False},
    ],
    time_budget=30,
    # Panel data dimensions
    date_col="date",
    geo_col="geo",
    item_col="item",
    geo_name_col="geo_name",
    item_name_col="item_name",
    # Sensitivity analysis (Omitted Variable Bias)
    run_sensitivity=True,
    sensitivity_scope=["total", "geo", "item"],
    # Ground truth validation (set True if ground truth effect columns exist)
    ground_truth_existence=False,
    ground_truth_effect_column=None,
    # Treatment type classification & monetary spend mapping
    treatment_types={
        "discount_rate": "percentage",
    },
    treatment_spend_cols={
        "discount_rate": "discount_cost",
    },
    sales_col="sales_revenue",
)

# 4. Execute the full 4-phase pipeline (Grid Search -> Shortlist -> Ensemble -> Consolidation)
orchestrator.run_full_pipeline(
    df=df,
    scaler=scaler,
    top_n=3,
    n_reps=10,
)
```

The orchestrator will execute the models in parallel, select the best
estimators, run repetitions to calculate robust confidence bounds, perform
sensitivity checks, and write all dataframes and plot artifacts to the specified
`./causal_results` directory.

### Pipeline Execution Flow (4 Phases)

1.  **Grid Search Exploration**: Fits nuisance models for outcome `Y` and treatment `T` across all candidate AutoML models and fold combinations in parallel.
2.  **Shortlisting**: Ranks and selects top-performing models (`top_n`) based on combined error metrics.
3.  **Robust Ensemble Estimation**: Re-estimates shortlisted models across `n_reps` iterations to build a weighted ensemble with empirical confidence bounds (2.5%–97.5%).
4.  **Consolidation & Reporting**: Aggregates causal effects (CATE, incremental KPI, ROI), runs OVB sensitivity checks, and outputs charts.

### Output Directory Structure

When the pipeline runs, it saves all intermediate evaluation metrics, candidate
model repetitions, consolidated datasets, and visualization plots within the
specified `output_dir` (e.g., `./causal_results/`) as shown below:

![Output Directory Structure](img/output_directory_structure.png)

### Important Notes on Outputs & Optimization

1.  **ROI Charts:** The ROI chart folders under `roi_output/<treatment>/`
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

2.  **Optimization:** Optimization is limited to a **single treatment**.
    Two modes are available:

    | `optimize` | `simple_optimization` | `saturation` block | Behavior |
    | :--- | :--- | :--- | :--- |
    | `True` | `True` *(derived when no family is set)* | Not required | Linear (simple) optimization, unchanged from previous releases. |
    | `True` | `False` *(derived when a family is set)* | **Required** | Saturation-aware optimization (see below). |
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

3.  **Cluster DML keys (chosen automatically, every run):** The cross-fitting
    folds and the cluster-robust standard errors use a **2-way Cluster DML**
    whose keys are picked by the cluster-key determination rule — this runs
    whether or not `saturation_config` is given. The candidates are
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
    [`docs/output_csv_schema.md`](docs/output_csv_schema.md).

4.  **Hyperparameter policy (`hparam_policy`, default `"global"`):** Phases
    1–3 fit a nuisance learner in every cross-fitting fold of every candidate
    and every repetition. With `hparam_policy="global"` the FLAML search
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

--------------------------------------------------------------------------------

## Saturation DML (optional)

By default the pipeline is unchanged: omit the `saturation_config` argument
and Phases 1–4 behave exactly as before. Setting
`saturation_config['saturation']['family']` enables
**Phase 0**, which estimates a diminishing-returns parameter `alpha` for each
treatment and then feeds the transformed treatment `f(T; alpha)` into the
existing DML pipeline.

### Response families

| `family` | `f(T; alpha)` | Interpretation of `alpha` |
| :--- | :--- | :--- |
| `"tanh"` *(default)* | `tanh(alpha * T / 2)` | Rate; `T_1/2 = ln 3 / alpha` is the half-saturation point. |
| `"exp"` | `1 - exp(-alpha * T)` | Rate; `1/alpha` is the saturation scale. |
| `"log"` | `log(1 + T / alpha)` | Saturation **scale** (smaller = returns diminish sooner). |
| `"power"` | `T ** alpha` | Elasticity; must satisfy `0 < alpha < 1`. |

### Configuration

Every saturation setting lives in a single `saturation_config` dict, and
**the presence of the `saturation` block is all that is required**
(`family` defaults to `"tanh"`) — everything else defaults.
Turning it on is therefore a one-key change:

```python
orchestrator = workflow.CausalWorkflowOrchestrator(
    # ... all existing arguments ...
    optimize=True,
    saturation_config={"saturation": {"family": "power"}},
)
```

`simple_optimization` is derived: it becomes `False` automatically when a
family is set, and `True` otherwise. Pass it explicitly to override.

The full set of keys, all shown at their defaults:

```python
orchestrator = workflow.CausalWorkflowOrchestrator(
    # ... all existing arguments ...
    optimize=True,
    saturation_config={
        "saturation": {
            "family": "tanh",             # "tanh" (default) | "exp" | "log" | "power"
            "grid": {                     # alpha grid searched in Phase 0
                "mode": "auto",           # "auto" | "manual"
                "num": 25,
                "c_min": 0.25,            # auto: T_1/2 in [c_min*q50, c_max*q95]
                "c_max": 4.0,
                "max_extensions": 2,      # auto: x4 extension at an edge argmin
                "min": None,              # manual only (alpha units)
                "max": None,              # manual only (alpha units)
                "scale": "log",           # "log" | "linear" (manual spacing)
            },
            "basis_dim_T": "auto",        # "auto" | int (B-spline T-basis size J)
            "basis_tol": 1e-3,            # auto: rel. RMS approximation error
            "basis_dim_range": [6, 20],   # auto: search range for J
            "ridge": 1e-4,                # ridge penalty in the closed form (absolute)
            "t_ref_quantile": 0.95,       # T_ref = q95(T>0): all candidates pass through (T_ref, 1)
        },
        "alpha_selection": {
            "sel_fraction": 0.20,         # D_sel : D_inf split
            "n_seeds": 10,                # S: repetitions of R-loss profiling
            "k_folds": 5,                 # K for the alpha-selection stage
            "refine": "brent",            # "brent" | "none" (grid + Brent only)
            "linear_test_k": 1.0,         # adopt saturation iff d_bar < -crit * se (se = two-way cluster-robust se_obs)
            "linear_test_t_critical": False,  # crit = k (False) | Student-t value t_{G-1}^{-1}(Phi(k)) (True)
            "fallback_half_saturation_T": None,  # half-saturation point (units of T) used when Diag1/Diag2 fires
            "flatness_ratio": 10.0,       # Diag2: fires when max/min T_1/2 over the 1-SE set exceeds this
            "t_ref_tolerance": 1.0e-9,    # Diag6: rel. error tolerated between Phase 0 and Phase 1 T_ref
            "learners": {                 # Stage B nuisance learners (fixed)
                "learner_selection": False,   # True: legacy FLAML candidate search
                "y_model": {"class": "xgboost.XGBRegressor",
                            "params": {...}, "early_stopping_rounds": 30},
                "t_model": {"class": "xgboost.XGBRegressor",
                            "params": {"multi_strategy": "multi_output_tree",
                                       ...},
                            "early_stopping_rounds": 30},
                "small_sample_threshold": 2000,  # n_sel below -> RidgeCV
                "small_sample_model": "ridge_cv",
            },
            "output_dir": "step0_saturation",  # under output_dir
            # "time_budget": 30,          # learner_selection only
        },
        "final_fit": {"data": "d_inf"},   # "d_inf" (default) | "full_sample"
        # Saturation-aware optimization. Note this lives
        # under `optimization`, not `saturation`.
        "optimization": {
            "budget_mode": "fixed",       # "fixed" (budget K) | "roi_target" (no budget; ROI >= r)
            "budget_increment": 0.0,      # fixed only: K = (1 + x) * actual total of the target units
            "budget_reallocate": True,    # fixed only: redistribute what the ROI cap / negative units free up
            "cap_quantile": 1.0,          # support cap T_bar_t = max(Q_q(cell, T>0), T_actual); 1.0 = cell maximum
            "reallocate_within": "all",   # "all" (one KKT problem; spend moves across days) | "date" (one per date)
        },
    },
    # top-level optimizer scope (None = all rows of the column).
    opt_geos=None,
    opt_items=None,
    opt_threshold_roi=1.0,                # r >= 0 (default 1.0 across all optimizer paths); < 0 raises at initialization
)
```

The keys in detail:

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `saturation.family` | `"tanh"` | Used when the `saturation` block is present but `family` is omitted. `family: None` in the block still means a linear run. |
| `saturation.grid.mode` | `"auto"` | `auto` builds a dimensionless half-saturation grid `T_1/2 in [c_min*q50(T>0), c_max*q95(T>0)]`, log-spaced, for `tanh`/`exp`/`log`. `power` always uses a linear alpha grid. `manual` uses `min`/`max`/`scale` in alpha units. A legacy grid with `min`/`max` but no `mode` is treated as `manual`. |
| `saturation.grid.c_min`, `c_max` | `0.25`, `4.0` | Auto-grid range multipliers. |
| `saturation.grid.max_extensions` | `2` | When the argmin is at an auto-grid edge, extend that side x4 in `T_1/2` (matrix algebra only, no ML refits). |
| `saturation.basis_dim_T` | `"auto"` | `auto` scans `J` in `basis_dim_range` and picks the smallest `J` whose worst-case relative RMS error of projecting `fbar(.;alpha)` is below `basis_tol`. The projection is a least-squares fit (SVD pseudo-inverse, never `B'B`) on a fixed grid over `[0, T_max]` made of 500 equally spaced points, 201 quantiles of `T>0` and the Greville abscissae of the basis, so the basis matrix `B` has full column rank however skewed `T` is; `T_max` is the 99.9 % quantile of `T>0` on the full fitting data (rows above it are clipped by the basis only, never by Phases 1-4). Bases with condition number `cond(B) >= 1e8` are skipped; if no `J` meets `basis_tol`, the `J` with the smallest error is used (Diag1 warns). An integer pins `J`. A pinned or only-available basis with `cond(B) >= 1e8` makes Stage B adopt the linear model (`decision_reason = "basis_ill_conditioned"`) instead of trusting it. |
| `saturation.basis_tol`, `basis_dim_range` | `1e-3`, `[6, 20]` | See above. |
| `saturation.t_ref_quantile` | `0.95` | Quantile of `T>0`, computed once on the full fitting data before the split, that defines `T_ref`. Every candidate is normalized to `fbar(T; alpha) = f(T; alpha) / f(T_ref; alpha)`, so all curves pass through `(T_ref, 1)` and the identity candidate is `fbar_T = T / T_ref`. This removes the alpha-dependent magnitude of `f` that let the ridge solve shrink near-identity candidates (large `T_1/2`) to zero. Always on. |
| `alpha_selection.linear_test_k` | `1.0` | Saturation is adopted only if `d_bar < -crit*se`, where `d_s = L_s(alpha_hat) - L_s,T` is the paired difference against the **identity candidate** `fbar_T = T / T_ref` and `se` is the **cluster-robust `se_obs`** of the per-observation differences `d_i`. With two adopted cluster keys (e.g. `geo × date block`) the se is **two-way** (Cameron–Gelbach–Miller: `V = V_key1 + V_key2 - V_cell`), which covers dependence along both keys; with a single key it is one-way on that key (there is no across-seed se). The one-way se on the fewer-cluster key is always reported next to it (`se_one_way`), and the decision line notes when it would have decided differently. `crit = k` by default (see the next row). Otherwise `selected_family = None` (linear). The adoption rate is tuned with `k` alone: `k = 1.0` is the 1-SE rule, `2.0` a conservative choice, smaller values adopt saturation more readily. Independently of this test, an `alpha_hat` still on the upper grid edge after `max_extensions` extensions forces `selected_family = None` with `decision_reason = "grid_edge_upper"` (the data cannot distinguish the curve from a line); a lower-edge hit only warns. |
| `alpha_selection.linear_test_t_critical` | `False` | `True` replaces `crit = k` by the Student-t value with the same one-sided tail probability, `crit = t_{G-1}^{-1}(Phi(k))`, `G` = number of clusters (the smaller of the two counts for the two-way se). It compensates the noisy se of few clusters (`k = 1`: `G = 30 -> 1.02`, `10 -> 1.06`, `4 -> 1.20`) and can only make linear decisions more frequent; it is not applied to the Diag2 1-SE set. Off by default. |
| `alpha_selection.fallback_half_saturation_T` | `None` | A fixed half-saturation point in units of `T` (`T_knee` for `log`, `alpha` itself for `power`; same scale as `half_saturation_T` / `T_ref` in `summary_alpha_selection.json`, i.e. `T / c`). When Diag1 or Diag2 fires it replaces the estimate downstream and keeps the family; otherwise the 1-SE decision stands. |
| `alpha_selection.flatness_ratio` | `10.0` | Diag2 fires when `T_1/2^max / T_1/2^min` over the 1-SE set `A_1SE = {alpha : L_bar(alpha) - L_bar(alpha_hat) <= se(alpha)}` exceeds this (`T_knee` for `log`; for `power` the fixed rule `alpha^max - alpha^min > 0.4`). Warning only when a family is adopted, INFO otherwise. |
| `alpha_selection.t_ref_tolerance` | `1e-9` | Diag6: at the start of Phase 1 `T_ref` is recomputed from the treatment column Phase 1 receives; a relative error at or above this stops the run with an error (unit / hand-over bug). |
| `optimization.budget_mode` | `"fixed"` | `fixed`: a budget `K` is given and the allocation is solved in one KKT pass. `roi_target`: no budget; every unit receives `min(T_bar_t, T_t^r)` (requires `opt_threshold_roi > 0`). |
| `optimization.budget_reallocate` | `True` | `fixed` only. `True` (default): `K' = K`, solve once with the effective caps `min(T_bar_t, T_t^r)` (spend freed by negative-ROI units and the ROI cap is redistributed). `False`: `K' = K_+` (negative-ROI spend removed), solve with the caps `T_bar_t`, then cut to `T_t^r` without redistribution (`unspent_budget`). |
| `optimization.cap_quantile` | `1.0` | `q` of the support cap `T_bar_t = max(Q_q(cell_t), T_t^actual)`: `Q_q` is the `q`-quantile of the **positive** actual treatment within the unit's geo × item cell (consistent with `T_ref = q95(T>0)`; zero-spend rows do not shift it), floored at the unit's own actual spend. `1.0` (default) is the cell maximum — today's behaviour, where one outlier day lifts the cap of every unit in its cell. With e.g. `0.95` the other units are capped at the 95 % point while the outlier day keeps its own spend as its cap: it is never increased, but may still be reduced by the KKT condition or the ROI cap like any unit. Must be in `(0, 1]`. |
| `optimization.reallocate_within` | `"all"` | Scope of the KKT problem. `"all"`: one problem over every allocation unit (spend moves across days, geos and items; today's behaviour). `"date"`: one problem per date with the per-date budget `K_d = (1 + budget_increment) * sum_{t in d} T_t^actual` — no cross-day movement; `mu*` becomes per date (median with `marginal_roi_at_optimum_min` / `_max` in `optimization_metrics.csv`). See the no-carryover note under "Saturation-aware optimization". |
| `opt_threshold_roi` (top level) | `1.0` | ROI floor `r >= 0`, applied per row across both `simple_optimization=True` and `False` (including `budget_mode='roi_target'`). |
| `opt_geos`, `opt_items` (top level) | `None` | Restrict the allocation units of the saturation-aware optimizer to these geos / items (`None` = all rows of the column). Rows outside keep their actual values in the `optimized_*` columns. |
| `alpha_selection.learners` | see above | Fixed Stage B learners. `learner_selection: True` restores the legacy FLAML search. Without XGBoost >= 2.0 a warning is emitted and scikit-learn stand-ins are used: `HistGradientBoostingRegressor` for Y and one native multi-output `RandomForestRegressor` for T. |
| `alpha_selection.output_dir` | `"step0_saturation"` | Sub-directory of `output_dir` that receives the Phase 0 files. |

> [!NOTE]
> **Linear fallback.** If Phase 0 selects `selected_family = None`
> (`decision_reason` is `1se_fail`, `grid_edge_upper` or
> `basis_ill_conditioned`), then Phases 1–3 use the raw treatment, and Phase 4
> runs the existing optimization (as with `simple_optimization=True`). This
> happens even with `optimize=True` and `simple_optimization=False`. The
> pipeline warns `saturation was not identified; falling back to linear
> optimization`. The config value itself is not rewritten.
> `summary_alpha_selection.json`, `metadata.json` and the HTML report record
> `phase4_optimization = "linear_fallback"`.

> [!WARNING]
> The `saturation=`, `alpha_selection=`, `final_fit=` and `budget_increment=`
> keyword arguments have been **removed**; `saturation_config` is now the only
> way to set them. It is also strictly validated — an unknown block or key
> raises `ValueError` rather than being ignored, so a typo cannot silently
> disable Phase 0.

### What Phase 0 adds

1.  **Top-level split** into `D_sel` (alpha selection) and `D_inf` (estimation).
    With two cluster keys, the split unit is their intersection cell
    (e.g. `geo × date block`). No cell straddles the two halves, and every
    level of each key appears in both halves and, when it has at least K
    cells in `D_sel`, in every Stage B fold (`others_split_coverage.csv`).
2.  **Alpha selection** by minimizing the profiled R-loss over the
    half-saturation grid, with optional Brent refinement. The same CATE basis
    `p(X)` is used as in Phases 1–3. Every nuisance prediction is out-of-fold
    (cross-fitting: each `(seed, fold)` model predicts only its held-out
    rows) and the inner CATE coefficients of fold `k` are fitted on the
    out-of-fold residuals of the other folds, so no in-sample residual enters
    the loss. The intercept-only curve is drawn for visualization only.
3.  **Identity-candidate comparison.** All candidates are normalized
    through the reference point `(T_ref, 1)`, `T_ref = q95(T>0)`:
    `fbar(T; alpha) = f(T; alpha) / f(T_ref; alpha)`. The saturated fit must
    beat the identity candidate `fbar_T = T / T_ref` by more than
    `crit = linear_test_k` cluster-robust standard errors (two-way on the two
    adopted keys when both are known; optionally a Student-t critical value
    with `linear_test_t_critical`). Otherwise
    `selected_family = None`. An `alpha_hat` left on the upper grid edge after
    all extensions also forces `None` (`decision_reason = "grid_edge_upper"`).
3b. **Diagnostics Diag1–Diag6.** Diag1: basis approximation error
    `eps(J)` on the final grid (warn if `>= basis_tol`; `J` is fixed after the
    initial grid) and basis conditioning `cond(B)` (warn if `>= 1e8`; the
    basis is then not trusted and the linear model is adopted,
    `decision_reason = "basis_ill_conditioned"`); the single Diag1 line also
    shows `T_max` (99.9 % quantile of `T>0`) and the number of `D_sel` rows
    the basis clips at it. Diag2: 1-SE flatness, the ratio of the largest to
    the smallest half-saturation point in the 1-SE set (warn if
    `> flatness_ratio` and a family is adopted; INFO otherwise). Diag3:
    `r_T = sd(T~)/sd(T/T_ref)` and Diag4: the shape quantiles
    `fbar(q_p(T>0); alpha_hat)` with their linear references `q_p / T_ref`
    (supporting values, no rule). Diag5: `kappa(alpha_hat) = f(T_ref;
    alpha_hat)` for `tanh` / `exp` (no rule). Diag6: unit consistency of
    `T_ref` at the start of Phase 1 (error stop). Diag1 / Diag2 warn and
    continue; the fixed `fallback_half_saturation_T`, if set, is used
    downstream when they fire.
4.  If saturation is adopted, `alpha` is treated as a **constant** and Phases
    1–3 run unmodified on `fbar(T; alpha)` (so the estimated `theta` is the
    effect of one `T_ref` worth of treatment). Otherwise they run on the raw
    treatment, and Phase 4 falls back to the linear optimizer.
5.  **Every row is scored.** With the default `final_fit.data = "d_inf"`,
    Phases 1–3 are *fitted* on `D_inf` only, but the fitted nuisance models and
    the CATE function are then applied to the held-out `D_sel` rows as well.
    Everything under `step4_consolidated_results/` — the consolidated CSV, the
    CATE / incremental-KPI / ROI summaries, the charts and the Phase 4
    optimization — therefore covers **all** rows, not just `D_inf`. The
    consolidated CSV carries an `estimation_sample` column (`D_inf` = fitted
    row / `D_sel` = scored row), the 1c metrics report
    `number_of_fitted_rows` and `number_of_scored_rows`, and the log states
    e.g. `Consolidated frame covers all 180 rows: 142 fitted (D_inf) + 38
    scored with the fitted CATE (D_sel)`. The OVB sensitivity analysis keeps
    using the fitted rows only.

> [!IMPORTANT]
> A saturated treatment must use a **divisive** normalization —
> `median_normalize` or `population_median_normalize` — or none at all.
> `standardize`, `population_standardize` and `min_max` all *shift* the
> treatment, which sends roughly half the rows negative (silently clipped to
> zero by `f`) and destroys `f(0) = 0`. Phase 0 rejects those combinations with
> a `ValueError` before doing any ML work.

> [!NOTE]
> `alpha` is estimated against the **normalized** treatment `T / c`, where
> `median(T / c) == 1` by construction, which is what makes the default grid
> scale-free. The divisor `c` is **spelled out** wherever `T / c` appears: the
> Phase 0 log prints e.g. `[spend] T/c = spend / 28.3788: the treatment divided
> by the median of its non-zero values (median_normalize; one constant for all
> rows, so T/c = 1 is the typical non-zero row).`; with
> `population_median_normalize` it reads `c_i = population_i x
> median(spend/population | spend/population > 0) = population_i x <const>`
> (a per-row divisor), and without a divisive normalization `c = 1` (raw
> units). The same formula is stored in `summary_alpha_selection.json`
> (`normalization`, `divisor_constant`, `divisor_population_col`,
> `divisor_formula`, `divisor_description`), shown in the HTML Phase 0 panel
> ("Normalized treatment scale (T/c)" / "Meaning of T/c"), and used as the
> x-axis label of the fitted-saturation-curve panel of
> `summary_alpha_profile_<treatment>.png`. `T_ref` is computed on the same normalized
> treatment, once
> on the full fitting data, and stored in `summary_alpha_selection.json`;
> Phases 1–3 and 4 read it from there. Phase 4 budgets in **raw** units; the
> two are reconciled exactly by `fbar(T / c_i; alpha, T_ref) = fbar(T;
> alpha_i, c_i * T_ref)` — the per-row `gain_i` of the un-normalized `f`
> cancels in the ratio, only `T_ref` is rescaled by the same `c_i`. As a
> consequence, `alpha_hat`
> from a normalized run is not comparable to one from an un-normalized run,
> whereas `half_saturation_T / T_ref` is.

> [!TIP]
> With the default fixed learners, Phase 0 costs `2 * n_seeds * k_folds` ML
> trainings per treatment on `D_sel` only (100 at the defaults). The treatment
> side is always a single **multi-output** model with `J` outputs. One unit is
> one `fit` call of a nuisance learner; the alpha grid, its extensions, the
> identity candidate and the basis dimension `J` are matrix algebra and are
> **not** counted. With `learners.learner_selection: True` the legacy FLAML search
> adds `(N_y + N_t)` learner-selection searches per treatment, and
> `alpha_selection['time_budget']` defaults to the orchestrator's budget scaled
> by `len(D_sel) / len(D)`.

> [!NOTE]
> **Phase 0 / Stage B display.** After the Stage A block the console
> shows exactly eight items, in this order and nothing else: 1) `Stage B:
> Number of ML trainings = <treatments> x 2 nuisance models x <S> x <K> =
> <total>`; 2) one `[Phase 0 Progress] elapsed … | done/total ML trainings
> completed | ETA …` line per minute while the trainings run; then per
> treatment 3) the learners (`XGBoost` / `RidgeCV`, with `n_sel` and
> `small_sample_threshold`), 4) the grid (`mode`, `T_1/2` range, points,
> `T_ref`, extensions, `grid_edge_hit`), 5) Diag1 (`J`, `eps(J)`; `WARNING`
> when fired), 6) the grid-argmin alpha, the Brent-refined `alpha_hat` and
> `L_bar(alpha_hat)`, 7) the Diag2 line (`WARNING` / `INFO`), Diag3 / Diag4
> only when Diag2 fired, Diag5 for an adopted `tanh` / `exp`, and the 1-SE
> decision line (`decision`, `decision_reason`, `d_bar`, `se`, `k`), which
> says `FALLBACK TO LINEAR DML` explicitly when `selected_family = None`;
> 8) the completion message. Diag6 prints only when it fails (`[ERROR]`,
> then the run stops) at the start of Phase 1. No internal step logs are
> printed; the detailed messages go to
> `summary_alpha_selection.json` (`messages`).

> [!IMPORTANT]
> Phase 4 always reports in **raw treatment units**. The transformed values used
> for fitting are preserved in the `<treatment>_transformed` column.

> [!WARNING]
> If Diag2 fires (a flat R-loss profile: the half-saturation point is not
> pinned down within one order of magnitude), the estimate of `alpha` is
> unreliable — see `step0_saturation/diag2_rloss_profile.png` and the Diag3 / Diag4
> lines. In that case prefer fixing the half-saturation point from prior
> knowledge via `fallback_half_saturation_T` (used automatically when Diag1 or
> Diag2 fires) or a narrow manual grid around the prior value
> (`{"mode": "manual", "min": a_lo, "max": a_hi}`; `num >= 2` and
> `min < max` are required).

### Phase 0 outputs

All written to `output_dir/step0_saturation/`. With several
treatments each one gets its own `step0_saturation/<treatment>/` directory.
Every file name carries a prefix that says what it belongs to: `diag1_` …
`diag6_` for the six identification diagnostics, `summary_` for the result of the alpha
selection, `others_` for supporting material without a diagnostic rule. Every
PNG ends with a footnote that explains its annotations, states the rule /
status of its diagnostic for this run and names the CSV columns behind it.

| File | Contents |
| :--- | :--- |
| `summary_alpha_selection.json` | `family`, `selected_family`, `alpha_hat`, `half_saturation_T` (`T_knee` for `log`), `alpha_grid_argmin`, `rloss_min` (`L_bar(alpha_hat)`), `refine_method`, `d_bar`, `se` (cluster-robust `se_obs`), `linear_test_k`, `decision`, `decision_reason` (`1se_pass` / `1se_fail` / `grid_edge_upper`), `diag1_fired`, `J`, `basis_approx_error`, `diag2_fired`, `diag2_ratio`, `r_T`, `fbar_q25` … `fbar_q90`, `linear_ref_q25` … `linear_ref_q90`, `saturation_at_T_ref` (`null` for `log` / `power`), `grid_mode`, `grid_extensions`, `grid_edge_hit` (`none` / `upper` / `lower`), `T_ref`, `T_max`, `cluster_keys`, `cluster_fallback_tier`, `n_sel`, `n_inf`, `n_seeds`, `k_folds`, `learners`, plus extras (for example `phase4_optimization`, `diag6_status`, `flatness_ratio`, `basis_tol`, `fallback_half_saturation_T_used`, `t_ref_quantile`, `T_ref_raw`, `grid_variable`, and the divisor of the normalized treatment: `normalization`, `divisor_constant`, `divisor_population_col`, `divisor_formula`, `divisor_description`). NaN is written as `null`. Phases 1–3 and Phase 4 read `T_ref` from this file. |
| `summary_alpha_grid.csv` | One row per grid point: `grid_index`, `half_saturation_T`, `alpha`, `mean_r_loss` (`L_bar(alpha)`), `se_alpha` (the Diag2 `se(alpha)`), `in_1se_set` (membership in `A_1SE`), `fbar_q25`, `fbar_q50`, `fbar_q75`, `fbar_q90` (`phi_p(alpha) = fbar(q_p(T>0); alpha)`), `basis_approx_error_alpha` (the Diag1 `eps(J)` per alpha). This is the complete data behind the R-loss profile charts (it replaces the former `alpha_profile_<treatment>.csv`). |
| `summary_rloss_curves.csv` | `seed`, `grid_index`, `alpha`, `rloss_cate`, `rloss_ate`, `rloss_identity`. |
| `summary_alpha_profile_<treatment>.png` | The one-page Phase 0 summary, 2 + 2 + 1 panels from top to bottom: **Grid construction** (histogram of `T>0` with `q50`, `q95 = T_ref`, the automatic grid range and every grid point) and **Seed-wise alpha distribution** (per-seed `alpha_hat_s` histogram); **Profiled R-loss curve** with `A_1SE` and **Fitted saturation curve (normalized through (T_ref, 1))** — `fbar(T; alpha_hat)`, the identity candidate `T / T_ref`, `(T_ref, 1)`, the shapes of the smallest / median / largest grid alpha and the curve Phases 1–4 actually use labelled **FINAL**; and the full-width **Key stats & process metrics** table whose `diagnostics` column tags each row with `diag1` … `diag6`. The footnotes define `T_ref`, `q05 / q50 / q95`, the grid ends `0.25*q50` (fastest saturation) / `4*q95` (mildest, practically linear) and list every file of the folder. |
| `summary_alpha_seed-wise_alpha_distribution.csv` | The data behind the seed-wise alpha distribution panel: `seed`, `alpha_hat_s`, `rloss_at_alpha_hat_s`, `rloss_identity_s`, `d_s` (visualization only, no rule). |
| `summary_alpha_selection.csv`, `summary_phase0_diagnostics.json` | Run-level: one row per treatment (decision, `alpha_hat`, Diag1–Diag5 flags, compute accounting) and the Stage A split / cluster-key plan with every Phase 0 warning. |
| `diag1_basis_approx_error.png` | Diag1: `eps(J)(alpha)` per grid point against `basis_tol`, with `J`, `max eps(J)` and the fired status in the title. |
| `diag2_rloss_profile.png` | Diag2: R-loss vs `T_1/2`: CATE curve (±1 se), intercept curve (visualization only), identity candidate `L_bar_T` (±k se), the 1-SE set `A_1SE`, `alpha_hat`, `decision`; the footnote states the flatness ratio against `flatness_ratio` and whether Diag2 fired. |
| `diag3_identification_diagnostics.csv` | Diag3: `r_T`, `sd_T`, `sd_T_resid`, `share_T_zero`, `q50_T_pos`, `q95_T_pos`, `T_max`, `T_ref`. |
| `diag4_f_transform_quantiles.png` | Diag4: the four shape quantiles `phi_25`, `phi_50`, `phi_75`, `phi_90` over alpha with the linear references `q_p / T_ref`. |
| `diag5_saturation_at_T_ref.png` | Diag5 (`tanh` / `exp` only): `kappa(alpha) = f(T_ref; alpha)`, the share of the ceiling reached at `T_ref`, over the grid with `kappa(alpha_hat)` marked. |
| `diag6_t_ref_consistency.csv` | Diag6, written at the start of Phase 1: `treatment`, `status` (`pass` / `fail` / `skipped` when `selected_family = None`), `T_ref_phase0`, `T_ref_phase1`, `rel_error`, `t_ref_tolerance`, `t_ref_quantile`, `n_rows_phase1`, `note`. |
| `others_split_coverage.csv`, `others_split_coverage_detail.csv`, `others_split_coverage.png` | Stage A / Stage B bookkeeping: per key level `n_cells_sel`, `n_cells_inf`, `covered`, `n_folds_covered_min`, `fold_covered` (the summary CSV, drawn as the bar chart of `n_cells_sel` / `n_cells_inf` per level with uncovered levels in red); and the per-cell detail `seed`, `cluster_id`, `key1_value`, `key2_value`, `fold` (`-1` for `D_inf`), `split` (`sel` / `inf`). |

The HTML report gains a **000_saturation** tab, which embeds
`diag2_rloss_profile.png` and `summary_alpha_profile_<treatment>.png` and shows
`decision`, `decision_reason`, `alpha_hat`, `half_saturation_T`, `T_ref`,
`d_bar`, `se`, `diag2_ratio`, `saturation_at_T_ref` and `r_T`.

### Saturation-aware optimization

With `optimize=True`, `simple_optimization=False`, one treatment of type
`spend` and `selected_family != None`, Phase 4 solves

`max sum_t theta_t fbar(T_t)  s.t.  sum_t T_t <= K',  0 <= T_t <= T_bar_t^eff`

in **one KKT pass**: one allocation unit is one row (item × geo × date) inside
`opt_periods` ∩ `opt_geos × opt_items`; `theta_t = p(X_t)' beta_hat`; the
support cap `T_bar_t = max(Q_q(cell_t), T_t^actual)` is the
`optimization.cap_quantile`-quantile `q` of the **positive** actual treatment
within the unit's geo × item cell, floored at the unit's own actual spend (the
default `q = 1.0` is the cell maximum, i.e. the historical cap; with
`q = 0.95` an outlier day keeps its own actual spend as its cap — it can be cut
by the KKT condition or the ROI cap but no unit is ever pushed above its cap —
while every other unit is capped at the cell's 95 % point); the ROI threshold
`r = opt_threshold_roi` is a per-unit cap `T_t^r` (the root of
`theta_t fbar(T)/T = r`; `0` when `theta_t fbar'(0) <= r`; `inf` when
`r = 0`), `T_bar_t^eff = min(T_bar_t, T_t^r)`; units with `theta_t <= 0`
receive `0`. `T_t*(mu) = clip((fbar')^-1(mu / theta_t), 0, T_bar_t^eff)` with
the closed forms of all four families and a single bisection on `mu` so that
`sum_t T_t*(mu) = K'` (`mu* = 0` when the caps absorb the budget). The former
"exclude below the threshold → re-solve → iterate" loop no longer exists.
`budget_mode = roi_target` skips the budget and sets `T_t* = T_bar_t^eff`.
Any other combination (several treatments or `selected_family = None`) falls
back to the existing optimizer with a warning; a non-`spend` treatment type
forces `simple_optimization=True` at construction (see above);
`simple_optimization=False` is otherwise never overridden when
`saturation_config` is given, and `opt_threshold_roi < 0` raises at
initialization.

> [!IMPORTANT]
> **No carryover is modelled.** The response of a row depends on that row's
> spend only (`theta_t fbar(T_t)`); this row-wise separability is what makes
> the one-pass KKT allocation exact, and it also means that the optimizer
> moves spend freely **across days as well as across geos and items** (the
> allocation units are rows). Read the optimized allocation at the
> geo × item × period level (totals over `opt_periods`), not day by day. To
> keep every day's total fixed, set `optimization.reallocate_within: "date"`:
> one KKT problem per date with the per-date budget
> `K_d = (1 + budget_increment) * sum_{t in d} T_t^actual` (spend still moves
> across geos and items within the day; `mu*` is then per date and
> `optimization_metrics.csv` reports its median with
> `marginal_roi_at_optimum_min` / `_max`). If a **lag of the treatment**
> (`<treatment>_l<p>` from `create_lagged_features`) is among `x_cols` or
> `covariates_for_cate`, `X_t` depends on past `T` and the separability no
> longer holds; Phase 4 warns about it. Adstock / carryover itself
> (`f(adstock(T); alpha)`) remains future work; see §9.4 of
> `saturation_dml_design_en.md`.

The saturation optimizer additionally writes `optimized_incremental_KPI_<t>` and
`optimized_roi_<t>` into the consolidated CSV and into
`007_optimization/optimization_metrics.csv`, where every geo / item / `all`
row also carries `budget_mode`, `budget_reallocate`, `reallocate_within`,
`cap_quantile`, `assumptions` (`no_carryover`), `K`, `K_out`,
`unspent_budget`, `marginal_roi_at_optimum` (`mu*`; with
`marginal_roi_at_optimum_min` / `_max`, equal to it under
`reallocate_within = "all"`), `n_units`, `n_units_excluded_negative_roi`,
`n_units_capped_by_roi` and `n_units_at_support_cap` (`K`, `K_out`,
`unspent_budget` and the counts are aggregated over the units of that row, so
a geo row can show a negative `unspent_budget` when budget was reallocated
into it; `budget_mode`, the scope and `mu*` describe the whole problem). The
same values appear in the HTML descriptive block of the `007_optimization`
tab. When `treatment_type="spend"` it also writes, per geo, per item and for
the total, the two-line chart `<prefix>_original_vs_optimal_treatment.png`
(purple original vs green optimal treatment); the incremental effect is covered
by the before / after chart below.

Both optimizers write, per geo, per item and for the total, the outcome chart
`<prefix>_<outcome>.png`: the red observed outcome (`Original KPI`) against the
blue projected outcome under the optimized spend (`Estimated total KPI with
optimal treatment`, `Y + (KPI_opt - KPI)` summed per date — for the simple
optimizer the KPI of the cut rows leaves the total). The HTML tabs
`007_optimization` and `roi_output_007_optimization (<treatment>)` show it in
the same position whatever `simple_optimization` is: right after the
optimizer's own chart and above the before / after charts.

Both optimizers (simple and saturation-aware) write, per geo, per item and for
the total, three before / after charts — two line charts stacked vertically on
an identical vertical scale, "before optimization" on top and "after
optimization" below, every panel on exactly the width and x-axis layout of
`<prefix>_timeseries_spend_vs_est.png` so the dates line up:
`<prefix>_timeseries_spend_before_after.png` (purple actual spend
(`treatment_spend_cols`) / dark green optimized spend),
`<prefix>_timeseries_incremental_effect_before_after.png` (red
incremental effect of the actual spend / blue incremental effect under the
optimized spend) and `<prefix>_timeseries_roi_before_after.png` (red ROI per
date = incremental KPI / spend, `0` without spend / blue ROI under the
optimized spend, with the ROI threshold as a dotted reference line) — every
panel of the three charts is exactly half the height of `spend_vs_est` at the
same vertical scale, so each before / after pair fits one screen — and
`optimization_yearly_summary.csv` (per dimension and calendar year plus
`Total`: `spend_before`, `incremental_kpi_before`, `roi_before`,
`spend_after`, `incremental_kpi_after`, `roi_after`). The HTML tabs
`007_optimization` and `roi_output_007_optimization (<treatment>)` show the
three charts in that order under the existing optimization chart and render
the CSV as a matrix below them that follows the dimension dropdown: one row per
metric (Spend, Incremental KPI, ROI) under "Before optimization" / "After
optimization", one column per year plus `Total`, numbers only in the cells; the
matrix always fits the width of the tab (its font size adapts to the widest
number, down to 9 px, after which the label columns narrow and, as a last
resort, the matrix scrolls inside the tab rather than cutting digits).

### HTML report

When `generate_html=True` the interactive report is written to the root of
`output_dir` as `<treatment>_<outcome>_<family>.html`, for example
`tv_spend-digital_spend_sales_power.html`. Multiple treatments are joined with a
hyphen, and the `_<family>` segment is omitted when saturation is not enabled
(`tv_spend_sales.html`).

--------------------------------------------------------------------------------

### Analyzing Results with AI (Browser Chat or ADK Agent)

There are two ways to explore the causal estimates with Gemini:

1. **Directly in `report.html` (no server required)**: Open
   `<output_dir>/report.html` in your browser, select the
   **`✨ AI Assistant (Gemini)`** tab, and enter a Gemini API Key to chat
   interactively with the precomputed `CausalResultsReader` outputs and bundled
   `retail-analytics` skill.
2. **Local [ADK](https://google.github.io/adk-docs/) Agent (`generate_agent=True`)**:
   By default, the pipeline also writes a runnable ADK agent to
   `<output_dir>/agents/causal_ml_insight_agent/`. The agent answers business
   questions such as "Which promotion pays back, where, and when?" from the
   pipeline outputs. Its tools read the results and do the arithmetic
   deterministically, so the LLM interprets the DoubleML estimates instead of
   computing causal effects itself. The domain knowledge comes from
   [Agent Skills](https://agentskills.io) bundled in
   [`skills/`](skills/), starting with `retail-analytics`. Coding agents can use
   these skills directly too.

```bash
cp causal_results/agents/causal_ml_insight_agent/.env.example \
   causal_results/agents/causal_ml_insight_agent/.env  # Set your project.
adk web causal_results/agents
```

To regenerate the agent with your own skills or model, or to add one to the
results of an earlier run:

```python
from doubleml_pipeline.agent import generate_agent

generate_agent("./causal_results", skills_dir="./my_skills")
```

If you run one orchestrator per channel, one agent can cover all of the runs.
It compares the channels side by side but never adds their effects up, since
each run is estimated separately:

```python
generate_agent(
    ["results/Spend_TV", "results/Spend_Digital", "results/Spend_Flyer"],
    output_dir="results",
)
# adk web results/agents
```

### User Guide

For a comprehensive step-by-step walkthrough of pipeline settings, modeling workflows, and visual slide guides, refer to the [User Guide](docs/guide.md).

[![User Guide Sample](img/guide_38.png)](docs/guide.md)

--------------------------------------------------------------------------------

## Engaging with the Project

### Documentation & Examples

*   **User Guide**: For a step-by-step walkthrough of pipeline settings and visual slide guides, refer to the [User Guide](docs/guide.md).
*   **API Reference**: For a comprehensive list of modules, classes, and
    function signatures, refer to the [API Reference](docs/api_reference.md).
*   **Output CSV Schema**: For detailed specifications of all output CSV files, including column names, data types, and descriptions, refer to the [Output CSV Schema](docs/output_csv_schema.md).
*   **Interactive Samples**: Check out the sample Jupyter notebooks
    ([`IN_skincare`](examples/IN_skincare/doubleml-pipeline_sample_notebook_IN.ipynb)
    and
    [`JP_retailer`](examples/JP_retailer/doubleml-pipeline_sample_notebook_JP.ipynb)),
    which walk through loading realistic simulated panel marketing data,
    engineering lagged features, scaling, running the DML pipeline, and plotting
    the results.

### Contributing & Feedback

We welcome contributions!

*   **Reporting Issues**: If you encounter bugs or want to request features,
    please open an issue in the GitHub repository issue tracker.
*   **Submitting Changes**: Please see [CONTRIBUTING.md](CONTRIBUTING.md) for our guidelines on
    submitting pull requests.

--------------------------------------------------------------------------------

## Acknowledgments & Dataset Credits

We gratefully acknowledge our partners for their collaboration on the simulation
datasets included under [`examples/`](examples/):

*   **[`examples/IN_skincare/`](examples/IN_skincare/)**: Simulation data and
    example materials provided in collaboration with **Analytic Edge**.
*   **[`examples/JP_retailer/`](examples/JP_retailer/)**: Simulation data
    designed and provided by **Hakuhodo Inc.**, informed by real-world Japanese
    marketing dynamics.

--------------------------------------------------------------------------------

## References

1.  **Debiased Machine Learning method** Victor Chernozhukov, Denis Chetverikov,
    Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins.
    *Double/debiased machine learning for treatment and structural parameters*.
    [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)

2.  **Hyperparameter Tuning for Causal Inference with Double Machine Learning: A
    Simulation Study** Martin Spindler, et al.
    [arXiv:2402.04674](https://arxiv.org/abs/2402.04674)

3.  **Debiased Machine Learning of Conditional Average Treatment Effects and
    Other Causal Functions** Vira Semenova, Victor Chernozhukov.
    [arXiv:1702.06240](https://arxiv.org/abs/1702.06240)

4.  **Multiway Cluster Robust Double/Debiased Machine Learning** Harold D.
    Chiang, Kengo Kato, Yukitoshi Matsushita, Takuya Ishihara.
    [arXiv:1909.03489](https://arxiv.org/abs/1909.03489)
