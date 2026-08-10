# `doubleml-pipeline` API Reference

This document outlines the core public classes and functions available in the
`doubleml-pipeline` package.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.workflow`

### Class: `CausalWorkflowOrchestrator`

The central manager for orchestrating the DoubleML execution pipeline, handling
parallelization, model selection, and visual reporting.

#### `__init__(self, y_col, d_cols, x_cols_list, ml_models_y, ml_models_t, n_folds_list, output_dir, n_jobs=-1, covariates_for_cate_list=None, cate_structure_list=None, time_budget=15, date_col=None, geo_col=None, item_col=None, geo_name_col=None, item_name_col=None, run_sensitivity=False, sensitivity_scope=None, ground_truth_existence=False, ground_truth_effect_column=None, treatment_types=None, treatment_spend_cols=None, sales_col=None, peak_months=None, optimize=False, simple_optimization=True, opt_periods=None, opt_threshold_roi=0.0, opt_other_cost_variables=None)`

**Parameters:**

-   `y_col` *(str)*: The outcome variable column name. Example: `'sales'`.
-   `d_cols` *(List[str])*: List of treatment variable column names. Example: `['discount_rate', 'tv_spend']`.
-   `x_cols_list` *(List[List[str]])*: List of confounding variables
    corresponding to each treatment variable. Example:
    `[['seasonality', 'competitor_price'], ['seasonality', 'macro_indicator']]`.
-   `ml_models_y` *(List[str])* / `ml_models_t` *(List[str])*: Candidate models
    for AutoML. Example: `['lgbm', 'xgboost', 'rf']`.
-   `n_folds_list` *(List[int])*: Candidate fold numbers for cross-fitting.
    Example: `[5, 10]`.
-   `output_dir` *(str)*: Path to the directory where results and artifacts will
    be saved. Example: `'./results/causal_output/'`.
-   `n_jobs` *(int, default=-1)*: Number of parallel jobs. `-1` uses all
    available CPU cores. Example: `-1` or `4`.
-   `covariates_for_cate_list` *(Optional[List[List[str]]])*: Variables to
    estimate Conditional Average Treatment Effects (CATE). Example:
    `[['geo_region'], ['store_type']]`.
-   `cate_structure_list` *(Optional[List[Any]])*: Specifies the CATE functional
    form. Example: `['auto_additive', 'auto_additive']`.
-   `time_budget` *(int, default=15)*: Time limit in seconds for FLAML AutoML
    tuning per model. Example: `30`.
-   `date_col` *(Optional[str])*: Date column name. Example: `'date'`.
-   `geo_col` *(Optional[str])*: Geography column name. Example: `'region_id'`.
-   `item_col` *(Optional[str])*: Item column name. Example: `'product_id'`.
-   `geo_name_col` *(Optional[str])*: Column containing geographical names (used
    for descriptive aggregation). Example: `'region_name'`.
-   `item_name_col` *(Optional[str])*: Column containing item names (used for
    descriptive aggregation). Example: `'product_name'`.
-   `run_sensitivity` *(bool, default=False)*: Whether to execute robust
    sensitivity bounds analysis. Example: `True`.
-   `sensitivity_scope` *(Optional[List[str]], default=['total'])*: Scope for
    sensitivity bounds analysis. Options include `'total'`, `'geo'`, or
    `'item'`. Example: `['total', 'item']`.
-   `ground_truth_existence` *(bool, default=False)*: Set to `True` if ground
    truth treatment effects are available for evaluation. Example: `False`.
-   `ground_truth_effect_column` *(Optional[List[str]], default=[])*: Column
    names containing ground truth effects for each treatment variable. Example:
    `['true_effect_discount', 'true_effect_tv']`.
-   `treatment_types` *(Optional[Dict[str, str]], default=None)*: Treatment type
    classification. Default type is `"spend"`. Note: Phase 4 ROI charts are
    generated only for `"percentage"`, `"impressions"`, or `"price"` (requires
    `treatment_spend_cols`). For `"spend"`, they are skipped as they are
    redundant with Phase 5 optimization. Example:
    `{'discount_rate': 'percentage', 'tv_spend': 'spend'}`.
-   `treatment_spend_cols` *(Optional[Dict[str, str]], default=None)*: Maps
    treatment variables to their monetary spend columns. Automatically maps to
    the treatment itself if type is `"spend"`. Example:
    `{'discount_rate': 'discount_cost'}`.
-   `sales_col` *(Optional[str], default=None)*: Outcome sales column, used for
    calculating ROI and optimization. Example: `'total_sales'`.
-   `peak_months` *(Optional[List[int]], default=None)*: List of peak months
    (1-12) to highlight in ROI charts. Defaults to `[3, 7, 11, 12]` if not
    specified. Example: `[11, 12]`.
-   `optimize` *(bool, default=False)*: Whether to run the optimization phase
    (Phase 5). Example: `True`.
-   `simple_optimization` *(bool, default=True)*: Currently, optimization is
    calculated only if `optimize = True` and `simple_optimization = True` with a
    single treatment. Advanced optimization (`simple_optimization = False`) and
    optimization for multiple treatments are under development and will not be
    calculated. Example: `True`.
-   `opt_periods` *(Optional[tuple[Any, Any]], default=None)*: Tuple of start
    and end periods for optimization. Example: `('2023-01-01', '2023-12-31')`.
-   `opt_threshold_roi` *(float, default=0.0)*: Threshold ROI for optimization.
    Example: `1.5`.
-   `opt_other_cost_variables` *(Optional[List[str]], default=None)*: Other cost
    variables to include in optimization. Example: `['fixed_marketing_cost']`.

#### `run_full_pipeline(self, df, scaler, top_n=5, n_reps=20, process_log='lightweight')`

Executes the full 4-phase pipeline (Grid Search -> Shortlisting -> Ensemble ->
Consolidation). **Parameters:**

-   `df` *(pd.DataFrame)*: The raw input dataset. Example: `pd.read_csv('data.csv')`.
-   `scaler` *(CausalDataScaler)*: Initialized scaler object. Example: `CausalDataScaler(standardize_cols=['spend'])`.
-   `top_n` *(int, default=5)*: Number of top models to select for the final
    ensemble. Example: `3`.
-   `n_reps` *(int, default=20)*: Number of repetitions for the robust ensemble
    step. Example: `10`.
-   `process_log` *(str, default='lightweight')*: Log level. If `'lightweight'`,
    the process will skip generating `1a_full_df_<model>.csv` in
    `step1_exploration` to save disk space. Another value is `'full'`. Example:
    `'full'`.

#### `format_cate_equation(self, coef_dict)`

Transforms CATE regression coefficients into a readable algebraic equation text
string. **Parameters:**

-   `coef_dict` *(Dict[str, float])*: Regression coefficients dictionary. Example: `{'Intercept': 1.5, 'region_NA': 0.2}`.
    **Returns:**

-   `str`: Formatted algebraic equation string.

#### `extract_priors(self, df_1a_consolidated, treatment_base_name, prior_type='roi', distribution_type='Normal')`

Generates shape parameters (mu, sigma, alpha, beta) for probabilistic prior
matching, useful for Bayesian Media Mix Modeling. **Parameters:**

-   `df_1a_consolidated` *(pd.DataFrame)*: Consolidated pipeline results. Example: `df_1a`.
-   `treatment_base_name` *(str)*: Treatment base name (excluding lag suffixes). Example: `'tv_spend'`.
-   `prior_type` *(str, default='roi')*: Prior type to extract. Either `'roi'`
    or `'contribution'`. Example: `'roi'`.
-   `distribution_type` *(str, default='Normal')*: Distribution to fit. Options:
    `'Normal'`, `'LogNormal'`, `'Beta'`. Example: `'Normal'`. **Returns:**

-   `Dict[str, float]`: Dictionary of fitted distribution parameters.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.preprocessing.scalers`

### Class: `CausalDataScaler`

Manages data scaling operations safely, preserving original units for reverse
transformation to ensure accurate business impact metrics (KPI, ROI).

#### `__init__(self, standardize_cols=None, population_standardize_cols=None, population_median_normalize_cols=None, min_max_cols=None, population_col='population')`

Initializes specific scaling strategies. Issues a `UserWarning` if
`population_col` is erroneously passed into `population_median_normalize_cols`
to prevent singularity bugs. **Parameters:**

-   `standardize_cols` *(Optional[List[str]])*: Columns to apply standard
    scaling. Example: `['tv_spend', 'search_spend']`.
-   `population_standardize_cols` *(Optional[List[str]])*: Columns to apply
    population standardization. *(Note: Placeholder, not implemented).* Example:
    `['competitor_sales']`.
-   `population_median_normalize_cols` *(Optional[List[str]])*: Columns to apply
    population median normalization. Example: `['sales']`.
-   `min_max_cols` *(Optional[List[str]])*: Columns to apply Min-Max scaling.
    *(Note: Placeholder, not implemented).* Example: `['temperature']`.
-   `population_col` *(str, default='population')*: Column specifying the
    population for normalization. Example: `'population'`.

#### `fit_transform(self, df: pd.DataFrame) -> pd.DataFrame`

Fits the scaling parameters and returns a normalized copy of the DataFrame.
Preserves a hidden raw population column for accurate descaling.

#### `inverse_transform_column(self, df_norm: pd.DataFrame, target_col: str, param_source_col: Optional[str] = None) -> pd.Series`

Reverts a scaled column (e.g., predicted Incremental KPI) back to its original
magnitude based on fitted parameters. Gracefully acts as a pass-through for
unscaled columns.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.preprocessing.features`

Vectorized, high-performance generation of lagged variables. Utilizes
`pd.Series.reindex` with exact date-matching if `date_col` is provided,
preventing alignment errors on discontinuous dates.

### Function: `create_lagged_features`

#### `create_lagged_features(df, columns, max_lag, date_col, time_freq='daily', unit_col=None, na_fill=True)`

Creates lagged features for specified columns in the dataframe.

**Parameters:**

-   `df` *(pd.DataFrame)*: The input dataframe. Example: `raw_df`.
-   `columns` *(List[str])*: List of column names to create lags for. Example: `['tv_spend']`.
-   `max_lag` *(int)*: The maximum number of lags to generate (e.g., if 3,
    generates `_l1`, `_l2`, `_l3`). Example: `3`.
-   `date_col` *(str)*: Column specifying the date/time for exact chronological
    lagging. Example: `'date'`.
-   `time_freq` *(str, default='daily')*: Time frequency for lagging. Options:
    `'daily'`, `'weekly'`, `'monthly'`. Example: `'weekly'`.
-   `unit_col` *(Optional[str], default=None)*: Optional column to group by
    before shifting (essential for Geo/Panel data). Example: `'geo_id'`.
-   `na_fill` *(bool, default=True)*: If True, fills the resulting NaN values
    with 0. Example: `True`.

**Returns:**

-   `pd.DataFrame`: A new dataframe with the lagged columns added.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.evaluation.runner`

### Class: `DoubleMLPipeline`

Wraps the `doubleml` library logic, integrating FLAML estimators and `patsy`
spline basis generation for CATE estimation.

#### `__init__(self, y_col, d_col, x_cols, ml_y_model, ml_t_model, n_folds=5, time_budget=15, cate_structure='auto_additive', covariates_for_cate=None, ground_truth_col=None, cluster_cols=None)`

Initializes the DoubleML PLR pipeline. **Parameters:**

-   `y_col` *(str)*: Outcome variable. Example: `'sales'`.
-   `d_col` *(str)*: Treatment variable. Example: `'discount_rate'`.
-   `x_cols` *(List[str])*: Confounding variables. Example: `['seasonality', 'competitor_price']`.
-   `ml_y_model` *(str)*: Candidate ML model for Y nuisance estimation. Example: `'lgbm'`.
-   `ml_t_model` *(str)*: Candidate ML model for T nuisance estimation. Example: `'xgboost'`.
-   `n_folds` *(int, default=5)*: Number of cross-fitting folds. Example: `5`.
-   `time_budget` *(int, default=15)*: AutoML tuning time budget in seconds. Example: `15`.
-   `cate_structure` *(Union[str, Dict[str, Any]], default='auto_additive')*:
    CATE functional form specifications. Example: `'auto_additive'`.
-   `covariates_for_cate` *(Optional[List[str]], default=None)*: Covariates used
    for CATE estimation. Example: `['geo_region']`.
-   `ground_truth_col` *(Optional[str], default=None)*: Column name for ground
    truth treatment effect. Example: `'true_effect'`.
-   `cluster_cols` *(Optional[List[str]], default=None)*: Column names used for
    cluster robust inference. Example: `['geo_region']`.

#### `run(self, df_orig: pd.DataFrame, scaler: CausalDataScaler, rep_id: int = 1) -> Dict[str, Any]`

Executes a single DoubleML Partial Linear Regression (PLR) pipeline run.
**Returns:** A dictionary containing the model name, result dataframe (`df_1a`),
execution metrics (`metrics_1c`), the DoubleML model object, and CATE
coefficients.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.evaluation.selection`

### Function: `shortlist_top_models`

#### `shortlist_top_models(df_metrics, top_n=5, y_rmse_col='nuisance_models_rmse_Y_model_evaluated_by_AutoML', t_rmse_col='nuisance_models_rmse_T_model_evaluated_by_AutoML')`

Shortlists top N performing estimator combinations using combined normalized
RMSE.

**Parameters:**

-   `df_metrics` *(pd.DataFrame)*: DataFrame containing evaluation metrics of
    models. Example: `df_metrics`.
-   `top_n` *(int, default=5)*: Number of top models to select for the final
    ensemble. Example: `3`.
-   `y_rmse_col` *(str)*: Column name for the outcome model RMSE. Example: `'nuisance_models_rmse_Y_model_evaluated_by_AutoML'`.
-   `t_rmse_col` *(str)*: Column name for the treatment model RMSE. Example: `'nuisance_models_rmse_T_model_evaluated_by_AutoML'`.

**Returns:**

-   `pd.DataFrame`: Sorted DataFrame with a `shortlisted` indicator column.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.evaluation.sensitivity`

### Function: `perform_gate_and_sensitivity_analysis`

#### `perform_gate_and_sensitivity_analysis(dml_model, df_orig, gate_cols, benchmark_covariates=None)`

Performs GATE (Group Average Treatment Effect) and robust sensitivity bounds
checks.

**Parameters:**

-   `dml_model` *(doubleml.DoubleMLPLR)*: Fitted DoubleML PLR model.
-   `df_orig` *(pd.DataFrame)*: Raw original input dataset.
-   `gate_cols` *(List[str])*: Column names to define groups for GATE
    estimation.
-   `benchmark_covariates` *(Optional[List[str]], default=None)*: Covariates to
    use as benchmark for sensitivity analysis.

**Returns:**

-   `Dict[str, Any]`: Dictionary containing the GATE summary, sensitivity
    elements, benchmark results, and groups dataframe.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.evaluation.metrics`

### Function: `calculate_smape`

#### `calculate_smape(y_true, y_pred)`

Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between true and
predicted values.

**Parameters:**

-   `y_true` *(pd.Series)*: True values.
-   `y_pred` *(pd.Series)*: Predicted values.

**Returns:**

-   `float`: The SMAPE percentage.

### Function: `calculate_ground_truth_metrics`

#### `calculate_ground_truth_metrics(y_true, y_pred, total_input)`

Calculates accuracy metrics against ground truth data, including SMAPE,
R-squared, MAE, and RMSE.

**Parameters:**

-   `y_true` *(pd.Series)*: Ground truth values.
-   `y_pred` *(pd.Series)*: Predicted values.
-   `total_input` *(float)*: Total treatment input for ROI calculation.

**Returns:**

-   `Dict[str, float]`: Dictionary containing `true_total_roi`, `SMAPE`,
    `R_squared`, `MAE`, and `RMSE`.

### Function: `evaluate_nuisance_residuals`

#### `evaluate_nuisance_residuals(residuals, treatment, covariates=None)`

Evaluates the properties of nuisance residuals, including variance ratio,
skewness, and normality.

**Parameters:**

-   `residuals` *(pd.Series)*: Nuisance model residuals.
-   `treatment` *(pd.Series)*: Treatment variable values.
-   `covariates` *(Optional[pd.DataFrame], default=None)*: Confounding variables
    to check correlation.

**Returns:**

-   `Dict[str, float]`: Dictionary containing residual properties
    (`residual_mean`, `residual_skewness`, `variance_ratio`,
    `normaltest_pvalue`, `max_correlation`).

### Function: `aggregate_geo_item_metrics`

#### `aggregate_geo_item_metrics(df, model_name, d_col, y_col, geo_col=None, item_col=None, geo_name_col=None, item_name_col=None, gt_col=None)`

Aggregates estimated KPIs and ROI metrics across specified geographical and item
dimensions.

**Parameters:**

-   `df` *(pd.DataFrame)*: Results DataFrame.
-   `model_name` *(str)*: Model name.
-   `d_col` *(str)*: Treatment column.
-   `y_col` *(str)*: Outcome column.
-   `geo_col` / `item_col` *(Optional[str])*: Grouping dimension columns.
-   `geo_name_col` / `item_name_col` *(Optional[str])*: Descriptive name columns
    for dimensions.
-   `gt_col` *(Optional[str])*: Ground truth KPI column.

**Returns:**

-   `pd.DataFrame`: Aggregated metrics DataFrame.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.visualization.plots`

A suite of publication-ready visualization tools generating Matplotlib
artifacts. Handles dynamic layout scaling based on data density (e.g.,
automatically thinning x-axis labels on multi-year daily data to prevent
overlapping).

### Function: `plot_incremental_kpi_line`

#### `plot_incremental_kpi_line(df, date_col, kpi_col, lower_col, upper_col, out_path, title, gt_col=None, actual_y_col=None)`

Plots a line chart showing estimated incremental KPI over time with 95%
confidence intervals. Automatically overlays ground truth and actual total KPIs
if provided.

**Parameters:**

-   `df` *(pd.DataFrame)*: Input DataFrame.
-   `date_col` *(str)*: Date column name.
-   `kpi_col` *(str)*: Estimated incremental KPI column name.
-   `lower_col` *(str)*: Confidence interval lower bound column name.
-   `upper_col` *(str)*: Confidence interval upper bound column name.
-   `out_path` *(str)*: Path to save the generated plot image.
-   `title` *(str)*: Plot title.
-   `gt_col` *(Optional[str], default=None)*: Ground truth KPI column name.
-   `actual_y_col` *(Optional[str], default=None)*: Actual total KPI column
    name.

### Function: `plot_incremental_kpi_bar_stacked`

#### `plot_incremental_kpi_bar_stacked(df, date_col, kpi_cols, out_path, title, gt_col=None, actual_y_col=None)`

Plots a stacked bar chart showing the composition of estimated incremental KPIs
across multiple treatment variables over time.

**Parameters:**

-   `df` *(pd.DataFrame)*: Input DataFrame.
-   `date_col` *(str)*: Date column name.
-   `kpi_cols` *(List[str])*: List of estimated incremental KPI column names to
    stack.
-   `out_path` *(str)*: Path to save the generated plot image.
-   `title` *(str)*: Plot title.
-   `gt_col` *(Optional[str], default=None)*: Ground truth KPI column name.
-   `actual_y_col` *(Optional[str], default=None)*: Actual total KPI column
    name.

### Function: `plot_roi_timeseries_bar`

#### `plot_roi_timeseries_bar(df, date_col, input_col, kpi_col, lower_col, upper_col, out_path, title, gt_col=None, actual_y_col=None)`

Generates a comprehensive 3-panel dynamic chart plotting Total Input,
Incremental KPI, and ROI over time. Includes robust handling for negative
confidence bounds.

**Parameters:**

-   `df` *(pd.DataFrame)*: Input DataFrame.
-   `date_col` *(str)*: Date column name.
-   `input_col` *(str)*: Treatment input column name.
-   `kpi_col` *(str)*: Estimated incremental KPI column name.
-   `lower_col` *(str)*: Confidence interval lower bound column name.
-   `upper_col` *(str)*: Confidence interval upper bound column name.
-   `out_path` *(str)*: Path to save the generated plot image.
-   `title` *(str)*: Plot title.
-   `gt_col` *(Optional[str], default=None)*: Ground truth KPI column name.
-   `actual_y_col` *(Optional[str], default=None)*: Actual total KPI column
    name.

### Function: `plot_entity_roi_comparison`

#### `plot_entity_roi_comparison(df, entity_col, input_col, kpi_col, lower_col, upper_col, out_path, title, gt_col=None, actual_y_col=None)`

Generates a cross-sectional 3-panel bar chart comparing inputs, incremental
performance, and ROI across specific entities (e.g., Geo locations or Items).

**Parameters:**

-   `df` *(pd.DataFrame)*: Input DataFrame.
-   `entity_col` *(str)*: Entity column name (e.g., `'Geo'`).
-   `input_col` *(str)*: Treatment input column name.
-   `kpi_col` *(str)*: Estimated incremental KPI column name.
-   `lower_col` *(str)*: Confidence interval lower bound column name.
-   `upper_col` *(str)*: Confidence interval upper bound column name.
-   `out_path` *(str)*: Path to save the generated plot image.
-   `title` *(str)*: Plot title.
-   `gt_col` *(Optional[str], default=None)*: Ground truth KPI column name.
-   `actual_y_col` *(Optional[str], default=None)*: Actual total KPI column
    name.

### Function: `plot_sensitivity_contour`

#### `plot_sensitivity_contour(dml_model, out_path, title, benchmark_covariates=None)`

Plots the Omitted Variable Bias (OVB) bounds via sensitivity contours. Displays
estimated Robustness Values (RV) with robust percentile clipping to avoid
rendering artifacts at mathematical singularities.

**Parameters:**

-   `dml_model` *(doubleml.DoubleMLPLR)*: Fitted DoubleML PLR model.
-   `out_path` *(str)*: Path to save the generated plot image.
-   `title` *(str)*: Plot title.
-   `benchmark_covariates` *(Optional[List[str]], default=None)*: Covariates to
    use as benchmark for sensitivity analysis.

### Function: `plot_cate_scatter_matrix`

#### `plot_cate_scatter_matrix(df_1a_consolidated, d_cols, cont_cols, bin_cols, eq_str_dict, out_path)`

Visualizes the heterogeneity of treatment effects (CATE) against specific
continuous or binary covariates using scatter plots with overlaid gradient
contours. Includes formatted CATE mathematical equations.

**Parameters:**

-   `df_1a_consolidated` *(pd.DataFrame)*: Consolidated results DataFrame.
-   `d_cols` *(List[str])*: List of treatment variable column names.
-   `cont_cols` *(List[str])*: List of continuous covariates to plot against.
-   `bin_cols` *(List[str])*: List of binary covariates to plot against.
-   `eq_str_dict` *(Dict[str, str])*: Dictionary containing CATE equations as
    strings.
-   `out_path` *(str)*: Path to save the generated plot image.

### Function: `plot_prior_distributions`

#### `plot_prior_distributions(roi_array, contrib_array, out_path, d_col_name)`

Fits and overlays Normal, LogNormal, and Beta distributions on effect outcomes.
Designed specifically to export data shapes compatible with Bayesian Media Mix
Modeling (MMM) platforms like LightweightMMM.

**Parameters:**

-   `roi_array` *(np.ndarray)*: Array of ROI estimates.
-   `contrib_array` *(np.ndarray)*: Array of contribution ratio estimates.
-   `out_path` *(str)*: Path to save the generated plot image.
-   `d_col_name` *(str)*: Treatment column name.

--------------------------------------------------------------------------------

## Module: `doubleml_pipeline.models.flaml_dml`

### Class: `FlamlRegressorDoubleML`

A wrapper for FLAML regression models to be used as an estimator within the
DoubleML framework.

#### `__init__(self, time, estimator_list, metric, verbose=0, **kwargs)`

Initializes the wrapper indicating the time budget, estimator candidates, and
metrics.

**Parameters:**

-   `time` *(int)*: Time budget in seconds for FLAML tuning.
-   `estimator_list` *(List[str])*: List of candidate estimators (e.g.,
    `['lgbm', 'xgboost']`).
-   `metric` *(str)*: Metric to optimize during tuning (e.g., `'r2'`).
-   `verbose` *(int, default=0)*: FLAML verbosity level.
-   `**kwargs` *(Any)*: Additional arguments passed to the FLAML AutoML
    constructor.
