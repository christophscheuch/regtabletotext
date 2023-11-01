# Version 0.0.9
- Added support for list of `linearmodels.panel.result.PanelEffectsresult` results 

# Version 0.0.8
- Added support for 'arch.univariate.base.ARCHModelResult'
- Simplified and extended `calculate_residuals_statistics()`
- Added covariance type and fixed effects table to `linearmodels.panel.result.PanelEffectsresult` print result
- Moved default values to function definitions
- Updated required install to pandas only
- Added support for `arch.univariate.base.ARCHModelResult`

# Version 0.0.7
- Moved type checks to functions and globals
- Removed unnecessary whitespace from model formula
- Introduced `max_width` with default to 64 characters 

# Version 0.0.6
- Introduced check if 't' column is present, otherwise use 'z' column to handle models with only one coefficient

# Version 0.0.5
- Replaced explicit options by flexible `**options` parameter
- Supported classes: `statsmodels.regression.linear_model.RegressionResultsWrapper` and `linearmodels.panel.result.PanelEffectsresult`
- Introduced `ALLOWED_OPTIONS` global

# Version 0.0.4
- First working version with statsmodels support