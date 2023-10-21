# Version 0.0.6
- Introduced check if 't' column is present, otherwise use 'z' column to handle models with only one coefficient

# Version 0.0.5
- Replaced explicit options by flexible `**options` parameter
- Supported classes: `statsmodels.regression.linear_model.RegressionResultsWrapper` and `linearmodels.panel.result.PanelEffectsresult`
- Introduced `ALLOWED_OPTIONS` global

# Version 0.0.4
- First working version with statsmodels support