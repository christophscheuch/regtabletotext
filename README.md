# regtabletotext: helpers to print regression output as text

## What is regtabletotext?

This package is a collection of helper functions to print regression output of the Python packages `statsmodels`, `linearmodels`, and `arch` as text strings. The helpers are particularly useful for users who want to render regression output in [Quarto](https://quarto.org/) to HTML and PDF.

If you want to export your results to LaTeX or HTML, please check out [stargazer](https://pypi.org/project/stargazer/).

## How to install regtabletotext?

The package is available on [pypi.org/project/regtabletotext](https://pypi.org/project/regtabletotext/):

```
pip install regtabletotext
```

## How to use regtabletotext?

Currently supported model types:

- `statsmodels.regression.linear_model.RegressionResultsWrapper`
- `linearmodels.panel.results.PanelEffectsResults`
- `arch.univariate.base.ARCHModelResult`

### For statsmodels regression output

The following code chunk
```
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from regtabletotext import prettify_result

n = 100 
data = pd.DataFrame({
    'age': np.random.randint(18, 65, n),
    'income': np.random.normal(50, 20, n) ,
    'education': np.random.randint(12, 22, n),
    'hours_worked': np.random.randint(20, 60, n),
    'satisfaction': np.random.randint(1, 11, n) 
})

mod = smf.ols(formula='income ~ age + education + hours_worked', data=data)
res = mod.fit()

prettify_result(res)
```

returns this text
```
OLS Model:
Lottery ~ Literacy + Wealth + Region

Coefficients:
             Estimate  Std. Error  Statistic  p-Value
Intercept      38.652       9.456      4.087    0.000
Region[T.E]   -15.428       9.727     -1.586    0.117
Region[T.N]   -10.017       9.260     -1.082    0.283
Region[T.S]    -4.548       7.279     -0.625    0.534
Region[T.W]   -10.091       7.196     -1.402    0.165
Literacy       -0.186       0.210     -0.886    0.378
Wealth          0.452       0.103      4.390    0.000

Summary statistics:
- Number of observations: 85
- R-squared: 0.338, Adjusted R-squared: 0.287
- F-statistic: 6.636 on 6 and 78 DF, p-value: 0.000
```

### For linearmodels regression output

This code chunk
```
import pandas as pd
import numpy as np
from linearmodels import PanelOLS
from regtabletotext import prettify_result

np.random.seed(42)
n_entities = 100
n_time_periods = 5

data = {
    'entity': np.repeat(np.arange(n_entities), n_time_periods),
    'time': np.tile(np.arange(n_time_periods), n_entities),
    'y': np.random.randn(n_entities * n_time_periods),
    'x1': np.random.randn(n_entities * n_time_periods),
    'x2': np.random.randn(n_entities * n_time_periods)
}

df = pd.DataFrame(data)
df.set_index(['entity', 'time'], inplace=True)

formula = 'y ~ x1 + x2 + EntityEffects + TimeEffects'
model = PanelOLS.from_formula(formula, df)
result = model.fit()

prettify_result(result, options={'digits':2, 'include_residuals': True})
```
produces this output
```
Panel OLS Model:
y ~ x1 + x2 + EntityEffects + TimeEffects

Covariance Type: Unadjusted

Residuals:
 Mean  Std  Min   25%  50%  75%  Max
  0.0 0.87 -2.7 -0.62  0.0 0.58 3.16

Coefficients:
    Estimate  Std. Error  t-Statistic  p-Value

x1     -0.06        0.05        -1.12     0.26
x2     -0.04        0.05        -0.80     0.42

Included Fixed Effects:
        Total
Entity  100.0
Time      5.0

Summary statistics:
- Number of observations: 500
- R-squared (incl. FE): 0.21, Within R-squared: 0.01
- F-statistic: 1.05, p-value: 0.35
```
For multiple models, you can also use `prettify_result`:
```
formula = 'y ~ x1 + x2'
model = PanelOLS.from_formula(formula, df)
result1 = model.fit()

formula = 'y ~ x2 + EntityEffects'
model = PanelOLS.from_formula(formula, df)
result2 = model.fit()

formula = 'y ~ x1 + x2 + EntityEffects + TimeEffects'
model = PanelOLS.from_formula(formula, df)
result3 = model.fit()

results = [result1, result2, result3]

prettify_result(results)
```
The result is:
```
Dependent var.        y               y               y

x1              -0.072 (-1.59)                  -0.058 (-1.12)
x2              -0.049 (-1.14)  -0.049 (-0.98)  -0.041 (-0.8)

Fixed effects                       Entity       Entity, Time
VCOV type         Unadjusted      Unadjusted      Unadjusted
Observations         500             500             500
R2 (incl. FE)       0.008           0.198           0.208
Within R2           0.006           0.002           0.006
```

### For arch estimation output

For arch estimations like these:
```
import datetime as dt
import pandas_datareader.data as web
from arch import arch_model
import arch.data.sp500
from regtabletotext import prettify_result

start = dt.datetime(1988, 1, 1)
end = dt.datetime(2018, 1, 1)
data = arch.data.sp500.load()
returns = 100 * data['Adj Close'].pct_change().dropna()
am_fit = arch_model(returns).fit(update_freq=5)

prettify_result(am_fit)
```
you get:
```
Constant Mean - GARCH Model Results

Mean Coefficients:
    Estimate  Std. Error  Statistic  p-Value
mu     0.056       0.011      4.906      0.0

Coefficients for GARCH(p: 1, q: 1):
          Estimate  Std. Error  Statistic  p-Value
omega        0.018       0.005      3.738      0.0
alpha[1]     0.102       0.013      7.852      0.0
beta[1]      0.885       0.014     64.125      0.0

Summary statistics:
- Number of observations: 5,030
- Distribution: Normal distribution
- Multiple R-squared: 0.000, Adjusted R-squared: 0.000
- BIC: 13,907.530,  AIC: 13,881.437
```