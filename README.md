# regtabletotext: helpers to print regression output as text

## What is regtabletotext?

This package is a collection of helper functions to print regression output of the Python packages `statsmodels` and `linearmodels` as text strings. The helpers are particularly useful for users who want to render regression output in [Quarto](https://quarto.org/) to HTML and PDF.

If you want to export your results to LaTeX or HTML, please check out [stargazer](https://pypi.org/project/stargazer/).

## How to install regtabletotext?

The package is available on [pypi.org/project/regtabletotext](https://pypi.org/project/regtabletotext/):

```
pip install regtabletotext
```

## How to use regtabletotext?

### For statsmodels regression output

The following code chunk
```
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from regtabletotext import prettify_result

df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()

mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
res = mod.fit()

prettify_result(res)
```

returns this text
```
Model:
Lottery ~ Literacy + Wealth + Region

Coefficients:
             Estimate  Std. Error  t-Statistic  p-Value

Intercept      38.652       9.456        4.087    0.000
Region[T.E]   -15.428       9.727       -1.586    0.117
Region[T.N]   -10.017       9.260       -1.082    0.283
Region[T.S]    -4.548       7.279       -0.625    0.534
Region[T.W]   -10.091       7.196       -1.402    0.165
Literacy       -0.186       0.210       -0.886    0.378
Wealth          0.452       0.103        4.390    0.000

Summary statistics:
- Number of observations: 85
- Multiple R-squared: 0.338, Adjusted R-squared: 0.287
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

prettify_result(result)
```

produces this output

```
Model:
y ~ x1 + x2 + EntityEffects + TimeEffects

Coefficients:
    Estimate  Std. Error  t-Statistic  p-Value

x1    -0.058       0.052       -1.122    0.263
x2    -0.041       0.050       -0.804    0.422

Summary statistics:
- Number of observations: 500
- Overall R-squared: 0.008, Within R-squared: 0.006
- F-statistic: 1.048, p-value: 0.351
```