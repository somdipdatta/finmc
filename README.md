# finmc

[![CI](https://github.com/somdipdatta/finmc/actions/workflows/main.yml/badge.svg)](https://github.com/somdipdatta/finmc/actions/workflows/main.yml)

This package contains Monte-Carlo implementations of many financial models derived from a common interface class. This interface allows computation of instruments with european, and american payoffs, as well as path dependent calculations.

See complete [documentation here.](https://somdipdatta.github.io/finmc/)

<table>
<tr>
<td> <img src="https://raw.githubusercontent.com/somdipdatta/finmc/main/images/impliedvol.png" alt="implied vol"/> </td>
<td> <img src="https://raw.githubusercontent.com/somdipdatta/finmc/main/images/rate_mc.png" alt="rate mc"/> </td>
</tr>
</table>

## Install it from PyPI

```bash
pip install finmc
```

### Example
This is an example of pricing a vanilla option using the local volatility model.

```py
import numpy as np
from finmc.models.localvol import LVMC
from finmc.calc.option import opt_price_mc

# Define Dataset with zero rate curve, and forward curve.
dataset = {
    "MC": {"PATHS": 100_000, "TIMESTEP": 1 / 250},
    "BASE": "USD",
    "ASSETS": {
        "USD":("ZERO_RATES", np.array([[2.0, 0.05]])),
        "SPX": ("FORWARD", np.array([[0.0, 5500], [1.0, 5600]])),
        },
    "LV": {"ASSET": "SPX", "VOL": 0.3},
}

model = LVMC(dataset)
price = opt_price_mc(5500.0, 1.0, "Call", "SPX", model)
```
