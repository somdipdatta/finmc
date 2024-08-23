# Hull-White

<figure markdown="1">
  ![Simulation](../images/hw_rate_mc.png)
</figure>

### Model

In the Hull White model, the short-rate follows the following process.
$$
dr_t = [\theta_t - a r_t]dt + \sigma dW_t
$$

where \(dW_t\) is a Wiener process.

The model specific component in the dataset (`HW`) is a dict with two parameters, and the name of the asset:

* \(a\), the mean reversion rate (MEANREV)).
* \(\sigma\), the volatility of rate (VOL).

\(\theta_t\) is calibrated by the model from the zero rate curve.

### Example

```python
from finmc.models.hullwhite import HullWhiteMC

dataset = {
    "MC": {"PATHS": 100_000, "TIMESTEP": 1 / 250, "SEED": 1},
    "BASE": "USD",
    "ASSETS": {"USD": ("ZERO_RATES", np.array([[2.0, 0.05]]))},
    "HW": {
        "ASSET": "USD",
        "MEANREV": 0.1,
        "VOL": 0.03,
    },
}

model = HullWhiteMC(dataset)
model.advance(1.0)
discount_factors = model.get_df()
```
