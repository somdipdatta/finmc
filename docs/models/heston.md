### Model

<figure markdown="1">
  ![Simulation](../images/heston_vol_mc.png)
</figure>

In the Heston model  the lognormal stock process \(X_t\) is given by,

$$
dX_t = (\mu - \frac{\nu_t}{2}) dt + \sqrt \nu_t dW_s
$$

and the variance follows the process
$$
d \nu_t = \kappa (\theta - \nu_t) dt + \xi \sqrt \nu_tdW_t
$$

where \(dW_s\) and \(dW_t\) are Wiener processes with correlation \(\rho\).

The model specific component in the dataset (`HESTON`) is a dict with five parameters, and the name of the asset:

* \(\nu_0\), the initial variance (INITIAL_VAR).
* \(\theta\), the long variance (LONG_VAR).
* \(\rho\), the correlation (CORRELATION).
* \(\kappa\), the mean reversion rate (MEANREV)).
* \(\xi\), the volatility of the volatility (VOL_OF_VOL).

### Example

```python
from finmc.models.heston import HestonMC

heston_params = {
    "ASSET": "SPX",
    "INITIAL_VAR": 0.015,
    "LONG_VAR": 0.052,
    "VOL_OF_VOL": 0.88,
    "MEANREV": 2.78,
    "CORRELATION": -0.85,
}
```

```python
dataset = {
    "MC": {"PATHS": 100_000, "TIMESTEP": 1 / 250},
    "BASE": "USD",
    "ASSETS": {
        "USD": ("ZERO_RATES", np.array([[2.0, 0.05]])),
        "SPX": ("FORWARD", np.array([[0.0, 5500], [1.0, 5600]])),
    },
    "HESTON": heston_params
}
model = HestonMC(dataset)
model.advance(1.0)
spots = model.get_value("SPX")
```
