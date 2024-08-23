# Local-Vol

<figure markdown="1">
  ![Simulation](../images/lv_mc.png)
</figure>

### Model

In the Local Vol model the lognormal stock process \(X_t\) is given by,

$$
dX_t = (\mu - \frac{\sigma_t^2}{2}) dt + \sigma_t dW_s
$$

Where \(\sigma_t\) is a function of \(X_t\) and \(t\).

The model specific component in the dataset (`LV`) is a dict with two parameters `ASSET` and `VOL`.

### Examples

#### Constant Vol

This is an example with constant local volatility, in which case it reduces to the Black-Scholes Model.

```python
from finmc.models.localvol import LVMC

lv_params = {"ASSET": "SPX", "VOL": 0.015}

dataset = {
    "MC": {"PATHS": 100_000, "TIMESTEP": 1 / 250},
    "BASE": "USD",
    "ASSETS": {
        "USD": ("ZERO_RATES", np.array([[2.0, 0.05]])),
        "SPX": ("FORWARD", np.array([[0.0, 5500], [1.0, 5600]])),
    },
    "LV": lv_params
}
model = LVMC(dataset)
model.advance(1.0)
spots = model.get_value("SPX")
```


#### Vol Function

`VOL` can be a function as shown below.

```python
def volfn(points):
    # t is float, x_vec is a np array
    (t, x_vec) = points

    at = 5.0 * t + .01
    atm = 0.04 + 0.01 * np.exp(-at)
    skew = -1.5 * (1 - np.exp(-at)) / at
    return np.sqrt(np.maximum(0.001, atm + x_vec * skew))


lv_params = {"ASSET": "SPX", "VOL": volfn}
```

#### Vol Interpolator

`VOL` can be an interpolator as below

```python
from scipy.interpolate import RegularGridInterpolator

times = [0.01, 0.2, 1.0]
strikes = [-5.0, -0.5, -0.1, 0.0, 0.1, 0.5, 5.0]
vols = np.array([
    [2.713, 0.884, 0.442, 0.222, 0.032, 0.032, 0.032],
    [2.187, 0.719, 0.372, 0.209, 0.032, 0.032, 0.032],
    [1.237, 0.435, 0.264, 0.200, 0.101, 0.032, 0.032]
])
volinterp = RegularGridInterpolator(
    (times, strikes), vols, fill_value=None, bounds_error=False
)

lv_params = {"ASSET": "SPX", "VOL": volinterp}
```