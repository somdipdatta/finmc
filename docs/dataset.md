# Dataset

The dataset for finmc models is a dictionary with the following components

 - **BASE** String containing the name of the base asset, i.e. the currency in which the price is denominated. e.g. "USD".
 - **MC** Dict containing common MC parameters.
 - **ASSETS** Dict containing forwards of all assets in the model, including the base asset. See [Forwards](forwards.md) and [Rates](rates.md) for more.
 - **{Model Name}** Dict containing parameters specific to the model. See the [models](./models/api.md) section for more.


### MC Parameters

The MC section has the following parameters.

- **PATHS**: The number of Monte-Carlo paths.
- **TIMESTEP**: The incremental timestep of simulation (in years). 
- **SEED** (Optional): The seed for the random number generator.

e.g.
```python
"MC": {
    "PATHS": 100_000,
    "TIMESTEP": 1 / 250,
    "SEED": 1,
},
```

### Complete Example
```python
import numpy as np

dataset = {
    "BASE": "USD",
    "ASSETS": {
        "USD": ("ZERO_RATES", np.array([[2.0, 0.05]])),
        "SPX": ("FORWARD", np.array([[0.0, 5500], [1.0, 5600]])),
    },
    "MC": {
        "PATHS": 100_000,
        "TIMESTEP": 1 / 250,
        "SEED": 1,
    },
    "HESTON": {
        "ASSET": "SPX",
        "INITIAL_VAR": 0.015,
        "LONG_VAR": 0.052,
        "VOL_OF_VAR": 0.88,
        "MEANREV": 2.78,
        "CORRELATION": -0.85,
    }
}
```
