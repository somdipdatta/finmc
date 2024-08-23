# Forwards

You can describe any asset forward using a two-column (N X 2) numpy array, where the first column is time, and the second represents forwards, e.g.

```python
fwd_data = ("FORWARD", np.array([
    [0.0, 5500],
    [1.0, 5600],
    [2.0, 5700]
]))

dataset = {
    "MC": ...,
    "BASE": "USD",
    "ASSETS": {
        "USD": ...,
        "SPX": ("FORWARD", fwd_data),
    },
    ...model specific parameters
}
```


or alternatively, using `np.column_stack` from two arrays


```python
spot = 2900
div_rate = 0.01
times = np.array([0.0, 1.0, 2.0, 5.0])
rates = np.array([0.04, 0.04, 0.045, 0.05])
fwds = spot * np.exp((rates - div_rate) * times)
fwd_data = ("FORWARDS", np.column_stack((times, fwds)))
```

For complete dataset see [dataset](dataset.md)