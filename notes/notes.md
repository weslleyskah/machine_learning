## Datframes and Arrays

### DataFrames:
```python
dataset = dataset.to_numpy()                      # Converts DataFrame to 2D array
dataset_label = dataset_label.to_numpy().ravel()  # Converts DataFrame/Series to 1D array
```

### Arrays:
```python
dataset = dataset                             # Leave as is
dataset_label = dataset_label.ravel()         # Flatten labels if needed
```

### Example:
```python
algorithm.fit(dataset.to_numpy(), dataset_label.to_numpy().ravel())
```