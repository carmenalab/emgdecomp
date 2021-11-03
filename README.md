# EMGDecomp

![](https://img.shields.io/pypi/v/emgdecomp) [![DOI](https://zenodo.org/badge/423892522.svg)](https://zenodo.org/badge/latestdoi/423892522)

Package for decomposing EMG signals into motor unit firings, created for Formento et al 2021. Based heavily on Negro et al, 2016. Supports GPU via CUDA and distributed computation via Dask.


## Installation

```bash
pip install emgdecomp
```

For those that want to either use [Dask](https://dask.org/) and/or [CUDA](https://cupy.dev/), you can alternatively run:

```bash
pip install emgdecomp[dask]
pip install emgdecomp[cuda]
```

## Usage

### Basic

```python
# data should be a numpy array of n_channels x n_samples
sampling_rate, data = fetch_data(...)

decomp = EmgDecomposition(
  params=EmgDecompositionParams(
    sampling_rate=sampling_rate
  ))

firings = decomp.decompose(data)
print(firings)
```

The resulting `firings` object is a NumPy structured array containing the columns `source_idx`, `discharge_samples`, and `discharge_seconds`. `source_idx` is a 0-indexed ID for each "source" learned from the data; each source is a putative motor unit. `discharge_samples` indicates the sample at which the source was detected as "firing"; note that the algorithm can only detect sources up to a delay. `discharge_seconds` is the conversion of `discharge_samples` into seconds via the passed-in sampling rate.

As a structured NumPy array, the resulting `firings` object is suitable for conversion into a Pandas DataFrame:

```python
import pandas as pd
print(pd.DataFrame(firings))
```

And the "sources" (i.e. components corresponding to motor units) can be interrogated as needed via the `decomp.model` property:

```python
model = decomp.model
print(model.components)
```

### Advanced

Given an already-fit `EmgDecomposition` object, you can then decompose a new batch of EMG data with its existing sources via `transform`:

```python
# Assumes decomp is already fit
new_data = fetch_more_data(...)
new_firings = decomp.transform(new_data)
print(new_firings)
```

Alternatively, you can add new sources (i.e. new putative motor units) while retaining the existing sources with `decompose_batch`:

```python
# Assumes decomp is already fit

more_data = fetch_even_more_data(...)
# Firings corresponding to sources that were both existing and newly added
firings2 = decomp.decompose_batch(more_data)
# Should have at least as many components as before decompose_batch()
print(decomp.model.components)
```

Finally, basic plotting capabilities are included as well:

```python
from emgdecomp.plots import plot_firings, plot_muaps
plot_muaps(decomp, data, firings)
plot_firings(decomp, data, firings)
```

### File I/O
The `EmgDecomposition` class is equipped with `load` and `save` methods that can save/load parameters to disk as needed; for example:

```python
with open('/path/to/decomp.pkl', 'wb') as f:
  decomp.save(f)

with open('/path/to/decomp.pkl', 'rb') as f:
  decomp_reloaded = EmgDecomposition.load(f)
```

### Dask and/or CUDA
Both Dask and CUDA are supported within EmgDecomposition for support for distributed computation across workers and/or use of GPU acceleration. Each are controlled via the `use_dask` and `use_cuda` boolean flags in the `EmgDecomposition` constructor.

### Parameter Tuning

See the list of parameters in [EmgDecompositionParameters](https://github.com/carmenalab/emgdecomp/blob/master/emgdecomp/parameters.py). The defaults on `master` are set as they were used for Formento et. al, 2021 and should be reasonable defaults for others.

## Documentation
See documentation on classes `EmgDecomposition` and `EmgDecompositionParameters` for more details.

## Acknowledgements
If you enjoy this package and use it for your research, you can:

- cite the Journal of Neural Engineering paper, Formento et. al 2021, for which this package was developed: TODO
- cite this github repo using its DOI: 10.5281/zenodo.5641426
- star this repo using the top-right star button.

## Contributing / Questions

Feel free to open issues in this project if there are questions or feature requests. Pull requests for feature requests are very much encouraged, but feel free to create an issue first before implementation to ensure the desired change sounds appropriate.
