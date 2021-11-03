import abc
import logging
from typing import Union, Optional, Any, Tuple, Callable

import numpy as np


class EmgDataManager(abc.ABC):
    """
    Base class encapsulating all of the operations done on the EMG data. This class abstracts away the implementations
    for the various interactions with the (whitened, extended) EMG data, in order to hide whether the EMG data lives
    on the CPU or GPU, or whether Dask is being used for computation.

    All input and output values to these methods assume "vanilla" numpy arrays on the CPU, even for GPU-enabled data
    managers. For GPU-enabled methods, this entails more copying back and forth between the CPU and GPU, but we figure
    that's okay since the source vectors / source matrix are generally pretty small.
    """

    @property
    @abc.abstractmethod
    def shape(self):
        """
        Shape of the underlying data. Should be n_channels x n_samples (n_channels for the extended data)
        """
        pass

    @abc.abstractmethod
    def squared_sum(self) -> np.ndarray:
        """
        Computes the squared sum across all channels for each sample.
        :return: n_samples x 1 numpy array
        """
        pass

    @abc.abstractmethod
    def mean_slice(self, indices: np.ndarray) -> np.ndarray:
        """
        Slices the EMG data at the given sample indices, and then takes the mean across samples.
        :return: n_channels x 1
        """
        pass

    @abc.abstractmethod
    def fast_ica_iteration(
            self, wi: np.ndarray, g: Callable[[Any], Any], sources: np.ndarray, extras: Optional[Any] = None) -> \
            Tuple[np.ndarray, Optional[Any]]:
        """
        Performs a single FastICA iteration. Can return some `extras` which will be provided on the next FastICA
        iteration as a simple way to maintain state through ICA iterations.

        :param wi: initial candidate source vector, to be tuned in this method
        :param g: "g" function for ICA
        :param sources: all other existing sources, n_channels x n_sources
        :param extras: any data returned from previous iterations of ICA

        :return: a new candidate source vector, and anything that should be returned on the next iteration
        """
        pass

    @abc.abstractmethod
    def project(self, sources: np.ndarray) -> np.ndarray:
        """
        Returns (wi.T * data)
        """
        pass

    @abc.abstractmethod
    def gamma(self, wi: np.ndarray) -> np.ndarray:
        """
        Returns (wi.T * data) .^2
        """
        pass


class CpuDataManager(EmgDataManager):
    """
    Implementation of EmgDataManager where all data is managed via numpy arrays on the CPU.
    """

    def __init__(self, whitened_data: np.ndarray):
        self._data = whitened_data

    @property
    def shape(self):
        return self._data.shape

    def squared_sum(self) -> np.ndarray:
        return (self._data ** 2).sum(axis=0)

    def mean_slice(self, indices: np.ndarray) -> np.ndarray:
        return self._data[:, indices].mean(axis=1)

    def fast_ica_iteration(self, wi: np.ndarray, g, sources: np.ndarray, extras: Optional[Any] = None) -> \
            Tuple[np.ndarray, Optional[Any]]:
        if extras is not None:
            sourcesTsources = extras
        else:
            sourcesTsources = np.matmul(sources, sources.T)

        wiTwhitened = np.dot(wi, self._data)
        gwtx, g_wtx = g(wiTwhitened)
        whitenedGwtx = np.multiply(self._data, gwtx)
        wi = whitenedGwtx.mean(axis=1) - g_wtx * wi

        # Orthogonalization
        wi = wi - np.dot(sourcesTsources, wi)
        #  Normalization
        wi = wi / np.linalg.norm(wi, 2)
        return wi, sourcesTsources

    def project(self, sources: np.ndarray) -> np.ndarray:
        return np.dot(sources.T, self._data)

    def gamma(self, wi: np.ndarray) -> np.ndarray:
        return np.dot(wi, self._data) ** 2


class GpuDataManager(EmgDataManager):
    """
    Implementation of EmgDataManager where all data is managed via cupy arrays on the GPU.
    """

    def __init__(self, whitened_data: np.ndarray):
        # Also, explicitly specify float64, since for cupy np.float32 is the default.
        logging.info('Copying whitened data to the GPU...')
        import cupy as xp
        self._xp = xp
        self._data = self._xp.asarray(whitened_data, dtype=np.float64)

    @property
    def shape(self):
        return self._data.shape

    def squared_sum(self) -> np.ndarray:
        return self._xp.asnumpy((self._data ** 2).sum(axis=0))

    def mean_slice(self, indices: np.ndarray) -> np.ndarray:
        return self._xp.asnumpy(self._data[:, indices].mean(axis=1))

    def fast_ica_iteration(self, wi: np.ndarray, g, sources: np.ndarray, extras: Optional[Any] = None) -> Tuple[
        np.ndarray, Optional[Any]]:
        wi = self._xp.asarray(wi)
        if extras is not None:
            sourcesTsources = extras
        else:
            sources_gpu = self._xp.asarray(sources)
            sourcesTsources = self._xp.matmul(sources_gpu, sources_gpu.T)

        wiTwhitened = self._xp.dot(wi, self._data)
        gwtx, g_wtx = g(wiTwhitened)
        whitenedGwtx = self._xp.multiply(self._data, gwtx)
        wi = whitenedGwtx.mean(axis=1) - g_wtx * wi

        # Orthogonalization
        wi = wi - self._xp.dot(sourcesTsources, wi)
        #  Normalization
        wi = wi / self._xp.linalg.norm(wi, 2)
        return self._xp.asnumpy(wi), sourcesTsources

    def project(self, sources: np.ndarray) -> np.ndarray:
        return self._xp.asnumpy(self._xp.dot(self._xp.asarray(sources).T, self._data))

    def gamma(self, wi: np.ndarray) -> np.ndarray:
        return self._xp.asnumpy(self._xp.dot(self._xp.asarray(wi), self._data) ** 2)


class DaskCpuDataManager(EmgDataManager):
    """
    Implementation of EmgDataManager where all data is managed via Dask arrays on the CPU.
    """

    def __init__(self, whitened_data: np.ndarray, dask_chunk_size_samples: int):
        from dask import array as da
        self._da = da

        self._data = da.from_array(whitened_data, chunks=(
            whitened_data.shape[0], dask_chunk_size_samples))

    @property
    def shape(self):
        return self._data.shape

    def squared_sum(self) -> np.ndarray:
        return (self._data ** 2).sum(axis=0).compute()

    def mean_slice(self, indices: Union[int, np.ndarray]) -> np.ndarray:
        return self._data[:, indices].mean(axis=1).compute()

    def fast_ica_iteration(self, wi: np.ndarray, g, sources: np.ndarray, extras: Optional[Any] = None) -> \
            Tuple[np.ndarray, Optional[Any]]:
        if extras is not None:
            sourcesTsources = extras
        else:
            sourcesTsources = self._da.from_array(np.matmul(sources, sources.T)).persist()

        wiTwhitened = self._da.dot(wi, self._data)
        gwtx, g_wtx = g(wiTwhitened)
        whitenedGwtx = self._da.multiply(self._data, gwtx)
        wi = whitenedGwtx.mean(axis=1) - g_wtx * wi

        # Orthogonalization
        wi = wi - self._da.dot(sourcesTsources, wi)
        #  Normalization
        wi = wi / self._da.linalg.norm(wi, 2)
        wi = wi.compute()

        return wi, sourcesTsources

    def project(self, sources: np.ndarray) -> np.ndarray:
        return self._da.dot(sources.T, self._data).compute()

    def gamma(self, wi: np.ndarray) -> np.ndarray:
        gamma_da = self._da.dot(wi, self._data) ** 2
        return gamma_da.compute()


class DaskGpuDataManager(EmgDataManager):
    """
    Implementation of EmgDataManager where all data is managed via Dask arrays primarily on the GPU.
    """

    def __init__(self, whitened_data: np.ndarray, dask_chunk_size_samples: int):
        from dask import array as da
        self._da = da

        import cupy as xp
        self._xp = xp

        whitened_data_da = da.from_array(whitened_data, chunks=(
            whitened_data.shape[0], dask_chunk_size_samples))

        # Also, explicitly specify float64, since for cupy np.float32 is the default.
        logging.info('Copying whitened data to the GPU...')
        self._data = whitened_data_da.map_blocks(self._xp.asarray, np.float64)

    @property
    def shape(self):
        return self._data.shape

    def squared_sum(self) -> np.ndarray:
        return self._xp.asnumpy((self._data ** 2).sum(axis=0).compute())

    def mean_slice(self, indices: np.ndarray) -> np.ndarray:
        return self._xp.asnumpy(self._data[:, indices].mean(axis=1).compute())

    def fast_ica_iteration(self, wi: np.ndarray, g, sources: np.ndarray, extras: Optional[Any] = None) -> \
            Tuple[np.ndarray, Optional[Any]]:
        wi = self._xp.asarray(wi)
        if extras is not None:
            sourcesTsources = extras
        else:
            sources_gpu = self._xp.asarray(sources)
            sourcesTsources = self._xp.matmul(sources_gpu, sources_gpu.T)
            sourcesTsources = self._da.from_array(sourcesTsources).persist()

        wiTwhitened = self._da.dot(wi, self._data)
        gwtx, g_wtx = g(wiTwhitened)
        whitenedGwtx = self._da.multiply(self._data, gwtx)
        wi = whitenedGwtx.mean(axis=1) - g_wtx * wi

        # Orthogonalization
        wi = wi - self._da.dot(sourcesTsources, wi)
        #  Normalization
        wi = wi / self._da.linalg.norm(wi, 2)
        wi = wi.compute()

        return self._xp.asnumpy(wi), sourcesTsources

    def project(self, sources: np.ndarray) -> np.ndarray:
        return self._xp.asnumpy(self._da.dot(self._xp.asarray(sources), self._data).compute())

    def gamma(self, wi: np.ndarray) -> np.ndarray:
        gamma_da = self._da.dot(self._xp.asarray(wi).T, self._data) ** 2
        gamma = gamma_da.compute()
        return self._xp.asnumpy(gamma)
