import collections
import dataclasses
import logging
import pickle
import time
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List, Callable

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks
from scipy.stats import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from emgdecomp._data import EmgDataManager
from emgdecomp.parameters import EmgDecompositionParams
from ._util import minimum_distances, find_disconnected_subgraphs


@dataclass
class Component:
    """
    Holds a decomposition component (vector) that projects zero-mean, extended, non-whitened input data into the
    source space. The raw sources (source) are also included, as well as the mean waveform from the raw data.
    """
    unit_index: int
    vector: np.ndarray
    source: np.ndarray
    waveform: np.ndarray
    threshold: float


@dataclass
class Components:
    """
    Holds decomposition components.
    """
    data: List[Component]

    def get_components(self) -> np.ndarray:
        return np.stack([component.vector for component in self.data])

    def get_sources(self) -> np.ndarray:
        return np.stack([component.source for component in self.data]).T

    def get_thresholds(self) -> np.ndarray:
        return np.array([component.threshold for component in self.data], dtype=np.float64)

    def get_unit_indexes(self) -> np.ndarray:
        return np.array([component.unit_index for component in self.data], dtype=np.int32)

    def get_waveforms(self) -> Dict[int, np.ndarray]:
        return {component.unit_index: component.waveform for component in self.data}

    def __len__(self):
        return len(self.data)


@dataclass
class EmgDecompositionModel(object):
    """
    Holds data relevant to the decomposition model.
    """
    extended_data_mean: np.ndarray
    whitening_matrix: np.ndarray
    components: Components


def compute_percentage_coincident(spike_train_1: np.ndarray, spike_train_2: np.ndarray) -> float:
    # Stable ordering of the spike trains
    if len(spike_train_2) < len(spike_train_1):
        spike_train_1, spike_train_2 = spike_train_2, spike_train_1
    distances = minimum_distances(spike_train_1, spike_train_2)
    mode, _ = stats.mode(distances)
    mode = mode[0]
    counts = np.sum((distances == mode) | (distances == mode + 1) | (distances == mode - 1))
    return counts / max(len(spike_train_1), len(spike_train_2))


def compute_rate_of_agreement(spike_train_1: np.ndarray, spike_train_2: np.ndarray) -> float:
    """ Rate of agreement as (possibly) computed in Barsakcioglu et al. 2020 """
    if len(spike_train_1) > len(spike_train_2):
        spike_train_1, spike_train_2 = spike_train_2, spike_train_1
    distances = minimum_distances(spike_train_1, spike_train_2)
    mode, _ = stats.mode(distances)
    mode = mode[0]
    n_common = np.sum((distances == mode) | (distances == mode + 1) | (distances == mode - 1))
    roa = 100 * n_common / (len(spike_train_1) + len(spike_train_2) - n_common)
    return roa


def get_compute_spike_distance():
    # pip install pyspike fails on python > 3.7 - install from sources as described here:
    # http://mariomulansky.github.io/PySpike/#install-from-github-sources
    import pyspike as spk

    def _compute_spike_distance(spike_train_1: np.ndarray, spike_train_2: np.ndarray) -> float:
        t_start = np.min([np.min(spike_train_1), np.min(spike_train_2)])
        t_stop = np.max([np.max(spike_train_2), np.max(spike_train_2)])
        spike_train_1 = spk.SpikeTrain(spike_train_1, [t_start, t_stop])
        spike_train_2 = spk.SpikeTrain(spike_train_2, [t_start, t_stop])
        return 1 - spk.spike_distance(spike_train_1, spike_train_2)

    return _compute_spike_distance


def find_duplicates(spike_trains_source_indexes: Union[np.ndarray, pd.Series],
                    spike_trains_sample_indexes: Union[np.ndarray, pd.Series],
                    similarity_func: Callable[[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]], float],
                    max_similarity: float = 0.5,
                    keep_first_n_sources: int = 0) -> List[List[int]]:
    unique_sources = np.unique(spike_trains_source_indexes)
    similarity = np.zeros((len(unique_sources), len(unique_sources)), dtype=np.float64)
    for i, source_i in enumerate(unique_sources):
        for j in range(keep_first_n_sources, len(unique_sources)):
            if i <= j:
                # Matrix is symmetric, only compute half
                continue
            spike_train_1 = spike_trains_sample_indexes[spike_trains_source_indexes == source_i]
            spike_train_2 = spike_trains_sample_indexes[spike_trains_source_indexes == unique_sources[j]]
            similarity[i, j] = similarity_func(spike_train_1, spike_train_2)

    # Group sources by high (> params.max_similarity) similarity
    edges = collections.defaultdict(set)
    for x, y in np.argwhere(similarity > max_similarity):
        source_x = unique_sources[x]
        source_y = unique_sources[y]
        edges[source_x].add(source_y)
        edges[source_y].add(source_x)

    # Make sure all unique and/or old sources are counted
    tot_sources = set(list(range(keep_first_n_sources)) + unique_sources.tolist())
    for source in tot_sources:
        if source not in edges:
            edges[source] = set([])
    return find_disconnected_subgraphs(edges)


def remove_duplicates(spike_trains_source_indexes: Union[np.ndarray, pd.Series],
                      spike_trains_sample_indexes: Union[np.ndarray, pd.Series],
                      similarity_func: Callable[[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]], float],
                      max_similarity: float = 0.5,
                      keep_first_n_sources: int = 0):
    subgraphs = find_duplicates(
        spike_trains_source_indexes=spike_trains_source_indexes,
        spike_trains_sample_indexes=spike_trains_sample_indexes,
        similarity_func=similarity_func,
        max_similarity=max_similarity,
        keep_first_n_sources=keep_first_n_sources)
    groups_by_source_idx = {}
    for group_idx, group in enumerate(subgraphs):
        for source_idx in group:
            groups_by_source_idx[source_idx] = group_idx

    num_groups = len(groups_by_source_idx)
    tot_sources = set(groups_by_source_idx.keys())

    # For each group, just keep the source with the highest number of detected spike
    kept_sources_by_group = {}
    for source_idx in tot_sources:
        group = groups_by_source_idx[source_idx]
        if group not in kept_sources_by_group or np.sum(spike_trains_source_indexes == source_idx) > np.sum(
                spike_trains_source_indexes == kept_sources_by_group[group]):
            kept_sources_by_group[group] = source_idx

    source_mapping = {}
    for group, remaining_source_idx in kept_sources_by_group.items():
        for source_idx, group_idx in groups_by_source_idx.items():
            if group_idx == group:
                source_mapping[source_idx] = remaining_source_idx
    for source_idx in tot_sources:
        if source_idx not in source_mapping:
            source_mapping[source_idx] = source_idx

    remaining_source_idxs = list(kept_sources_by_group.values())
    logging.info('{}/{} total groups found [for each source idx: {}]. '
                 'Keeping the following for each group idx: {}'.format(num_groups,
                                                                       len(tot_sources),
                                                                       groups_by_source_idx,
                                                                       kept_sources_by_group))

    return remaining_source_idxs, source_mapping


class EmgDecomposition(object):
    """
    Decomposes surface or intramuscular EMG primarily according to Negro 2016, "Multichannel Blind Source
    Separation...". See EmgDecompositionParams for configurable parameters for this metadata.
    """

    params: EmgDecompositionParams
    _model: Optional[EmgDecompositionModel]

    def __init__(
            self,
            params: EmgDecompositionParams,
            verbose: bool = True,
            use_cuda: bool = False,
            use_dask: bool = False
    ):
        """
        :param params: see documentation on EmgDecompositionParams.
        :param verbose: controls verbosity of the logs.
        :param use_cuda: whether CUDA (i.e. GPU computation) should be used for decomposition. Your GPU should be able
        to fit the *extended* data, i.e. if you have n_channels x n_samples in your EMG data, the extended data will be
        n_channels x extension_factor x n_samples of float64's.
        :param use_dask: whether Dask should be used for decomposition. For smaller arrays, Dask is slower than raw
        NumPy or CuPy, but its advantage is that it can be used for larger-than-memory arrays (i.e., when data needs to
        span multiple workers). See Dask documentation on how to set up a cluster.
        """
        self.params = params
        self._verbose = verbose
        self._use_cuda = use_cuda
        self._use_dask = use_dask

        if use_dask:
            if use_cuda:
                from emgdecomp._data import DaskGpuDataManager
                self._manager_factory = lambda data: DaskGpuDataManager(data, params.dask_chunk_size_samples)
                from dask import array as da
                import cupy
                g_da = da
                g_xp = cupy
                logging.info('CUDA and dask enabled.')
            else:
                from emgdecomp._data import DaskCpuDataManager
                self._manager_factory = lambda data: DaskCpuDataManager(data, params.dask_chunk_size_samples)
                from dask import array as da
                g_da = da
                g_xp = np
                logging.info('Dask enabled.')
        else:
            if use_cuda:
                from emgdecomp._data import GpuDataManager
                self._manager_factory = GpuDataManager
                import cupy
                g_da = cupy
                g_xp = cupy
                logging.info('CUDA enabled.')
            else:
                from emgdecomp._data import CpuDataManager
                self._manager_factory = CpuDataManager
                g_da = np
                g_xp = np

        if self.params.contrast_function == 'logcosh':
            g = _logcosh
        elif self.params.contrast_function == 'exp':
            g = _exp
        elif self.params.contrast_function == 'cube':
            g = _cube
        elif self.params.contrast_function == 'square':
            g = _square
        else:
            raise ValueError(
                f'Invalid contrast function {params.contrast_function}; expected one of logcosh, exp, cube,'
                f' or square.')
        self._g = lambda x: g(g_da, g_xp, x)

        assert self.params.clustering_algorithm == 'kmeans' or self.params.clustering_algorithm == 'ward'

        if self.params.firings_similarity_metric == 'perc_coincident':
            self.compute_firings_similarity = compute_percentage_coincident
        elif self.params.firings_similarity_metric == 'spike_distance':
            if self._use_cuda:
                raise ValueError('"spike_distance" firings_similarity_metric not supported with CUDA')
            self.compute_firings_similarity = get_compute_spike_distance()
        else:
            raise ValueError(
                f'Invalid firings similarity metric {params.firings_similarity_metric}; expected one of perc_coincident'
                f' or spike_distance.')

        self._raw_sources: Optional[np.array] = None  # in Negro 2016 this is the B matrix
        self._model = None

    def clear(self):
        # self._raw_sources = None
        self._model = None
        if self._use_cuda:
            logging.info('Clearing gpu memory.')
            import cupy as xp
            mem_pool = xp.get_default_memory_pool()
            mem_pool.free_all_blocks()

    @property
    def model(self) -> Optional[EmgDecompositionModel]:
        return self._model

    @model.setter
    def model(self, model: Optional[EmgDecompositionModel]):
        self._model = model
        if model is not None:
            self._raw_sources = model.components.get_sources()
        else:
            self._raw_sources = None

    def decompose(self, data: np.ndarray) -> np.ndarray:
        """
        :param data: n_channels x n_samples array, or n_channels x n_samples x n_chunks if your data is chunked (e.g.
         if you're just decomposing threshold crossings or small snippets of data).
        This data should already be bandpass filtered to remove noise and downsampled sufficiently to decrease
        computational load. Negro 2016 utilized bandpass filter of 100-4400 Hz with sampling rate of 10 kHz for
        intramuscular, and bandpass filter of 10-900 Hz and downsampled to 2 kHz for surface EMG. Formento 2021 used
        the latter as well. Data are casted here to np.float64. This class will take care of further preprocessing
        (e.g. whitening, de-meaning, etc.).

        :return: structured array with columns 'source_idx', 'discharge_samples', and 'discharge_seconds', where
        'discharge_samples' is the discharge timing in samples, and 'discharge_seconds' timing in seconds. Other
        relevant parameters can be found in the `model` property.
        """

        # 1) Data preprocessing: extend, subtract mean, whiten
        whitened_data = self._data_preprocessing(data)

        # 2) Define 'good' indices to initialize new sources (wi)
        # The scipy find peaks algorithm doesn't work on cp.arrays. An option would be to use scupy.
        np_power_whitened_data = whitened_data.squared_sum()
        max_num_sources = min(self.params.maximum_num_sources, np_power_whitened_data.shape[0])
        wi_init_indices = self._compute_init_indices(np_power_whitened_data, max_num_sources)

        if len(wi_init_indices) == 0:
            logging.warning('Cannot initialize sources as no peak was present in the provided data; aborting.')
            return np.empty((0,), dtype=self._firings_dtype())

        # 3) Do decomposition similar to Negro 2016
        self._decompose(whitened_data=whitened_data,
                        power_whitened_data=np_power_whitened_data,
                        wi_init_indices=wi_init_indices,
                        starting_index=0,
                        max_num_sources=max_num_sources)

        # 4) Postprocessing: (i) remove duplicated sources, (ii) sources capturing artifacts (rarely active),
        # (iii) get detected peaks (the timings of the detected MUAs), and the thresholds used to separate a source from
        # baseline activity.
        return self._do_post_processing(whitened_data, original_data=data, old_thresholds=None, old_waveforms=None)

    def decompose_batch(self, data: np.ndarray):
        """
        Given a previous decomposition parameters, this searches for new sources in a batch of emg data. Existing
        sources, as given by the `model` property, will not be touched; only new sources will be added if any are found.
        The returned firings will contain firings belonging to any existing sources and the newly detected sources.

        :param data: n_channels x n_samples array, or n_channels x n_samples x n_chunks if your data is chunked (e.g.
         if you're just decomposing threshold crossings or small snippets of data).
        This data should already be bandpass filtered to remove noise and downsampled sufficiently to decrease
        computational load. Negro 2016 utilized bandpass filter of 100-4400 Hz with sampling rate of 10 kHz for
        intramuscular, and bandpass filter of 10-900 Hz and downsampled to 2 kHz for surface EMG. Formento 2021 used
        the latter as well. Data are casted here to np.float64. This class will take care of further preprocessing
        (e.g. whitening, de-meaning, etc.).

        :return: structured array with columns 'source_idx', 'discharge_samples', and 'discharge_seconds', where
        'discharge_samples' is the discharge timing in samples, and 'discharge_seconds' timing in seconds.
        """
        self._check_decomposed()
        old_thresholds = self._model.components.get_thresholds()
        old_waveforms = self._model.components.get_waveforms()

        # 1) Data preprocessing: extend, subtract mean, whiten
        whitened_data = self._data_preprocessing(data)

        power_whitened_data = whitened_data.squared_sum()

        # 2) Define 'good' indices to initialize new sources (wi)
        max_num_sources = min(self.params.maximum_num_sources, power_whitened_data.shape[0])

        # The scipy find peaks algorithm doesn't work on cp.arrays. An option would be to use scupy.
        wi_init_indices = self._compute_init_indices(power_whitened_data, max_num_sources)

        # 3) Do decomposition similar to Negro 2016
        self._decompose(whitened_data=whitened_data,
                        power_whitened_data=power_whitened_data,
                        wi_init_indices=wi_init_indices,
                        starting_index=self._model.components.get_sources().shape[1],
                        max_num_sources=max_num_sources)

        # 4) Postprocessing: (i) remove duplicated sources, (ii) sources capturing artifacts (rarely active),
        # (iii) get detected peaks (the timings of the detected MUAs), and the thresholds used to separate a source from
        # baseline activity.
        return self._do_post_processing(
            whitened_data,
            original_data=data,
            old_thresholds=old_thresholds,
            old_waveforms=old_waveforms)

    def _data_preprocessing(self, data: np.ndarray) -> EmgDataManager:
        # Extend the data, computes the whitening matrix, and then creates the whitened data on the GPU or CPU as
        # specified. Note that this method implicitly assumes that the entire extended/whitened data can fit in
        # *at least* CPU memory. If using GPU, the entire extended dataset must also fit in GPU memory, though this
        # supports dask and so out-of-core memory can be used for the GPU.

        # Shape checking
        if len(data.shape) == 2:
            num_channels, num_samples = data.shape
            n_blocks = 1
        elif len(data.shape) == 3:
            num_channels, num_samples, n_blocks = data.shape
        else:
            raise ValueError(f'Data must be either 2 or 3 dimensional; provided data is {len(data.shape)} dimensional.')
        if num_channels > num_samples * n_blocks:
            raise ValueError(f'Fewer channels {num_channels} than samples {num_samples} provided. '
                             f'Did you forget to transpose?')

        # Create the "extended" version of the data
        # extended_data = [xi(k), xi(k-1), ..., xi(k-extension_factor)], i = 1, n_channels
        extended_data = self._extend_data(data, num_channels, num_samples)

        # Subtract mean from data
        if self._model is None:
            extended_data_mean = extended_data.mean(keepdims=True, axis=1)
        else:
            extended_data_mean = self._model.extended_data_mean
        normalized_data = extended_data - extended_data_mean

        if self._model is None:
            # Whiten data with regularization factor of the average of the smallest half of the evals
            logging.info('Computing covariance matrix...')
            Rxx = np.cov(normalized_data)
            logging.info('Covariance computed. Calculating whitening matrix...')

            if self.params.regularization_method == 'truncate':
                U, S_vec, Vh = np.linalg.svd(Rxx)
                num_noise_evals = int(len(S_vec) * self.params.regularization_factor_eigvals)
                eig_threshold = (S_vec[len(S_vec) - num_noise_evals:]).mean()
                indices = (S_vec > eig_threshold)
                whitening_matrix = np.matmul(np.matmul(U[:, indices],
                                                       np.diag(np.divide(1, np.sqrt(S_vec[indices])))),
                                             Vh[indices, :])
            elif self.params.regularization_method == 'add':
                Ds, U = np.linalg.eigh(Rxx)
                eig_threshold = (Ds[:int(len(Ds) * self.params.regularization_factor_eigvals)]).mean()
                whitening_matrix = np.matmul(
                    np.matmul(U, np.diag(np.sqrt(np.divide(1, Ds + eig_threshold)))), U.T)
            else:
                raise ValueError('Invalid regularization method {}; expected one of truncate,add'.format(
                    self.params.regularization_method))
        else:
            whitening_matrix = self._model.whitening_matrix

        if self._model is None:
            self._model = EmgDecompositionModel(
                extended_data_mean=extended_data_mean,
                whitening_matrix=whitening_matrix,
                components=Components(data=[])
            )

        whitened_data = np.matmul(whitening_matrix, normalized_data)

        ret = self._manager_factory(whitened_data)
        logging.info(f'Whitened data created. Array: {ret}')
        return ret

    def _extend_data(self, data: np.ndarray, num_channels: int, num_samples: int) -> np.ndarray:
        """
        Create the extended version of the data, only on the CPU.
        """
        extended_data = None
        if len(data.shape) == 3:
            n_blocks = data.shape[2]
            n_samples_block = num_samples - self.params.extension_factor + 1
            extended_data = np.zeros((self.params.extension_factor * num_channels, n_samples_block * n_blocks),
                                     dtype=np.float64)
            for i in range(n_blocks):
                extended_data[:, i * n_samples_block:(i + 1) * n_samples_block] = self._extend_data(data[:, :, i],
                                                                                                    num_channels,
                                                                                                    num_samples)
        elif len(data.shape) == 2:
            extended_data = np.zeros((self.params.extension_factor * num_channels,
                                      num_samples - self.params.extension_factor + 1), dtype=np.float64)
            for extension_idx in range(self.params.extension_factor):
                extended_data[extension_idx::self.params.extension_factor, :] = data[:, self.params.extension_factor
                                                                                        - extension_idx - 1:
                                                                                        num_samples - extension_idx]

        return extended_data

    def _compute_init_indices(self, power_whitened_data: np.ndarray, max_num_sources: int) -> np.ndarray:
        # 10ms on either side for finding these peaks
        power_whitened_data_peaks, _ = find_peaks(power_whitened_data,
                                                  distance=int(round(10e-3 * self.params.sampling_rate)))
        peak_heights = power_whitened_data[power_whitened_data_peaks]
        sorted_peak_indices = np.argsort(peak_heights)[::-1]
        # Find peaks in the whitened data (whitened_data) to use as initialization points for the fixed-point algorithm
        max_wi_indices = power_whitened_data_peaks[sorted_peak_indices]

        # We will initialize according to a random peak in the top 25% (parameter that can be tuned, but shouldn't
        # really affect that much). Bound this by some factor of the number of iterations we'll do.
        top_max_wi_indices = len(max_wi_indices) // 4
        if top_max_wi_indices < 4 * max_num_sources:
            top_max_wi_indices = 4 * max_num_sources
        return max_wi_indices[:top_max_wi_indices]

    def _decompose(self,
                   whitened_data: EmgDataManager,
                   power_whitened_data: np.ndarray,
                   wi_init_indices: np.ndarray,
                   starting_index: int,
                   max_num_sources: int):
        t_start = time.time_ns()

        if self._raw_sources is None:
            self._raw_sources = np.zeros((whitened_data.shape[0], 0), dtype=np.float64)

        for source_idx in range(starting_index, max_num_sources):
            logging.info('===== SOURCE INDEX {}'.format(source_idx))
            self._compute_next_source(whitened_data,
                                      power_whitened_data,
                                      wi_init_indices,
                                      source_idx,
                                      max_num_sources)
        elapsed = time.time_ns() - t_start
        logging.info(f'==================================')
        logging.info(f'Source estimation and improvements completed: {self._raw_sources.shape[1]} sources '
                     f'detected in {elapsed / 1e9} seconds.')

    def _compute_next_source(self,
                             whitened_data: EmgDataManager,
                             power_whitened_data: np.ndarray,
                             wi_init_indices: np.ndarray,
                             source_idx: int,
                             max_num_sources: int):
        if self.params.w_init_indices is not None and source_idx < len(self.params.w_init_indices):
            logging.info('Initialization done using provided index')
            wi = whitened_data.mean_slice(np.array([self.params.w_init_indices[source_idx]]))
        elif source_idx > self.params.fraction_peaks_initialization * max_num_sources:
            logging.info('Initialization done randomly')
            wi = np.random.normal(loc=0.0, scale=1.0, size=(whitened_data.shape[0],))
        else:
            wi_index = np.random.choice(wi_init_indices, size=1)[0]
            logging.info(
                'Initialization done using data index {}, value {}'.format(wi_index, power_whitened_data[wi_index]))
            wi = whitened_data.mean_slice(np.array([wi_index]))

        sources = self._raw_sources
        wi = wi - np.dot(np.matmul(sources, sources.T), wi)
        wi = wi / np.linalg.norm(wi, 2)

        # 1) Do ICA via fixed-point iteration, orthogonalization, and normalization.
        converged, wi = self._fast_ica_iterations(wi, whitened_data, sources)
        if not converged:
            logging.info('Reinitializing; did not reach convergence in ICA. Perhaps reduce the number of sources '
                         'that we\'re looking for?')
            return
        # wi is now an estimate of the source process up to a scale factor and a delay.

        # 2) Improvement and detection
        gamma = whitened_data.gamma(wi)
        peak_indices, prev_obj = self._improvement_iteration_inner(gamma)

        if prev_obj is None:
            logging.info('Improvement iteration could not be done; objective unable to be assessed.')
        else:
            # Do the "improvement iteration", which will update `wi` and `peak_indices` correspondingly until
            # a user-defined metric (e.g. SIL or CoV ISI) converges.
            iter_idx = 0
            sources = self._raw_sources
            while iter_idx < 100:
                new_wi = whitened_data.mean_slice(peak_indices)

                # Orthogonalization and normalization
                new_wi = new_wi - np.dot(np.matmul(sources, sources.T), new_wi)
                new_wi = new_wi / np.linalg.norm(new_wi, 2)

                new_gamma = whitened_data.gamma(new_wi)
                new_peak_indices, obj = self._improvement_iteration_inner(new_gamma)
                if obj is None:
                    logging.info('Improvement iteration {} failed to assess objective. Breaking.'.format(iter_idx))
                    break
                if np.abs(obj - prev_obj) < self.params.convergence_tolerance:
                    wi = new_wi
                    logging.info('Improvement iteration converged after {} iterations; final diff {}'.format(
                        iter_idx + 1,
                        np.abs(obj - prev_obj)))
                    break

                if obj > prev_obj:
                    # We went the wrong way! Break and don't use this new_wi
                    logging.info('Improvement iteration {} increased the objective from {} to {}; breaking'.format(
                        iter_idx, prev_obj, obj))
                    break
                # We went the right way: new_wi is good.
                if self._verbose:
                    logging.info('Improvement iteration {}: obj {} => {} [{} peaks]'.format(iter_idx, prev_obj, obj,
                                                                                            len(new_peak_indices)))
                wi = new_wi
                peak_indices = new_peak_indices
                prev_obj = obj
                iter_idx += 1

        # Compute SIL on the gamma according to labels of when we think the spikes are
        should_accept_source = self._should_accept_source_sil(whitened_data, wi)
        if should_accept_source:
            # Finally, add the new source to the sources matrix!
            logging.info(f'Adding a column to the sources matrix (before adding '
                         f'sources shape: {self._raw_sources.shape})')
            self._raw_sources = np.concatenate((sources, wi.reshape((-1, 1))), axis=1)
        else:
            logging.info('Source rejected.')

    def _should_accept_source_sil(self, whitened_data: EmgDataManager, wi: np.ndarray) -> bool:
        if self.params.source_acceptance_metric == 'sil':
            gamma = whitened_data.gamma(wi)
            _, sil = self._improvement_iteration_inner(gamma, compute_sil=True)
            sil = abs(sil)
            ret = sil >= self.params.sil_threshold
            if ret:
                logging.info(f'SIL for source above threshold: {sil}')
            else:
                logging.info('SIL below threshold [SIL={}], not adding this vector'.format(sil))
            return ret
        elif self.params.source_acceptance_metric == 'davies_bouldin':
            gamma = whitened_data.gamma(wi)
            _, db_score = self._improvement_iteration_inner(gamma, compute_davies_bouldin=True)
            db_score = abs(db_score)
            ret = db_score < self.params.davies_bouldin_threshold
            if ret:
                logging.info(f'DB for source below threshold: {db_score}')
            else:
                logging.info('DB above threshold [DB={}], not adding this vector'.format(db_score))
            return ret

        else:
            return True

    def _fast_ica_iterations(self, wi: np.ndarray, whitened_data: EmgDataManager, sources):
        iter_idx = 0
        converged = True

        prev_wi = np.zeros_like(wi)

        extras = None
        while 1 - abs(float(np.dot(wi, prev_wi))) > self.params.convergence_tolerance:
            prev_wi = wi

            wi, extras = whitened_data.fast_ica_iteration(wi, self._g, sources, extras)
            iter_idx += 1

            if self._verbose:
                logging.info('ICA iteration {}: {}'.format(iter_idx, 1 - abs(float(np.dot(wi, prev_wi)))))
            if iter_idx > self.params.max_iter:
                converged = False
                break

        if converged:
            logging.info('ICA iteration completed: # iterations {}, resulting obj {}'.format(
                iter_idx, 1 - abs(float(np.dot(wi, prev_wi)))))

        return converged, wi

    def _find_peaks(self, gamma: np.ndarray,
                    threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                Optional[np.ndarray], float]:
        """
        Find peaks in the estimated sources.
        When threshold is None (default), after finding the peaks, kmeans is run to classify between an active source
        and baseline activity. If threshold is provided, classification is performed by simply looking at peaks above
        the provided value.
        """
        peak_distance = None
        if self.params.min_peaks_distance_ms is not None:
            # Some nominal distance for frequency normalization
            peak_distance = int(round((self.params.min_peaks_distance_ms / 1000.0) * self.params.sampling_rate))
            peak_distance = max(1, peak_distance)

        peak_indices, _ = find_peaks(gamma, distance=peak_distance,
                                     height=self.params.improvement_iteration_min_peak_heights)
        gamma_peaks = gamma[peak_indices].reshape((-1, 1))

        if threshold is None:
            if self.params.clustering_algorithm == 'kmeans':
                # Run K-Means++ to separate out "signal" peaks from "noise" peaks
                clusters, labels = kmeans2(gamma_peaks, 2, minit='++')
            elif self.params.clustering_algorithm == 'ward':
                # Run Ward clustering to separate out "signal" peaks from "noise" peaks
                ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(gamma_peaks)
                clusters = [np.mean(gamma_peaks[ward.labels_ == 0]), np.mean(gamma_peaks[ward.labels_ == 1])]
                labels = ward.labels_
            else:
                raise ValueError(f'Expecting either "kmeans" or "ward", got "{self.params.clustering_algorithm}".')

            high_cluster_idx = 0 if clusters[0] > clusters[1] else 1
            ps = peak_indices[labels == high_cluster_idx]
            threshold = float(np.mean(clusters))
            if np.size(ps) > 1:
                pt_indices_high = np.squeeze(ps)
            else:
                pt_indices_high = ps
        else:
            pt_indices_high = peak_indices[np.squeeze(gamma_peaks) > threshold]
            labels = None

        return peak_indices, gamma_peaks, labels, pt_indices_high, threshold

    def _improvement_iteration_inner(
            self,
            np_gamma: np.ndarray,
            compute_sil: bool = False,
            compute_davies_bouldin: bool = False) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        The inner loop of the improvement iterations. Returns an array of indices corresponding to detected/clustered
        peaks in the estimated source along with a objective score according to a user-defined metric. If the objective
        is None, then the metric could not be computed.

        Note that we assume we want the metric to decrease, so some metrics might need a negative sign (e.g. SIL).

        :param gamma: the square of the estimated source
        :param compute_sil: whether SIL should be the computed metric regardless of the user's preference.
        :param return_threshold: whether to return the kmeans threshold used to classify a source from baseline
        activity. This output argument is given inplace of the computed metric.
        :return: the peak indices and the computed objective function value
        """

        # NB: the scipy find peaks algorithm doesn't work on cp.arrays. An option would be to use scupy.
        if self.params.improvement_iteration_metric in ('sil', 'isi') or compute_sil or compute_davies_bouldin:
            peak_indices, gamma_peaks, labels, pt_indices_high, _ = self._find_peaks(np_gamma)
            if self.params.improvement_iteration_metric == 'sil' or compute_sil:
                if 0 < self.params.sil_max_samples < labels.shape[0]:
                    sil = 1.0
                    for _ in range(10):
                        indices = np.random.permutation(labels.shape[0])[:self.params.sil_max_samples]
                        gamma_peaks_subset = gamma_peaks[indices, :]
                        labels_subset = labels[indices]
                        if len(np.unique(labels_subset)) < 2:
                            continue
                        sil = silhouette_score(gamma_peaks_subset, labels_subset, metric='euclidean')
                        break
                else:
                    sil = silhouette_score(gamma_peaks, labels, metric='euclidean')

                if self._verbose:
                    logging.info(
                        '{}/{} peaks considered high [SIL={}]'.format(len(pt_indices_high), len(peak_indices), sil))
                # Negative SIL since we want the metric to decrease
                return pt_indices_high, -sil
            elif compute_davies_bouldin:
                db_score = davies_bouldin_score(gamma_peaks, labels)

                if self._verbose:
                    logging.info(
                        '{}/{} peaks considered high [DB score={}]'.format(len(pt_indices_high), len(peak_indices), db_score))
                return pt_indices_high, db_score
            else:
                if len(pt_indices_high) == 1:
                    logging.info('Only single peak detected; cannot iterate on ISI computation. Breaking out.')
                    return pt_indices_high, None

                isi = np.diff(peak_indices)
                cov = np.std(isi) / np.mean(isi)
                if self._verbose:
                    logging.info(
                        '{}/{} peaks considered high [CoV ISI={}]'.format(len(pt_indices_high), len(peak_indices), cov))
                return pt_indices_high, cov
        elif self.params.improvement_iteration_metric in ('csm1', 'csm3'):
            peak_distance = None
            if self.params.min_peaks_distance_ms is not None:
                # Some nominal distance for frequency normalization
                peak_distance = int(round((self.params.min_peaks_distance_ms / 1000.0) * self.params.sampling_rate))
                peak_distance = max(1, peak_distance)

            best_peaks = None
            best_metric = 1e9

            thresh = np.max(np_gamma)
            num_iter = 100
            for t in np.logspace(np.log10(thresh), 0, num_iter):
                indices, _ = find_peaks(np_gamma, height=t, distance=peak_distance)
                if len(indices) <= 1:
                    continue
                candidate_peaks = np.zeros(np_gamma.shape, dtype=np.bool)
                candidate_peaks[indices] = True
                isi = np.diff(indices)
                cov = np.std(indices) / np.mean(indices)
                csm1 = 100 * cov + np.abs((np.max(indices) - np.min(indices)) / (np.median(isi)) - len(indices))

                pnr = 10.0 * np.log10(np.mean(np_gamma[candidate_peaks]) / np.mean(np_gamma[~candidate_peaks]))
                csm3 = csm1 + (100 - pnr)
                if self.params.improvement_iteration_metric == 'csm1':
                    metric = csm1
                else:
                    metric = csm3
                if metric < best_metric:
                    best_metric = metric
                    best_peaks = indices
            if best_peaks is None:
                return None, None
            return best_peaks, best_metric
        else:
            raise ValueError(
                'Invalid improvement iteration metric! Got {}'.format(self.params.improvement_iteration_metric))

    def _do_post_processing(self,
                            whitened_data: EmgDataManager,
                            original_data: np.ndarray,
                            old_thresholds: Optional[np.ndarray] = None,
                            old_waveforms: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        """
        Some artifacts can be detected as a source. To minimize the number of 'artifact' sources we extract here we
        remove all sources that are inactive, i.e. those that only fire a small number of times (as given by
        min_n_peaks in parameters).

        The convolutional algorithm utilized here can only detect sources up to a delay, hence it will likely extract
        multiple, delayed versions of the same sources. We thus group sources by their similarity to one another
        according to the given similarity function, and only keep one out of a group of highly correlated sources.

        :return: the firings
        """

        sources = self._raw_sources
        timings, thresholds = self._detect_spikes(whitened_data, sources, thresholds=old_thresholds)

        # Remove units that detect less than x spikes - we consider them as artifact sources
        n_sources = np.max(timings['source_idx']) + 1
        keep_first_n_sources = 0 if old_thresholds is None else len(old_thresholds)
        n_events = {}
        for source in range(n_sources):
            if source < keep_first_n_sources:
                continue
            n_events[source] = sum(timings['source_idx'] == source)
            if n_events[source] < self.params.min_n_peaks:
                logging.info(f'Source {source} was detected only {n_events[source]} times; removing it.')
                timings = timings[np.flatnonzero(timings['source_idx'] != source)]

        # Remove duplicates
        remaining_source_idxs, _ = remove_duplicates(
            spike_trains_source_indexes=timings['source_idx'],
            spike_trains_sample_indexes=timings['discharge_samples'],
            similarity_func=self.compute_firings_similarity,
            max_similarity=self.params.max_similarity,
            keep_first_n_sources=keep_first_n_sources)

        if len(remaining_source_idxs) == 0:
            return np.empty((0,), dtype=self._firings_dtype())

        deduped_sources = self._raw_sources[:, np.array(remaining_source_idxs)]

        timings = timings[np.isin(timings['source_idx'], remaining_source_idxs)]
        peaks = timings.copy()
        for i in range(len(remaining_source_idxs)):
            x = timings['source_idx'] == remaining_source_idxs[i]
            peaks['source_idx'][x] = i
        thresholds = thresholds[remaining_source_idxs]

        # Done! Get the waveforms and store everything in results.
        waveforms = self._muap_waveforms(
            firings=peaks,
            data=original_data,
            num_sources=deduped_sources.shape[1],
            waveform_duration_ms=self.params.waveform_duration_ms,
            pre_spike_waveform_duration_ms=self.params.pre_spike_waveform_duration_ms)
        mean_waveforms = {i: np.mean(waveforms[i], axis=0) for i in waveforms}
        if old_waveforms is not None:
            for source_idx, waveform in old_waveforms.items():
                mean_waveforms[source_idx] = waveform
        self._populate_results(deduped_sources, mean_waveforms, thresholds)
        return peaks

    def _detect_spikes(
            self,
            whitened_data: EmgDataManager,
            sources: np.ndarray,
            thresholds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        The final output of the algorithm: detects discharge timings for each source found in the columns of B.
        ----------
        thresholds: Optional numpy array of dimension lower or equal the number of components. If provided peaks in
        the projected data that are above threshold are considered spikes; kmeans is otherwise used. If the provided
        array is shorter than the number of components, thresholding is only performed in the first n components, with
        n the size of the provided array. This is useful in case of batched decompositions where we want to keep the
        previously computed thresholds.
        -------
        """
        detected_spike_trains = []
        thresholds_out = []
        for sidx in range(sources.shape[1]):
            estimated = whitened_data.project(sources[:, sidx])
            threshold = thresholds[sidx] if thresholds is not None and len(thresholds) > sidx else None
            est_square = np.power(estimated, 2)
            _, _, _, _spike_indices, _threshold = self._find_peaks(est_square, threshold)
            thresholds_out.append(_threshold)
            for spike in _spike_indices:
                detected_spike_trains.append((sidx, spike, spike / self.params.sampling_rate, est_square[spike]))
        detected_spike_trains = np.array(detected_spike_trains, dtype=self._firings_dtype())
        return detected_spike_trains, np.array(thresholds_out)

    def _firings_dtype(self):
        return np.dtype([
            ('source_idx', np.int),
            ('discharge_samples', np.int),
            ('discharge_seconds', np.float),
            ('squared_amplitude', np.float),
        ])

    def _populate_results(self, sources: np.ndarray, mean_waveforms: Dict[int, np.ndarray], thresholds: np.ndarray):
        # Use the property setter to ensure raw_sources stays in sync with the de-duped sources
        self.model = dataclasses.replace(
            self._model,
            components=Components([
                Component(
                    unit_index=i,
                    vector=np.dot(sources[:, i], self._model.whitening_matrix),
                    source=sources[:, i],
                    waveform=mean_waveforms[i],
                    threshold=float(thresholds[i]))
                for i in range(sources.shape[1])
            ]))

    def save(self, io):
        """
        Saves the various parameters to a file given by `io` necessary to reconstruct this EmgDecomposition.
        """
        pickle.dump({
            'params': self.params,
            'model': self._model,
            'verbose': self._verbose,
            'use_dask': self._use_dask,
            'use_cuda': self._use_cuda,
        }, io)

    @staticmethod
    def load(io) -> 'EmgDecomposition':
        """
        Loads and creates an instance of EmgDecomposition that was created using #save().
        """
        obj = pickle.load(io)
        ret = EmgDecomposition(
            params=obj['params'],
            verbose=obj['verbose'],
            use_cuda=obj['use_cuda'],
            use_dask=obj['use_dask'],
        )
        ret.model = obj['model']
        return ret

    def transform(self, data: np.ndarray):
        """
        Detects firings in the given data using a model that was already fit on this data. There must be a `model` set
        on this EmgDecomposition; assign a new model via the `model` setter or use #load().
        """
        self._check_decomposed()

        whitened_data = self._data_preprocessing(data)
        thresholds = self._model.components.get_thresholds()
        waveforms = self._model.components.get_waveforms()
        return self._do_post_processing(
            whitened_data=whitened_data,
            original_data=data,
            old_thresholds=thresholds,
            old_waveforms=waveforms)

    def _muap_waveforms(self,
                        firings: np.ndarray,
                        data: np.ndarray,
                        num_sources: int,
                        waveform_duration_ms: float = 30.0,
                        pre_spike_waveform_duration_ms: Optional[float] = None,
                        align_to_global_maxima: bool = False,
                        return_extra_info: bool = False) -> \
            Union[Dict[int, np.ndarray], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Extract the MUAP waveforms around the detected discharge.

        :param firings: structured numpy array containing the timing of the detected MUAPs.
        :param data: the original data passed to decomp.
        :param waveform_duration_ms: how much of the waveform should be extracted in total.
        :param pre_spike_waveform_duration_ms: how much before the spike a waveform should start (if this is None
        - default - the detected firing time will be at the midpoint of the extracted waveform).
        :param align_to_global_maxima: whether the MUAP waveforms should be aligned to the global maxima across all
        channels within the specified waveform window.
        :param return_extra_info: whether the dictionary per source should return:
        - (1) the waveforms themselves
        - (2) an np.ndarray vector of size n_spikes that contains the 0-indexed channel which was used for
        alignment
        - (3) the EMG data index corresponding to the beginning of the window.

        :return: Dict of np.ndarrays containing the muaps waveforms. Each array is n_discharges x n_channels x
        n_waveform_samples.
        """
        num_blocks, num_samples_block = None, None
        if len(data.shape) == 2:
            num_channels, num_samples = data.shape
        elif len(data.shape) == 3:
            num_channels, num_samples, num_blocks = data.shape
            num_samples_block = num_samples - self.params.extension_factor + 1
        else:
            raise ValueError('Data needs to be either 2- or 3- dimensional.')

        wf_samples = int(round(waveform_duration_ms * 1e-3 * self.params.sampling_rate))
        if pre_spike_waveform_duration_ms is None:
            wf_pre_offset = wf_samples // 2
        else:
            wf_pre_offset = int(round(pre_spike_waveform_duration_ms * 1e-3 * self.params.sampling_rate))

        ret = {}
        for source_idx in range(num_sources):
            if len(data.shape) == 2:
                discharges = firings[firings['source_idx'] == source_idx]['discharge_samples']
                discharges = discharges[discharges >= (wf_pre_offset - self.params.extension_factor)]
                discharges = discharges[discharges <= (num_samples - wf_samples + wf_pre_offset -
                                                       self.params.extension_factor)]

                waveforms = np.zeros((len(discharges), num_channels, wf_samples), dtype=np.float64)
                aligned_channel_indices = np.empty((len(discharges),), dtype=np.int)
                emg_data_indices = np.empty((len(discharges),), dtype=np.int)
                num_discharges = 0
                for discharge_idx, discharge in enumerate(discharges):
                    mask = np.arange(wf_samples) - wf_pre_offset + discharge + self.params.extension_factor
                    snippet = data[:, mask]
                    if align_to_global_maxima:
                        max_channel_index, max_sample_index = np.unravel_index(snippet.argmax(), snippet.shape)
                        mask = mask + max_sample_index - wf_pre_offset
                        if mask[-1] >= data.shape[1]:
                            continue
                        waveforms[discharge_idx, :, :len(mask)] = data[:, mask]
                        aligned_channel_indices[discharge_idx] = max_channel_index
                    else:
                        waveforms[discharge_idx, :, :len(mask)] = snippet
                        aligned_channel_indices[discharge_idx] = -1
                    emg_data_indices[discharge_idx] = mask[0]
                    num_discharges += 1

                if return_extra_info:
                    ret[source_idx] = (waveforms[:num_discharges, :, :],
                                       aligned_channel_indices[:num_discharges],
                                       emg_data_indices[:num_discharges])
                else:
                    ret[source_idx] = waveforms[:num_discharges, :, :]
            else:
                if align_to_global_maxima or return_extra_info:
                    # No good reason that this is unimplemented, just lazy
                    raise ValueError('Unimplemented')
                discharges = firings[firings['source_idx'] == source_idx]['discharge_samples']
                waveforms = np.zeros((len(discharges), num_channels, wf_samples), dtype=np.float64)
                n_inserted_waveforms = 0
                for discharge_idx, discharge in enumerate(discharges):
                    block = discharge // num_samples_block
                    index = discharge % num_samples_block
                    mask = np.arange(wf_samples) - wf_pre_offset + index + self.params.extension_factor
                    mask = mask[mask < num_samples]
                    mask = mask[mask >= 0]
                    if len(mask) > 0.5 * wf_samples:
                        waveforms[discharge_idx, :, :len(mask)] = data[:, mask, block]
                        n_inserted_waveforms += 1
                    elif self._verbose:
                        logging.info(f'Not enough samples to extract waveform in discharge {discharge_idx} '
                                     f'and source {source_idx}.')
                ret[source_idx] = waveforms[:n_inserted_waveforms]
        return ret

    def muap_waveforms(self,
                       data: np.ndarray,
                       firings: np.ndarray,
                       waveform_duration_ms: Optional[float] = None,
                       pre_spike_waveform_duration_ms: Optional[float] = None,
                       align_to_global_maxima: bool = False,
                       return_extra_info: bool = False) -> \
            Union[Dict[int, np.ndarray], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Extract the MUAP waveforms around the detected discharge. Interface of the internal _muap_waveforms.

        :param data: This is an n_channels x n_samples, or n_channels x n_samples x n_blocks, array of EMG data.
        :param waveform_duration_ms: as defined in _muap_waveforms.
        :param pre_spike_waveform_duration_ms: as defined in _muap_waveforms.
        :return: Dict of np.ndarrays containing the muaps waveforms. Each array is n_discharges x n_channels x
        n_waveform_samples.
        """
        if waveform_duration_ms is None:
            waveform_duration_ms = self.params.waveform_duration_ms

        num_sources = self.num_sources()
        return self._muap_waveforms(firings,
                                    data,
                                    num_sources,
                                    waveform_duration_ms=waveform_duration_ms,
                                    pre_spike_waveform_duration_ms=pre_spike_waveform_duration_ms,
                                    align_to_global_maxima=align_to_global_maxima,
                                    return_extra_info=return_extra_info)

    def projected_data(self, data: np.ndarray) -> np.ndarray:
        """
        :return: a <num_sources, num_time_samples> array of the computed sources activity. Projects the given data
        into "source space".
        """
        self._check_decomposed()
        whitened_data = self._data_preprocessing(data)
        return whitened_data.project(self._model.components.get_sources())

    def source_vectors(self) -> np.ndarray:
        """
        :return: a <num_channels x extension_factor, num_sources> array of the source vectors that were learned
        """
        self._check_decomposed()
        return self._model.components.get_sources()

    def num_sources(self) -> int:
        """
        :return: how many sources were identified during the metadata.
        """
        self._check_decomposed()
        return len(self._model.components)

    def _check_decomposed(self):
        if self._model is None:
            raise ValueError('#decompose() has not been called yet!')


# Standard non-linear functions coming from sklearn/metadata/_fastica.py
# log-cosh (with slight modifications)
def _logcosh(da, xp, x):
    # As opposed to scikit-learn here we fix alpha = 1 and we vectorize the derivation
    gx = da.tanh(x, x)  # apply the tanh inplace
    g_x = (1 - gx ** 2).mean(axis=-1)
    return gx, g_x

def _exp(da, xp, x):
    exp = xp.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)

def _cube(da, xp, x):
    return x ** 3, (3 * x ** 2).mean(axis=-1)

# added to match implementation of Negro 2016
def _square(da, xp, x):
    return x ** 2, (2 * x).mean(axis=-1)
