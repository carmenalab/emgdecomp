from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EmgDecompositionParams:
    """
    Holds parameters relevant to EMG metadata. Separated in a separate
    class in order to ease persistence/loading of past runs.

    Attributes:
        sampling_rate (float): Sampling rate of the data in Hertz.

        extension_factor (int): How many time lags to add to each data point. Negro 2016 recommends an extension
        parameter equal to 1000/m, where m is the number of channels in the recording. Note that this parameter is a
        tradeoff between metadata accuracy and computational load.

        maximum_num_sources (int): the maximum number of sources to find. Due to the fact this metadata can only
        find sources up to a delay less than the extension factor, the total number of sources found can be higher than
        the number of channels.

        min_peaks_distance_ms (Optional[float]): Minimum distance — in milliseconds — used in the find peaks algorithm
        during the improvement iteration. If None all peaks are used (default = 10ms).

        contrast_function (string): as in sklearn.metadata.FastICA, this is the functional form of the G' function
        used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’ or 'square'. Note that
        'square' is rather rarely used but was here included because used in Negro 2016. Thomas 2006 used 'cube'.

        max_iter (int): maximum number of iterations for computing ICA for each source.

        convergence_tolerance (float): Tolerance for determining convergence in both the fixed-point ICA iterations and
        the improvement iterations that optimize for ISI.

        sil_threshold (float): the threshold for the Silhoutte Score (SIL) above which sources are considered "good" and
        added to our results. Negro 2016 uses a threshold of 0.9. Set this to 1.0 or greater to skip SIL calculation.

        regularization_method (str): method of regularization during whitening. Can be:
          - 'truncate', where the evals under a noise threshold are truncated to zero
          - 'add', where the noise threshold is added to each eigenvalue during whitening

        regularization_factor_eigvals (float): fraction of the eigenvalues of the data matrix to be considered as part
        of noise; the eigenvalues are ordered in ascending order and the mean of this fraction of those eigenvalues are
        used as a regularization factor during whitening. Negro 2016 uses a factor of 0.5.

        improvement_iteration_metric (string): one of 'sil', 'isi', 'csm1', 'csm3'. Refers to which metric is computed
        during the "improvement iteration" (the iterations after fastICA). Negro 2016 and others use the coefficient of
        variation of the interspike interval (CoV ISI); their experiments utilize isometric contractions held at a
        particular % force, and thus it's reasonable to expect consistent ISIs in all of their detected units.
        For more variable experiments such as free motion, SIL might be a better option to generically emphasize
        "signal" vs "noise".

        improvement_iteration_min_peak_heights (Optional[float]): min peak heights to be used for the detecting the peaks
        in  the improvement iteration step. According to Negro2016 this should be None - meaning that all peaks are
        considered. However, k-means++ doesn't always do a good job at dividing what is signal vs noise.  Setting a min
        peak heights might help with this (an heuristically found good starting value is 0.9 if needed).

        fraction_peaks_initialization (float): fraction of sources that are normalized used a randomly selected peak
        from the whitened data.

        firings_similarity_metric (string): one of 'perc_coincident', or 'spike_distance'. 'spike_distance' uses the
        SPIKE-distance metric implemented in http://mariomulansky.github.io/PySpike/#spike-distance. Refer to this
        article http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony for a quick overview. This is the
        metric used to compare two different putative sources to see if they're duplicates of one another.

        max_similarity (float): value between 0 and 1, defining how similar (according to the metric above) two spike
        trains need to be to be considered coming from the same source. This affects which sources are kept after the
        decomposition.

        min_n_peaks (int): minimum number of detected peaks that a source needs to have in order to be considered a
         motor unit. This is used to filter noise from actual sources in the clean step of the decomposition algorithm.

        w_init_indices (Optional[list]): list of indices used to initialize the first n sources (n = len(list)). When all
        provided indices have been used, the algorithm falls back to the standard behavior: using peaks until
        'fraction_peaks_initialization' and then randomly initialized values.

        waveform_duration_ms (float): Duration of the waveforms to extract at the end of the decomposition to save the
        mean muap waveform of each source.

        pre_spike_waveform_duration_ms (Optional[float]): If provided is used to determine the offset of the extracted
        muap waveforms wrt the detected spikes. If None, the extracted waveform is centered around the spike.

        clustering_algorithm (str): one of 'kmeans' or 'ward' (default is kmeans as from Negro 2016). This defines the
        clustering algorithm used to detect motor unit action potentials from the projected data. Note that the
        decomposition algorithm uses spike detections to refine the computed components. Thus this parameter has a
        significant effect on decomposition performance. While 'kmeans' was suggested by Negro 2016 and is here the
        default algorithm, this algorithm assumes even clusters (clusters need to have similar variance). This
        assumption is largely unmet in less controlled / more noisy cases. 'Ward' is an agglomerative clustering
        technique that allows for uneven cluster sizes (see
        https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) for a quick reference) and is
        thus more suited for outliers (i.e. spikes) detections. Note that the thresholds for online classification are
        always set to the mean between the noise and spike clusters (i.e. the online
        classification performs 'kmeans' with means computed using the clustering results of the here selected
        algorithm). Also, it's worth considering that kmeans is approximately 2 times faster than ward.

        dask_chunk_size_samples (int): number of samples in each chunk in Dask. This should end up resulting in chunks
        between ~100MB-3GB for best performance in Dask. Ignored if use_dask is False.

        sil_max_samples (int): size of random subset to take when computing SIL (full SIL takes O(n^2) memory /
        computation). Leave negative to sample the entire dataset.
    """

    sampling_rate: float
    extension_factor: int = 16
    maximum_num_sources: int = 25
    min_peaks_distance_ms: Optional[float] = 15.0
    contrast_function: str = 'logcosh'
    max_iter: int = 100
    convergence_tolerance: float = 1e-4
    sil_threshold: float = 0.85
    davies_bouldin_threshold: float = 0.2
    source_acceptance_metric: str = 'sil'
    regularization_factor_eigvals: float = 0.5
    regularization_method: str = 'truncate'
    improvement_iteration_metric: str = 'isi'
    firings_similarity_metric: str = 'perc_coincident'
    max_similarity: float = 0.3
    min_n_peaks: int = 20
    fraction_peaks_initialization: float = 0.75
    w_init_indices: Optional[np.ndarray] = None
    improvement_iteration_min_peak_heights: Optional[float] = None
    waveform_duration_ms: float = 30.0
    pre_spike_waveform_duration_ms: Optional[float] = 10.0
    clustering_algorithm: str = 'kmeans'
    dask_chunk_size_samples: int = 100000
    sil_max_samples: int = -1

