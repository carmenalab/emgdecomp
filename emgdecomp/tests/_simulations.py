from typing import Tuple

import numpy as np


def simulate_emg(n_units: int, tot_time_s: float, firing_rate: float, sampling_rate: float, n_channels: int):
    simulated_signal = None
    spike_indices = []
    for unit in range(n_units):
        _simulated_signal, _spike_indices = poisson_neuron_recording(tot_time_s, firing_rate, sampling_rate, n_channels)
        spike_indices.append(_spike_indices)
        if simulated_signal is None:
            simulated_signal = _simulated_signal
        else:
            simulated_signal += _simulated_signal
    return simulated_signal, spike_indices


def poisson_neuron_recording(tot_time_s: float,
                             firing_rate: float,
                             sampling_rate: float,
                             n_channels: int,
                             perc_ch_per_unit: float = 0.5,
                             noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    spikes_period_ms = 1000 / firing_rate
    n_samples = int(tot_time_s * sampling_rate)
    spikes = np.zeros(n_samples)
    muap_frequency = np.random.normal(100, 10)  # in Hz
    amplitude = np.random.normal(1, .1)
    spike_template = amplitude * np.sin(2 * np.pi * muap_frequency * np.arange(0, 1. / muap_frequency, 1. /
                                                                               sampling_rate))
    next_spike = int(np.random.uniform(0, spikes_period_ms / 1e3 * sampling_rate))
    while True:
        next_spike += int(np.random.poisson(spikes_period_ms) / 1e3 * sampling_rate)
        if next_spike >= n_samples:
            break
        spikes[next_spike] = 1

    n_visibile_chs = int(np.random.normal(perc_ch_per_unit * n_channels, perc_ch_per_unit * n_channels * 0.1) + 1)
    simulated_signal_template = np.convolve(spikes, spike_template, mode='same')
    simulated_signal = np.zeros((n_channels, n_samples))
    for ch in np.random.choice(np.arange(n_channels), n_visibile_chs):
        delay_sammples = int(np.random.normal(5, 1) / 1000 * sampling_rate)
        amp_mod = 1 + np.random.normal(0, .2)
        simulated_signal[ch, :] = amp_mod * np.roll(simulated_signal_template, delay_sammples)
        simulated_signal[ch, :] += np.random.normal(0, noise, n_samples)

    return simulated_signal, np.flatnonzero(spikes)
