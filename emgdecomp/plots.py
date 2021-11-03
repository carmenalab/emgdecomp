import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from emgdecomp.decomposition import EmgDecomposition


def plot_muaps(
        decomp: EmgDecomposition,
        data: np.ndarray,
        firings: np.ndarray,
        waveform_duration_ms: float = 40.,
        n_rows: int = 7,
        n_cols: int = 8,
        fig_size=1.5,
        ylim: float = -.3e-3,
        only_average: bool = True):
    muaps = decomp.muap_waveforms(data=data, firings=firings, waveform_duration_ms=waveform_duration_ms)
    is_first = True
    for key in muaps:
        muap = muaps[key].reshape((muaps[key].shape[0], n_rows, n_cols, muaps[key].shape[-1]))
        if is_first or not only_average:
            fig = plt.figure(figsize=(muap.shape[1] * fig_size, muap.shape[2] * fig_size))
            gs = gridspec.GridSpec(muap.shape[1], muap.shape[2], left=0.05, right=.99, bottom=0.05, top=0.99,
                                   wspace=.01, hspace=.01, figure=fig)
            axs = {}
        for i in range(n_rows):
            for j in range(n_cols):

                if is_first or not only_average:
                    axs[(i, j)] = plt.subplot(gs[i, j])

                if not only_average:
                    axs[(i, j)].plot(muap[:, i, j].T, alpha=0.25, color='k')

                axs[(i, j)].plot(np.mean(muap[:, i, j], axis=0), linewidth=2)
                axs[(i, j)].set_ylim([-ylim, ylim])
                axs[(i, j)].set_yticks([])
                axs[(i, j)].set_xticks([])
                axs[(i, j)].text(0.5, 0.05, f"ch {1 + i * n_cols + j}", transform=axs[(i, j)].transAxes,
                                 ha='center')

        is_first = False


def plot_firings(decomp: EmgDecomposition, data, firings, fig_width=10, fig_height=4):
    """
    Plots the detected firings on top of the projected data.
    By default it plots the internal data, but firings and projected data can also be provided for instance to plot
    the results of .transform()
    """
    n_sources = len(decomp.model.components)
    projected = decomp.projected_data(data)
    gamma = np.power(projected, 2)
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    time_s = np.arange(gamma.shape[1]) / decomp.params.sampling_rate
    for i in range(gamma.shape[0]):
        ax.plot(time_s, i + gamma[i, :] / np.max(gamma[i, :] * 2), 'k', alpha=0.4)
    ax.eventplot([firings['discharge_seconds'][firings['source_idx'] == unit] for unit in range(n_sources)])
