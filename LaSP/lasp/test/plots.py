import numpy as np

import matplotlib.pyplot as plt
from lasp.spikes import plot_raster


def test_plot_raster():

    np.random.seed(123456789)
    max_nspikes = 20
    ngroups = 8
    max_cells_per_group = 4

    spike_trials = list()
    groups = dict()
    for k in range(ngroups):
        current_ntrials = len(spike_trials)
        ncells = np.random.randint(1, max_cells_per_group+1)
        nspikes = max_nspikes - k*2
        for n in range(ncells):
            spikes = np.random.rand(nspikes)
            spike_trials.append(spikes)
        groups['G%d' % k] = range(current_ntrials, current_ntrials+ncells)
    print groups

    plt.figure()
    plot_raster(spike_trials, duration=1.0, ylabel='', groups=groups)
    plt.show()


if __name__ == '__main__':
    test_plot_raster()

