import numpy as np
import matplotlib.pyplot as plt
from integrated_channel_and_test import channel_rss
from test import Node, node_to_beam_idx, beam_gain_map
# Main function of test.py is too big to integrate. In simulation function, when method=1, HBA algorithm is used


def simulation(n, k, angle, method):
    # Terminal and converge time is used to describe the performance of different algorithm
    if method == 1:
        # For HBA method, n=k=2^m.
        measured_beam = []
        measured_rss = []
        regret = []
        best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int).T
        best_array_shift = 1 / np.sqrt(n) * \
            np.exp(1j * np.pi * (2 * best_beam_index / k - 1) * np.arange(n).reshape(-1, 1))
        optimal_rss = channel_rss(best_array_shift, angle)
        terminal_time = 1000
        head_node = Node(0)
        for t in range(terminal_time+1):
            # if break
            converge_time = t
    else:
        measured_beam = []
        measured_rss = []
        regret = []
        terminal_time = 1000
        converge_time = 1000
    return converge_time, terminal_time, measured_beam, measured_rss, regret


def main():
    pass


if __name__ == '__main__':
    main()
