import numpy as np
import matplotlib.pyplot as plt
from integrated_channel_and_test import channel_rss
from test import Node, node_to_beam_idx, beam_gain_map
# Main function of test.py is too big to integrate. In simulation function, when method=1, HBA algorithm is used


def simulation(m, n, k, angle, method):
    # Terminal and converge time is used to describe the performance of different algorithm
    if method == 1:
        # For HBA method, n=k=2^m.
        measured_beam = []
        measured_rss = []
        regret_record = []
        regret = 0
        terminal_time = 1000
        converge_time = 0
        p = []
        exhausted_r = []
        best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int).T[0][0]
        best_array_shift = 1 / np.sqrt(n) * \
            np.exp(1j * np.pi * (2 * best_beam_index / k - 1) * np.arange(n).reshape(-1, 1))
        _, optimal_rss = channel_rss(best_array_shift, angle)
        optimal_reward = beam_gain_map(optimal_rss)
        head_node = Node(0)
        for t in range(1, terminal_time+1):
            # if break
            p = []  # start from head_node, left_child is 0, right_child is 1
            node = head_node
            # new node selection
            while 1:
                if (node.left_child is not None) and (node.right_child is not None):
                    if node.left_child.Q > node.right_child.Q:
                        node = node.left_child
                        p.append(0)
                        continue
                    if node.left_child.Q < node.right_child.Q:
                        node = node.right_child
                        p.append(1)
                        continue
                    if node.left_child.Q == node.right_child.Q:
                        if np.random.rand() >= 0.5:
                            node = node.left_child
                            p.append(0)
                        else:
                            node = node.right_child
                            p.append(1)
                        continue
                if (node.left_child is None) and (node.right_child is None):
                    if np.random.rand() >= 0.5:
                        node.left_child = Node(node.depth + 1)
                        node = node.left_child
                        p.append(0)
                    else:
                        node.right_child = Node(node.depth + 1)
                        node = node.right_child
                        p.append(1)
                    break
                if (node.left_child is None) and (node.right_child is not None):
                    node.left_child = Node(node.depth + 1)
                    node = node.left_child
                    p.append(0)
                    break
                if (node.left_child is not None) and (node.right_child is None):
                    node.right_child = Node(node.depth + 1)
                    node = node.right_child
                    p.append(1)
                    break
            if node.depth >= m:
                # if depth meet the condition, then take exhausted search in the region related to list p
                # the last measurement haven't been done, so the real converge time is t-1
                converge_time = t
                break
            # terminate the iteration
            # then attributes update
            beam_idx_start, beam_idx_medium, beam_idx_end = node_to_beam_idx(m, p)
            beam_measured_idx = beam_idx_medium
            array_shift = 1 / np.sqrt(n) * np.exp(
                1j * np.pi * (2 * beam_measured_idx / k - 1) * np.arange(n).reshape(-1, 1))
            noisy_rss, _ = channel_rss(array_shift, angle)
            measured_beam = np.append(measured_beam, beam_measured_idx)
            measured_rss = np.append(measured_rss, noisy_rss)
            r = beam_gain_map(noisy_rss)
            regret += optimal_reward-r
            regret_record = np.append(regret_record, regret)
            node = head_node
            # In fact, the value of head_node is no sense
            # Then the R value in P is updated
            for i in np.arange(len(p)):
                if p[i] == 0:
                    node = node.left_child
                if p[i] == 1:
                    node = node.right_child
                node.tree_p_reward_update(r)
            head_node.postorder_e(t)
            head_node.postorder_q()
        # Terminate the main iteration, the two deepest node should be measured. We find the index by list p.
        # In the last iteration, the measured beam is still helpful in latter code.
        # The result of node_to_beam_idx(m,p), the beam idx between beam_idx_start and end is going to be exhaustively
        # searched. However, one beam in them is already measured, which is the beam measured in last iteration
        beam_idx_start, _, beam_idx_end = node_to_beam_idx(m, p)
        print('----------\nin the function\nfrom', beam_idx_start, 'to', beam_idx_end, '\n----------')
        beam_measured_idx = beam_idx_start if beam_idx_end == beam_measured_idx else beam_idx_end
        array_shift = 1 / np.sqrt(n) * np.exp(
            1j * np.pi * (2 * beam_measured_idx / k - 1) * np.arange(n).reshape(-1, 1))
        noisy_rss, _ = channel_rss(array_shift, angle)
        measured_beam = np.append(measured_beam, beam_measured_idx)
        measured_rss = np.append(measured_rss, noisy_rss)
        r = beam_gain_map(noisy_rss)
        regret += optimal_reward - r
        regret_record = np.append(regret_record, regret)
        selected_beam_idx = measured_beam[-2] if measured_rss[-2] > measured_rss[-1] else measured_beam[-1]
        print(selected_beam_idx)
    else:
        measured_beam = []
        measured_rss = []
        regret = []
        terminal_time = 1000
        converge_time = 0
    return converge_time, terminal_time, selected_beam_idx, measured_beam, measured_rss, regret_record


def main():
    m = 7
    n = np.power(2, m)
    k = n
    l = 3
    angle = np.pi * np.random.rand(l, 1)
    best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int).T
    print('best_beam_index: ', best_beam_index)
    best_array_shift = 1 / np.sqrt(n) * \
        np.exp(1j * np.pi * (2 * best_beam_index / k - 1) * np.arange(n).reshape(-1, 1))
    _, optimal_rss = channel_rss(best_array_shift, angle)
    print('best_beam_rss', optimal_rss)
    method = 1
    converge_time, terminal_time, selected_beam_idx, measured_beam, measured_rss, regret_record \
        = simulation(m, n, k, angle, method)
    print('converge time: ', converge_time)
    print('measured_beam: ', measured_beam)
    print('measured_rss: ', measured_rss)
    print('regret_record: ', regret_record)

    print(1+np.arange(converge_time))
    print(regret_record)
    test = 1
    if test == 1:
        array_shift = \
            1 / np.sqrt(n) * \
            np.exp(1j * np.pi * np.dot(np.arange(n).reshape(-1, 1), 2 * np.arange(k).reshape(1, -1) / k - 1))
        noisy_rss, mean_rss = channel_rss(array_shift, angle)
        plt.figure(num='gain')
        plt.xlabel('beam idx')
        plt.ylabel('gain')
        plt.plot(np.arange(k), mean_rss, 'r--', np.arange(k), noisy_rss, 'k-')
        plt.grid()
    plt.figure(num='regret')
    plt.xlabel('time slot')
    plt.ylabel('cumulative regret')
    plt.plot(1+np.arange(converge_time), regret_record, 'k-')
    plt.show()


if __name__ == '__main__':
    main()
