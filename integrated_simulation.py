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
        exhausted_rss = []
        measured_reward = []
        terminal_time = 1000
        converge_time = 0
        p = []
        best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int).T[0][0]
        best_array_shift = 1 / np.sqrt(n) * \
            np.exp(1j * np.pi * (2 * best_beam_index / k - 1) * np.arange(n).reshape(-1, 1))
        _, optimal_rss = channel_rss(best_array_shift, angle, 0)
        optimal_reward = beam_gain_map(optimal_rss)
        head_node = Node(0)
        for t in range(1, terminal_time+1):  # from 1 to terminal_time
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
                converge_time = t-1
                break
            # terminate the iteration
            # then attributes update
            _, beam_medium_idx, _ = node_to_beam_idx(m, p)
            beam_measured_idx = beam_medium_idx
            array_shift = 1 / np.sqrt(n) * np.exp(
                1j * np.pi * (2 * beam_measured_idx / k - 1) * np.arange(n).reshape(-1, 1))
            noisy_rss, _ = channel_rss(array_shift, angle)
            measured_beam.append(beam_measured_idx)
            # print(beam_measured_idx)
            measured_rss = np.append(measured_rss, noisy_rss)
            r = beam_gain_map(noisy_rss)
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
        p.pop()
        beam_start_idx, beam_medium_idx, beam_end_idx = node_to_beam_idx(m, p)
        # There, beam_end_idx = beam_start_idx + 2. Max rss is produced in the three beam, where more than one beam is
        # already measured.
        beam_idx = [beam_start_idx, beam_medium_idx, beam_end_idx]
        # print('-----------\n test\n', beam_idx)
        for i in range(len(beam_idx)):
            # print(beam_idx[i])
            if beam_idx[i] in measured_beam:
                new_rss = measured_rss[measured_beam.index(beam_idx[i])]
            else:
                array_shift = 1 / np.sqrt(n) * np.exp(
                    1j * np.pi * (2 * beam_idx[i] / k - 1) * np.arange(n).reshape(-1, 1))
                new_rss, _ = channel_rss(array_shift, angle)
                measured_beam.append(beam_idx[i])
                # print('append', beam_idx[i])
                measured_rss = np.append(measured_rss, new_rss)
                converge_time += 1
            exhausted_rss = np.append(exhausted_rss, new_rss)
        selected_beam_idx = beam_idx[exhausted_rss.tolist().index(max(exhausted_rss))]
        # print('select', selected_beam_idx)
        # print('-------------')
        # calculate the cumulative regret
        array_shift = 1 / np.sqrt(n) * np.exp(
            1j * np.pi * (2 * selected_beam_idx / k - 1) * np.arange(n).reshape(-1, 1))
        if converge_time < terminal_time:
            for t in range(converge_time, terminal_time):
                new_rss, _ = channel_rss(array_shift, angle)
                measured_beam.append(selected_beam_idx)
                measured_rss = np.append(measured_rss, new_rss)
        length = len(measured_rss)
        for r in measured_rss:
            measured_reward = np.append(measured_reward, beam_gain_map(r))
        regret_record = \
            np.dot(optimal_reward-measured_reward.reshape(1, -1), np.triu(np.ones([length, length])))
        regret_record = regret_record[0]
    else:
        measured_beam = []
        measured_rss = []
        regret_record = []
        terminal_time = 1000
        converge_time = 0
        selected_beam_idx = None
    return converge_time, terminal_time, selected_beam_idx, measured_beam, measured_rss.tolist(), regret_record.tolist()


def main():
    m = 7
    n = np.power(2, m)
    k = n
    l = 3
    # angle = np.pi * np.random.rand(l, 1)
    angle = np.array([0.5, 1, 2]).reshape(-1, 1)
    best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int).T
    print("the scale of beam code and antenna: ", n)
    print('\033[7mTheoretical optimal result\033[0m')
    print('best_beam_index: ', best_beam_index[0])
    best_array_shift = 1 / np.sqrt(n) * \
        np.exp(1j * np.pi * (2 * best_beam_index / k - 1) * np.arange(n).reshape(-1, 1))
    _, optimal_rss = channel_rss(best_array_shift, angle)
    print('best_beam_rss', optimal_rss)
    method = 1
    converge_time, terminal_time, selected_beam_idx, measured_beam, measured_rss, regret_record \
        = simulation(m, n, k, angle, method)
    print('\033[7mThe result of HBA:\033[0m')
    print('selected beam: ', selected_beam_idx)
    print('terminal time:', terminal_time)
    print('converge time: ', converge_time)
    print('measured_beam: ', measured_beam)
    print('measured_rss: ', measured_rss)
    print('regret_record: ', regret_record)
    print(len(regret_record))
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
        plt.plot(np.arange(terminal_time)+1, regret_record, 'k-')
    test_stability = 1
    test_generalization = 0
    repeat_time = 10
    ts_converge_time = []
    ts_selected_beam = []
    ts_best_beam = []
    ts_regret = []
    tg_converge_time = []
    tg_selected_beam = []
    tg_best_beam = []
    tg_regret = []
    if test_stability == 1:
        process_time = 0
        while repeat_time != process_time:
            process_time += 1
            print(process_time)
            converge_time, _, selected_beam_idx, _, _, regret_record = simulation(m, n, k, angle, method)
            ts_converge_time.append(converge_time)
            ts_selected_beam.append(selected_beam_idx)
            ts_best_beam.append(best_beam_index[0][0])
            ts_regret.append(regret_record[-1])
    if test_generalization == 1:
        process_time = 0
        while repeat_time != process_time:
            process_time += 1
            print(process_time)
            test_angle = np.pi * np.random.rand(l, 1)
            best_beam_index = np.mod(np.round(k * (1 - np.cos(test_angle)) / 2, 0), k).astype(np.int).T[0][0]
            converge_time, _, selected_beam_idx, _, _, regret_record = simulation(m, n, k, test_angle, method)
            tg_converge_time.append(converge_time)
            tg_selected_beam.append(selected_beam_idx)
            tg_best_beam.append(best_beam_index)
            tg_regret.append(regret_record[-1])
    # 一般认为没对准时会导致最后的regret偏大，一点就是证明波束没选到最优导致regret显著偏大；另一方面regret协方差偏大也说明没最优对准
    if test_stability == 1:
        ts_count = [ts_selected_beam[i]-ts_best_beam[i] for i in range(repeat_time)].count(0)
        print("test stability. accuracy:", ts_count/repeat_time)
    if test_generalization == 1:
        tg_count = [tg_selected_beam[i]-tg_best_beam[i] for i in range(repeat_time)].count(0)
        print("test generalization. accuracy:", tg_count/repeat_time)

    plt.show()


if __name__ == '__main__':
    main()
