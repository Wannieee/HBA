import sys
import numpy as np
import matplotlib.pyplot as plt
# Main function of test.py is too big to integrate. In simulation function, when method=1, HBA algorithm is used
upper_limit = -40
lower_limit = -100


def node_to_beam_idx(m, p):
    # m为数目的2次方阶数，p为常规列表
    # m is the binary number of beam. For example, beams = 128, m = log2(128) = 7.
    # p is the list of tree p recorded in every iteration
    # Obviously, when len(p) should is more than m, the output is too rough to distinguish
    n = np.power(2, m)
    l = len(p)
    a = np.arange(l) + 1
    x_l = np.dot(np.array(p), np.power(1 / 2, a))
    x_h = x_l + np.power(1 / 2, l)
    x_a = (x_l + x_h) / 2
    [beam_l, beam_a, beam_h] = np.mod(np.round(np.array([x_l, x_a, x_h]) * n), n).astype(np.int)
    return beam_l, beam_a, beam_h


def beam_gain_map(gain):
    # 输入什么格式就输出什么格式
    # the limit set is flexible
    # I think that the rss isn't able to be more than -40dbm, and the lower limit may be set as -90 or -100 in this code
    r = (gain - lower_limit) / (upper_limit - lower_limit)
    if r < 0:
        r = 0
    if r > 1:
        r = 1
    return r


def code(n, k, number):
    # DFT码本，返回一个n*k维的波束赋形向量构成的矩阵
    if number == 1:
        array_shift = \
            1 / np.sqrt(n) * \
            np.exp(1j * np.pi * np.dot(np.arange(n).reshape(-1, 1), 2 * np.arange(k).reshape(1, -1) / k - 1))
        pass
    else:
        print('\033[31m生成波束码本时未指定正确编号\033[0m')
        sys.exit(0)
    return array_shift


def angle_to_beam(k, angle, number):  # 通常方向与波束编号有直接关系->采用唯一一个特定编号可以在指定方向产生最大波束
    angle = np.array(angle).reshape(1, -1)
    # DFT码本
    if number == 1:
        beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int)[0]
    else:
        print('\033[31m未指定波束码本\033[0m')
        sys.exit(0)
    return beam_index


def beam_to_angle(k, beam, number):
    beam = np.array(beam).reshape(1, -1)
    # DFT码本
    if number == 1:
        angle = np.arccos(1-2*beam/k)[0]  # 实际上对于线阵，在angle和2*pi-angle上的波束响应是一样的
        pass
    else:
        print('\033[31m未指定波束码本\033[0m')
        sys.exit(0)
    return angle


class Node:
    def __init__(self, h=None):
        # As the value of Q, R, E is updated with node itself, some simulation parameters are embedded in the Node class
        # sigma is shadow fading, as the paper shows, rho1*gama^depth accounts for the maximum variation of the mean
        # reward function in region C(h,j), where rho1 > 0 and gama belong to (0, 1)
        self.sigma = 2
        self.rho1 = 3
        self.gama = 0.5
        self.depth = h
        self.N = 0
        self.R = 0
        self.E = np.inf
        self.Q = np.inf
        self.left_child = None
        self.right_child = None

    def tree_p_reward_update(self, r):
        # Attributes update in P tree
        self.N = self.N + 1
        self.R = ((self.N - 1) * self.R + r) / self.N

    def tree_t_estimated_reward_update(self, t):
        # Attributes update in T tree
        if self.N == 0:
            self.E = np.inf
        else:
            # choose=1,2,3 stand for origin HBA, adjusted HBA and origin HOO
            choose = 1
            if choose == 1:
                # HBA: the parameter sigma is divided by 60, because the sigma(dbm) is the variance of shadow
                # fading. The RSS is mapped to [0, 1], then the sigma maybe also zoomed out the same size,
                # which is -40+100
                self.E = self.R + np.sqrt(2 * np.power(self.sigma / (upper_limit - lower_limit), 2) * np.log(t)/self.N)\
                    + self.rho1 * np.power(self.gama, self.depth)
            if choose == 2:
                # HOO: as the paper shows, the HOO algorithm set a /eta_h as 0.1, the parameter c_1 is equal to rho1
                self.E = self.R + 0.1 * np.sqrt(2 * np.log(t) / self.N) \
                    + self.rho1 * np.power(self.gama, self.depth)

                # when use the adjust HBA, a better performance than HOO should be gotten

    def postorder_e(self, t):  # update E value of t tree in the postorder
        if self.left_child is not None:
            self.left_child.postorder_e(t)
        if self.right_child is not None:
            self.right_child.postorder_e(t)
        if self.depth is not None:
            self.tree_t_estimated_reward_update(t)
            # print(self.depth)

    def postorder_q(self):  # update Q value of t tree in the postorder
        if self.left_child is not None:
            self.left_child.postorder_q()
        if self.right_child is not None:
            self.right_child.postorder_q()
        if self.depth is not None:
            if (self.left_child is None) or (self.right_child is None):
                self.Q = self.E
            else:
                self.Q = np.minimum(self.E, np.maximum(self.left_child.Q, self.right_child.Q))
        # print(self.depth)


def channel_rss(array_shift, angle, sf=1, d=20e-3, sigma=2, xi=1.74, f=60e3):
    # 输入的array_shift为列向量的组合，如果单独输入dim为1的列向量则将其改为dim为2的列向量
    array_shift = np.array(array_shift)
    if array_shift.ndim == 1:
        array_shift = array_shift.reshape(-1, 1)
    n = array_shift.shape[0]  # n antennas
    k = array_shift.shape[1]  # k beams
    p = 50 - 10 * np.log10(n)
    l = angle.shape[0]
    g = np.zeros((l, k), dtype=np.float)
    loss = 32.5 + 20 * np.log10(f) + 10 * xi * np.log10(d)
    if sf == 1:
        chi = sigma*np.random.normal(size=(1, k))
    else:
        chi = sigma * np.zeros((1, k))
    for i in range(l):
        if i == 0:
            g[i, :] = -loss - chi
        else:
            g[i, :] = -loss - chi - (7 + 6 * np.random.rand(1, k)) * sf - 10 * (1 - sf)
    array_response = np.exp(1j * np.pi * np.dot(np.cos(angle).reshape(-1, 1), np.arange(n).reshape(1, -1)))
    array_gain = np.power(np.abs(np.dot(array_response, array_shift)), 2)
    flag = 2
    if flag == 1:
        noisy_rss = 20 * np.log10((np.multiply(np.sqrt(np.power(10, g / 10)), np.sqrt(array_gain))).sum(axis=0)) + p
        g = np.dot(10 * np.log10((np.power(10, g / 10)).mean(axis=1).reshape(-1, 1)), np.ones((1, k)))
        # g = np.dot(g.mean(axis=1).reshape(-1, 1), np.ones((1, k), dtype=np.float))
        mean_rss = 20 * np.log10((np.multiply(np.sqrt(np.power(10, g / 10)), np.sqrt(array_gain))).sum(axis=0)) + p
    else:
        noisy_rss = 10 * np.log10((np.multiply(np.power(10, g / 10), array_gain)).sum(axis=0)) + p
        g = np.dot(10 * np.log10((np.power(10, g / 10)).mean(axis=1).reshape(-1, 1)), np.ones((1, k)))
        # g = np.dot(g.mean(axis=1).reshape(-1, 1), np.ones((1, k), dtype=np.float)) #两者差别是前者对dB做平均，后者对倍数做平均
        mean_rss = 10 * np.log10((np.multiply(np.power(10, g / 10), array_gain)).sum(axis=0)) + p
    return noisy_rss, mean_rss


def simulation(m, n, k, angle, cb_method, ba_method):
    array_shift = code(n, k, cb_method)
    # Terminal and converge time is used to describe the performance of different algorithm
    if ba_method == 1:
        # For HBA method, n=k=2^m.
        measured_beam = []
        measured_rss = []
        exhausted_rss = []
        measured_reward = []
        terminal_time = 1000
        converge_time = 0
        p = []
        best_beam_index = angle_to_beam(k, angle, cb_method)[0]
        best_array_shift = array_shift[:, best_beam_index]
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
            test_array_shift = array_shift[:, beam_measured_idx]
            noisy_rss, _ = channel_rss(test_array_shift, angle)
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
                measured_array_shift = array_shift[:, beam_idx[i]]
                new_rss, _ = channel_rss(measured_array_shift, angle)
                measured_beam.append(beam_idx[i])
                # print('append', beam_idx[i])
                measured_rss = np.append(measured_rss, new_rss)
                converge_time += 1
            exhausted_rss = np.append(exhausted_rss, new_rss)
        print(exhausted_rss)
        # selected_beam_idx = beam_idx[exhausted_rss.tolist().index(max(exhausted_rss))]
        selected_beam_idx = beam_idx[exhausted_rss.argmax()]
        # print('select', selected_beam_idx)
        # print('-------------')
        # calculate the cumulative regret
        selected_array_shift = array_shift[:, selected_beam_idx]
        if converge_time < terminal_time:
            for t in range(converge_time, terminal_time):
                new_rss, _ = channel_rss(selected_array_shift, angle)
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
    m = 9
    n = np.power(2, m)
    k = n
    l = 3
    cb_method = 1  # codebook method，等于1表示采用DFT码本
    ba_method = 1  # beam alignment method，等于1表示采用HBA
    array_shift = code(n, k, cb_method)
    angle = np.pi * np.random.rand(l, 1)
    # angle = np.array([0.5, 1, 2]).reshape(-1, 1)
    best_beam_index = angle_to_beam(k, angle, cb_method)
    print("the scale of beam code and antenna: ", n)
    print('\033[7mTheoretical optimal result\033[0m')
    print('best_beam_index: ', best_beam_index)
    best_array_shift = array_shift[:, best_beam_index]
    _, optimal_rss = channel_rss(best_array_shift, angle)
    print('best_beam_rss', optimal_rss)
    converge_time, terminal_time, selected_beam_idx, measured_beam, measured_rss, regret_record \
        = simulation(m, n, k, angle, cb_method, ba_method)
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

    test_stability = 0
    test_generalization = 0
    repeat_time = 20
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
            converge_time, _, selected_beam_idx, _, _, regret_record \
                = simulation(m, n, k, angle, cb_method, ba_method)
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
            best_beam_index = angle_to_beam(k, test_angle, cb_method)[0]
            converge_time, _, selected_beam_idx, _, _, regret_record\
                = simulation(m, n, k, test_angle, cb_method, ba_method)
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
