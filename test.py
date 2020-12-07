import numpy as np
from matplotlib import pyplot as plt
from integrated_channel_and_test import channel_rss


def node_to_beam_idx(m, p):
    # m is the binary number of beam. For example, beams = 128, m = log2(128) = 7.
    # p is the list of tree p recorded in every iteration
    # Obviously, when len(p) should is more than m, the output is too rough to distinguish
    n = np.power(2, m)
    l = len(p)
    a = np.arange(l)+1
    x_l = np.dot(np.array(p), np.power(1/2, a))
    x_h = x_l+np.power(1/2, l)
    x_a = (x_l+x_h)/2
    [beam_l, beam_a, beam_h] = np.mod(np.round(np.array([x_l, x_a, x_h])*n), n)
    return beam_l, beam_a, beam_h


def beam_gain_map(gain):
    # the limit set is flexible
    upper_limit = -40
    lower_limit = -90
    r = (gain-lower_limit)/(upper_limit-lower_limit)
    if r < 0:
        r = 0
    if r > 1:
        r = 1
    return r


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
        self.N = self.N+1
        self.R = ((self.N-1)*self.R+r)/self.N

    def tree_t_estimated_reward_update(self, t):
        # Attributes update in T tree
        if self.N == 0:
            self.E = np.inf
        else:
            self.E = \
                self.R+np.sqrt(2*np.power(self.sigma, 2)*np.log(t) / self.N)+self.rho1*np.power(self.gama, self.depth)

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


def main():
    # n and k are the number of antennas and beams. l is the number of path
    m = 7
    n = np.power(2, m)
    k = n
    l = 3
    # randomly generate the AoD of different path
    angle = np.pi * np.random.rand(l, 1)
    # the AoD is associated with its array response, the index of best beam is easy to get from the complex conjugate of
    # array response
    best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k).astype(np.int).T
    # HBA algorithm
    t = 0
    head_node = Node(0)
    while 1:
        t += 1
        print('\ntime slot: ', t)
        p = []  # start from head_node, left_child is 0, right_child is 1
        node = head_node
        # new node selection: the new node that selected are naturally added to the tree T, whose head node is head_node
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
                    node.left_child = Node(node.depth+1)
                    node = node.left_child
                    p.append(0)
                else:
                    node.right_child = Node(node.depth+1)
                    node = node.right_child
                    p.append(1)
                break
            if (node.left_child is None) and (node.right_child is not None):
                node.left_child = Node(node.depth+1)
                node = node.left_child
                p.append(0)
                break
            if (node.left_child is not None) and (node.right_child is None):
                node.right_child = Node(node.depth+1)
                node = node.right_child
                p.append(1)
                break
        # terminate the iteration
        # then attributes update
        beam_idx_start, beam_idx_medium, beam_idx_end = node_to_beam_idx(m, p)
        # beam_measured_idx = np.floor((beam_idx_start+beam_idx_end)/2)
        beam_measured_idx = beam_idx_medium
        array_shift = 1 / np.sqrt(n) * np.exp(1j * np.pi * (2*beam_measured_idx / k-1) * np.arange(n).reshape(-1, 1))
        noisy_rss, mean_rss = channel_rss(array_shift, angle)
        print('measured beam idx:', beam_measured_idx)
        print('measured beam RSS:', noisy_rss)
        r = beam_gain_map(noisy_rss)
        if node.depth >= m:
            break
        node = head_node
        if noisy_rss > -45:
            break
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
    beam_idx_start, beam_idx_medium, beam_idx_end = node_to_beam_idx(m, p)
    print('\nTerminating the iteration\n------------------------------------')
    print('converge time: ', t)
    print('the HBA result of beam idx: between', beam_idx_start, 'and', beam_idx_end)
    print('the best beam idx:', best_beam_index)
    # if want to see the RSS function over beam space, then set test = 1
    test = 1
    if test == 1:
        print('test channel_rss:')
        array_shift = \
            1 / np.sqrt(n) * \
            np.exp(1j * np.pi * np.dot(np.arange(n).reshape(-1, 1), 2 * np.arange(k).reshape(1, -1) / k - 1))
        noisy_rss, mean_rss = channel_rss(array_shift, angle)
        plt.figure(num='gain')
        plt.xlabel('beam idx')
        plt.ylabel('gain')
        plt.plot(np.arange(k), mean_rss, 'r--', np.arange(k), noisy_rss, 'k-')
        plt.plot(best_beam_index, -40 * np.ones_like(best_beam_index), 'ro')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    main()
