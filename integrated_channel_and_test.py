import numpy as np
import matplotlib.pyplot as plt


# angle is the AoD of different path
# sf determine if consider the shadow fading, xi is path loss exponent
# d is the distance between transmitter and receiver
# f is the frequency of mmWave
# array_shift is the matrix of shift value with size of (n, k), where every column in is a shift value vector, the k
# represent the number of the shift vector at now the function is called, there is actually no discrimination
# either the result is got from a shift value matrix nor the result matrix is formed by the output of making every
# column of the shift value matrix input. When I call the function, I usually either make array_shift a DFT codebook or
# just a single shift value vector
def channel_rss(array_shift, angle, d=20e-3, sf=1, sigma=2, xi=1.74, f=60e3):
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
            g[i, :] = -loss - chi - 7 - 6 * np.random.rand(1, k)
    array_response = np.exp(1j * np.pi * np.dot(np.cos(angle).reshape(-1, 1), np.arange(n).reshape(1, -1)))
    array_gain = np.power(np.abs(np.dot(array_response, array_shift)), 2)
    flag = 1
    if flag == 1:
        noisy_rss = 20 * np.log10((np.multiply(np.sqrt(np.power(10, g / 10)), np.sqrt(array_gain))).sum(axis=0)) + p
        g = np.dot(10 * np.log10((np.power(10, g / 10)).mean(axis=1).reshape(-1, 1)), np.ones((1, k)))
        # g = np.dot(g.mean(axis=1).reshape(-1, 1), np.ones((1, k), dtype=np.float)) #两者差别是一个是对dB做平均一个是对倍数做平均
        mean_rss = 20 * np.log10((np.multiply(np.sqrt(np.power(10, g / 10)), np.sqrt(array_gain))).sum(axis=0)) + p
    else:
        noisy_rss = 10 * np.log10((np.multiply(np.power(10, g / 10), array_gain)).sum(axis=0)) + p
        g = np.dot(10 * np.log10((np.power(10, g / 10)).mean(axis=1).reshape(-1, 1)), np.ones((1, k)))
        # g = np.dot(g.mean(axis=1).reshape(-1, 1), np.ones((1, k), dtype=np.float)) #两者差别是一个是对dB做平均一个是对倍数做平均
        mean_rss = 10 * np.log10((np.multiply(np.power(10, g / 10), array_gain)).sum(axis=0)) + p
    return noisy_rss, mean_rss


def main():
    n = 128
    k = 128
    l = 2
    angle = np.pi * np.random.rand(l, 1)
    best_beam_index = np.mod(np.round(k * (1 - np.cos(angle)) / 2, 0), k)  # 反推最优的波束编号，但只能从近似到0:K-1中的一个值
    array_shift = 1 / np.sqrt(n) * np.exp(
        1j * np.pi * np.dot(np.arange(n).reshape(-1, 1), 2 * np.arange(k).reshape(1, -1) / k - 1))
    noisy_rss, mean_rss = channel_rss(array_shift, angle)
    plt.figure(num='the gain of different beam in the DFT codebook')
    plt.xlabel('beam index')
    plt.ylabel('gain')
    plt.title('')
    plt.ylim((-100, -40))
    plt.plot(np.arange(k), mean_rss, 'r--', np.arange(k), noisy_rss, 'k-')
    plt.grid()
    plt.plot(best_beam_index, -50 * np.ones_like(best_beam_index), 'ro')
    plt.show()


if __name__ == '__main__':
    main()
