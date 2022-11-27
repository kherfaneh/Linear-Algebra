##def gamesolver(cells):
import numpy as np

adj = [[0, 0], [0, -1], [-1, 0], [0, 1], [1, 0]]
cells = np.array([[1, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0],
                  [1, 0, 0, 0, 1]])
n = len(cells)
cell = np.reshape(cells, [n * n, 1])


def xor2(a, b):
    c = a + b - 2 * a * b
    return c


adjs = []
ch = np.zeros([n * n, n * n], 'int32')
co = -1

for x in range(n):
    for y in range(n):
        c = np.zeros([n, n], 'int32')
        for i, j in adj:
            if (0 <= i + x < len(cells)) and (0 <= j + y < len(cells)):
                adjs += [[i + x, j + y]]
                c[adjs[-1][0], adjs[-1][1]] = 1
        co = co + 1
        ch[co, :] = np.reshape(c, [1, n * n])

ch1 = np.append(ch, cell, axis=1)
ch2 = np.copy(ch1)

for i in range(n * n):
    for j in range(n * n):
        if j != i:
            if ch2[j, i] == 1:
                for k in range(n * n + 1):
                    ch2[j, k] = xor2(ch2[j, k], ch2[i, k])

    if i <= n * n - 2 and ch2[i + 1, i + 1] == 0:
        for t in range(i + 1, n * n):
            if t != i + 1 and ch2[t, i + 1] == 1 and ch2[i + 1, i + 1] == 0:
                for kk in range(n * n + 1):
                    ch2[i + 1, kk] = xor2(ch2[i + 1, kk], ch2[t, kk])

print(np.reshape(ch2[:, n * n], [n, n]))

