import numpy as np
from typing import List
from structures import GFNumber, GFPolynomial
from numpy.typing import NDArray


def berlekamp_massey(S: List[GFNumber], modulus: int) -> GFPolynomial:
    N = len(S)
    p = modulus
    C = GFPolynomial([GFNumber(1, p)])
    B = GFPolynomial([GFNumber(1, p)])
    L = 0
    m = 1
    b = GFNumber(1, p)
    for n in range(N):
        d = S[n]
        for j in range(1, L + 1):
            d += C[j] * S[n - j]
        if d.number == 0:
            m += 1
        elif 2 * L <= n:
            T = C.copy()
            xm = GFPolynomial([GFNumber(0, p)] * m + [GFNumber(1, p)])
            C = C - (d * b.inverse_mul()) * xm * B
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            xm = GFPolynomial([GFNumber(0, p)] * m + [GFNumber(1, p)])
            C = C - (d * b.inverse_mul()) * xm * B
            m += 1
    return C


def wiedeman(A: NDArray[np.integer], b: NDArray[np.integer], modulus: int) -> NDArray[np.integer]:
    n = A.shape[0]
    t = b
    table = [t]
    for _ in range(1, 2 * n):
        t = (A @ t) % modulus
        table.append(t)
    k = 0
    g_k = GFPolynomial([GFNumber(1, modulus)])
    while g_k.degree() < n and k < n:
        d = g_k.degree()
        u_k = np.zeros((1, n), dtype=int)
        u_k[0, k] = 1
        s = [(u_k @ A_kb)[0] for A_kb in table]
        new_s = [
            GFNumber(sum(g_k[j].number * s[j + i] for j in range(d + 1)), modulus)
            for i in range(len(s) - d)
        ]
        f_k = berlekamp_massey(new_s, modulus)
        g_k = f_k * g_k
        k += 1
    x: NDArray[np.integer] = np.zeros((n,), dtype=int)
    for i in range(1, g_k.degree() + 1):
        x = (x + g_k[i].number * table[i - 1]) % modulus
    return -x
