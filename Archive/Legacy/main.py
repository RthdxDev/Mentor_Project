import numpy as np
import scipy as sp
import galois
from tqdm import tqdm
from typing import TypeAlias, Callable, Tuple, Optional


FieldPoly: TypeAlias = galois.Poly
FieldMatrix: TypeAlias = galois.FieldArray
FieldCSRMatrix: TypeAlias = sp.sparse.csr.csr_matrix

field = galois.GF(3)
modulus = field.characteristic

def berlekamp_massey(S: FieldMatrix) -> FieldPoly:
    N: int = len(S)
    C: FieldPoly = galois.Poly([1], field=field)
    # if N == 1:
    #     return C
    B: FieldPoly = galois.Poly([1], field=field)
    T: FieldPoly = galois.Poly([0], field=field)
    L: int = 0
    m: int = 1
    b: FieldMatrix = field([1])
    for n in range(N):
        d = (C.coeffs * S[n-L:n+1]).sum()
        if (d == 0):
            m += 1
        elif 2 * L <= n:
            T = C
            xm: FieldPoly = galois.Poly([1] + [0] * m, field=field)
            C   = C - (d / b) * xm * B
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            xm: FieldPoly = galois.Poly([1] + [0] * m, field=field)
            C = C - (d / b) * xm * B
            m += 1
    return C

def wiedeman(A: FieldMatrix, b: FieldMatrix, verbose: bool=True) -> FieldMatrix:

    def poly_on_matrix(p: FieldPoly, M: FieldMatrix, b: FieldMatrix) -> FieldMatrix:
        M_k =  b
        result =  b * p.coeffs[-1]
        for coeff in reversed(p.coeffs[:-1]):
            M_k = M @ M_k
            result += coeff * M_k
        return result

    n = A.shape[0]
    b_k: FieldMatrix = b
    y_k: FieldMatrix = field.Zeros(n)
    k: int = 0
    d_k: int = 0
    while not all(b_k == 0):
        if verbose:
            print(f"Iteration {k + 1}, d_k = {d_k}")
        u_k = field.Random(n)
        seq: FieldMatrix = field.Zeros(2 * (n - d_k))
        w_k = b_k.copy()
        seq[0] = u_k @ w_k
        for i in range(1, 2 * (n - d_k)):
            w_k = A @ w_k
            seq[i] = u_k @ w_k
        f_k = galois.berlekamp_massey(seq)
        f_k_deg = f_k.degree
        if f_k.degree == 0:
            f_k_minus: FieldPoly = FieldPoly([0], field=field)
        else:
            c0 = f_k.coeffs[-1]
            if c0 == 0:
                continue
            f_k = (c0 ** -1) * f_k
            f_k_minus: FieldPoly = FieldPoly(f_k.coeffs[:-1], field=field)
        y_k += poly_on_matrix(f_k_minus, A, b_k)
        b_k = b + A @ y_k
        d_k += f_k_deg
        k += 1
    if verbose:
        print(f"Finished in {k} iterations.")
    return -y_k

def wiedeman_sparse(A: FieldCSRMatrix, b: FieldMatrix, verbose: bool=True) -> FieldMatrix:
    modulus = field.characteristic

    def poly_on_matrix(p: FieldPoly, M: FieldCSRMatrix, b: FieldMatrix) -> FieldMatrix:
        M_k = b
        result = b * p.coeffs[-1]
        for i, coeff in enumerate(reversed(p.coeffs[:-1])):
            M_k = ((M @ M_k) % modulus).view(field)
            result += M_k * coeff
        return result

    n = A.shape[0]
    b_k: FieldMatrix = b
    y_k: FieldMatrix = field.Zeros(n)
    k: int = 0
    d_k: int = 0
    while not all(b_k == 0):
        if verbose:
            print(f"Iteration {k + 1}, d_k = {d_k}")
        u_k = field.Random(n)
        seq: FieldMatrix = field.Zeros(2 * (n - d_k))
        w_k = b_k.copy()
        seq[0] = u_k @ w_k
        for i in range(1, 2 * (n - d_k)):
            w_k = (A @ w_k % modulus).view(field)
            seq[i] = u_k @ w_k
        f_k = galois.berlekamp_massey(seq)
        f_k_deg = f_k.degree
        if f_k.degree == 0:
            f_k_minus: FieldPoly = FieldPoly([0], field=field)
        else:
            c0 = f_k.coeffs[-1]
            if c0 == 0:
                k += 1
                continue
            f_k = (c0 ** -1) * f_k
            f_k_minus: FieldPoly = FieldPoly(f_k.coeffs[:-1], field=field)
        y_k += poly_on_matrix(f_k_minus, A, b_k)
        b_k = b + (A @ y_k % modulus).view(field)
        d_k += f_k_deg
        k += 1

    return -y_k


def matvec(M: FieldCSRMatrix, x: FieldMatrix) -> FieldMatrix:
    modulus = field.characteristic
    return (M @ x % modulus).view(field)


def poly_on_matrix(
        matvec: Callable[[FieldCSRMatrix, FieldMatrix], FieldMatrix], 
        p: FieldPoly, M: FieldMatrix, b: FieldMatrix
        ) -> FieldMatrix:
    M_k = b
    result = b * p.coeffs[-1]
    for coeff in reversed(p.coeffs[:-1]):
        M_k = matvec(M, M_k)
        result += coeff * M_k
    return result


def wiedeman_singular(
        matvec: Callable[[FieldCSRMatrix, FieldMatrix], FieldMatrix],
        A: FieldCSRMatrix, b: FieldMatrix, verbose: bool = True
        ) -> Tuple[Optional[FieldMatrix], Optional[FieldMatrix]]:
    """
    Solve Ax = b using Wiedemann algorithm.
    Returns:
        (solution, None) if matrix is nonsingular
        (None, kernel_element) if matrix is singular
    """
    n = A.shape[0]
    b_k: FieldMatrix = b
    y_k: FieldMatrix = field.Zeros(n)
    k: int = 0
    d_k: int = 0
    
    while not all(b_k == 0):
        if verbose:
            print(f"Iteration {k + 1}, d_k = {d_k}")
        
        u_k = field.Random(n)
        seq: FieldMatrix = field.Zeros(2 * (n - d_k))
        w_k = b_k.copy()
        seq[0] = u_k @ w_k
        
        for i in range(1, 2 * (n - d_k)):
            w_k = matvec(A, w_k)
            seq[i] = u_k @ w_k
        
        f_k = galois.berlekamp_massey(seq)
        c0 = f_k.coeffs[-1]
        
        if c0 == 0:
            if f_k.degree == 0:
                continue
            
            f_k_minus = FieldPoly(f_k.coeffs[:-1], field=field)
            kernel_element = poly_on_matrix(matvec, f_k_minus, A, b_k)
            
            if verbose:
                print(f"Matrix is singular. Found kernel element.")
            
            return None, kernel_element
        
        f_k = (c0 ** -1) * f_k
        
        if f_k.degree == 0:
            f_k_minus = FieldPoly([0], field=field)
        else:
            f_k_minus = FieldPoly(f_k.coeffs[:-1], field=field)

        y_k += poly_on_matrix(matvec, f_k_minus, A, b_k)
        b_k = b + matvec(A, y_k)
        d_k += f_k.degree
        k += 1
    
    if verbose:
        print(f"Converged in {k} iterations.")
    
    return -y_k, None

