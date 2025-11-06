import numpy as np
import scipy as sp
import galois
from typing import TypeAlias, Tuple, Optional

FieldPoly: TypeAlias = galois.Poly
FieldMatrix: TypeAlias = galois.FieldArray
FieldCSRMatrix: TypeAlias = sp.sparse.csr_matrix


class WiedemannSolver:
    def __init__(self, field_characteristic: int) -> None:
        self.k = 0
        self.field = galois.GF(field_characteristic)
        self.modulus = self.field.characteristic

    def matvec(self, A: FieldCSRMatrix, b: FieldMatrix) -> FieldMatrix:
        return (A @ b % self.modulus).view(self.field)

    def matvec_masked(self, A: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray) -> FieldMatrix:
        b_active = b.copy()
        b_active[mask_zero] = 0
        return self.matvec(A, b_active)

    def apply_matrix_poly(self, p: FieldPoly, A: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray) -> FieldMatrix:
        M_k = b.copy()
        result = b * p.coeffs[-1]
        for coeff in reversed(p.coeffs[:-1]):
            M_k = self.matvec_masked(A, M_k, mask_zero)
            result += M_k * coeff
        return result

    def __wiedemann_iteration(self, A: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray, verbose: bool = True) -> Tuple[Optional[FieldMatrix], Optional[FieldMatrix], int]:
        n = A.shape[0]
        b_k = b.copy()
        y_k = self.field.Zeros(n)
        d_k = 0

        while not np.all(b_k == 0):
            if verbose:
                print(f"  Iteration {self.k + 1}, d_k = {d_k}")

            u_k = self.field.Random(n)
            if d_k >= n:
                # return None, None, d_k
                raise NotImplementedError(
                    "Handling for d_k >= n is not implemented yet.")
            seq = self.field.Zeros(2 * (n - d_k))
            w_k = b_k
            seq[0] = u_k @ w_k
            for i in range(1, 2 * (n - d_k)):
                w_k = self.matvec_masked(A, w_k, mask_zero)
                seq[i] = u_k @ w_k
            f_k = galois.berlekamp_massey(seq)
            c0 = f_k.coeffs[-1]

            if c0 == 0:
                if f_k.degree == 0:
                    self.k += 1
                    continue
                f_k_minus = FieldPoly(f_k.coeffs[:-1], field=self.field)
                kernel_vector = self.apply_matrix_poly(
                    f_k_minus, A, b_k, mask_zero)
                self.k += 1
                return None, kernel_vector, d_k + f_k.degree
            f_k = (c0 ** -1) * f_k
            if f_k.degree == 0:
                f_k_minus = FieldPoly([0], field=self.field)
            else:
                f_k_minus = FieldPoly(f_k.coeffs[:-1], field=self.field)
            y_k += self.apply_matrix_poly(f_k_minus, A, b_k, mask_zero)
            b_k = b + self.matvec_masked(A, y_k, mask_zero)
            d_k += f_k.degree
            self.k += 1
        return -y_k, None, d_k

    def solve(
        self, A: FieldCSRMatrix, b: FieldMatrix, verbose: bool = True
    ) -> Tuple[Optional[FieldMatrix], Optional[FieldMatrix]]:
        self.k = 0
        n = A.shape[0]
        m = A.shape[1]
        mask_zero = np.array([], dtype=int)
        max_iterations = m
        for iteration in range(max_iterations):
            partial_solution, kernel_vector, d_k = self.__wiedemann_iteration(
                A, b, mask_zero, verbose
            )
            if partial_solution is not None:
                full_solution = partial_solution
                full_solution[mask_zero] = 0
                return full_solution, None
            if kernel_vector is None or np.all(kernel_vector == 0):
                raise NotImplementedError(
                    "Handling for no kernel vector is not implemented yet.")
            nonzero_positions = np.asarray(kernel_vector != 0).nonzero()[0][0]
            mask_zero = np.concatenate(
                (mask_zero, [nonzero_positions])
            )
            if verbose:
                print(
                    f"  Updated eliminated columns, size: {mask_zero.size}"
                )
            if mask_zero.size == 0:
                raise NotImplementedError(
                    "Handling for cycle mask_zero is not implemented yet.")
            if d_k >= n:
                raise NotImplementedError(
                    "Handling for d_k >= n is not implemented yet.")
        if verbose:
            print("Maximum iterations reached without finding a solution.")
        return None, None
