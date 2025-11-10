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

    def __matvec_masked(self, A: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray) -> FieldMatrix:
        if mask_zero.size == 0:
            return self.matvec(A, b)

        mask_active = np.ones(len(b), dtype=bool)
        mask_active[mask_zero] = False
        A_active = A[:, mask_active]
        b_active = b[mask_active].view(self.field)
        return self.matvec(A_active, b_active)

    def __apply_matrix_poly(self, p: FieldPoly, A: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray, matvec) -> FieldMatrix:
        M_k = b.copy()
        result = b * p.coeffs[-1]
        for coeff in reversed(p.coeffs[:-1]):
            M_k = matvec(A, M_k, mask_zero)
            result += M_k * coeff
        return result

    def __wiedemann_iteration(self, A: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray, matvec, verbose: bool = True) -> Tuple[Optional[FieldMatrix], Optional[FieldMatrix], int]:
        n = A.shape[1]
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
                w_k = matvec(A, w_k, mask_zero)
                seq[i] = u_k @ w_k
            f_k = galois.berlekamp_massey(seq)
            c0 = f_k.coeffs[-1]

            if c0 == 0:
                if f_k.degree == 0:
                    self.k += 1
                    continue
                f_k_minus = FieldPoly(f_k.coeffs[:-1], field=self.field)
                kernel_vector = self.__apply_matrix_poly(
                    f_k_minus, A, b_k, mask_zero, matvec)
                self.k += 1
                return None, kernel_vector, d_k + f_k.degree
            f_k = (c0 ** -1) * f_k
            if f_k.degree == 0:
                f_k_minus = FieldPoly([0], field=self.field)
            else:
                f_k_minus = FieldPoly(f_k.coeffs[:-1], field=self.field)
            y_k += self.__apply_matrix_poly(f_k_minus,
                                            A, b_k, mask_zero, matvec)
            b_k = b + matvec(A, y_k, mask_zero)
            d_k += f_k.degree
            self.k += 1
        return -y_k, None, d_k

    def solve(
        self, A: FieldCSRMatrix, b: FieldMatrix, preconditioner: Optional[FieldCSRMatrix] = None, verbose: bool = True
    ) -> Tuple[Optional[FieldMatrix], Optional[FieldMatrix]]:
        self.k = 0
        n = A.shape[0]
        m = A.shape[1]
        mask_zero = np.array([], dtype=int)
        max_iterations = m
        for iteration in range(max_iterations):
            if preconditioner is None:
                partial_solution, kernel_vector, d_k = self.__wiedemann_iteration(
                    A, b, mask_zero, self.__matvec_masked, verbose
                )
            else:

                def lambda_matvec(A, b, mask_zero): return self.__matvec_preconditioned(
                    A, preconditioner, b, mask_zero)

                partial_solution, kernel_vector, d_k = self.__wiedemann_iteration(
                    A, b, mask_zero, lambda_matvec, verbose
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

    def __create_sparse_preconditioner(self, n_rows: int, n_cols: int, lambda_param: int = 3) -> FieldCSRMatrix:
        """
        Currently works only with GF(2) matrices.
        """
        if (self.modulus != 2):
            raise ValueError(
                "Sparse preconditioner can be created now only for GF(2).")

        rows = []
        cols = []

        for i in range(n_rows):
            selected_cols = np.random.choice(
                n_cols, size=lambda_param, replace=False)
            for j in selected_cols:
                rows.append(i)
                cols.append(j)

        data = np.ones(len(rows), dtype=np.uint8)
        R = sp.sparse.csr_matrix(
            (data, (cols, rows)),
            shape=(n_cols, n_rows),
        )
        return R

    def __precondition_system(self, A: FieldCSRMatrix, lambda_param=3):
        n_rows, n_cols = A.shape
        R = self.__create_sparse_preconditioner(n_rows, n_cols, lambda_param)
        return A, R

    def __matvec_preconditioned(self, A: FieldCSRMatrix, R: FieldCSRMatrix, b: FieldMatrix, mask_zero: np.ndarray = np.array([], dtype=int)) -> FieldMatrix:
        if mask_zero.size == 0:
            temp = self.matvec(A, b)
            return self.matvec(R, temp)
        temp = self.__matvec_masked(A, b, mask_zero=mask_zero)
        return self.matvec(R, temp)

    def find_kernel_vector(self, A: FieldCSRMatrix, preconditioning: bool = False, verbose: bool = True) -> FieldMatrix:
        n = A.shape[1]
        r: FieldMatrix = self.field.Random(n)
        if preconditioning:
            A, R = self.__precondition_system(A)
            y: FieldMatrix = self.__matvec_preconditioned(A, R, r)
        else:
            y: FieldMatrix = self.matvec(A, r)
            R = None
        x = self.solve(A, y, preconditioner=R, verbose=verbose)[0]
        if x is None:
            raise ValueError(
                "No solution found; cannot compute kernel vector.")
        return x - r
