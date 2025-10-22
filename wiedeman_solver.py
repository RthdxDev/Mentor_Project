import numpy as np
import scipy as sp
import galois
from typing import TypeAlias, Tuple, Optional


FieldPoly: TypeAlias = galois.Poly
FieldMatrix: TypeAlias = galois.FieldArray
FieldCSRMatrix: TypeAlias = sp.sparse.csr.csr_matrix

class WiedemanSolver:
    def __init__(self, field_characteristic: int) -> None:
        self.k = 0
        self.field = galois.GF(field_characteristic)
        self.modulus = self.field.characteristic

    def matvec(self, M: FieldCSRMatrix, x: FieldMatrix) -> FieldMatrix:
        modulus = self.field.characteristic
        return (M @ x % modulus).view(self.field)

    def masked_matvec(
            self, A: FieldCSRMatrix,
            x: FieldMatrix, mask: np.ndarray
            ) -> FieldMatrix:
        x_masked = x.copy()
        x_masked[~mask] = 0
        result = self.matvec(A, x_masked)
        return result

    def apply_matrix_poly(
            self, p: FieldPoly, A: FieldCSRMatrix,
            b: FieldMatrix, mask: np.ndarray
            ) -> FieldMatrix:
        
        M_k = b.copy()
        result = b * p.coeffs[-1]
        for coeff in reversed(p.coeffs[:-1]):
            M_k = self.masked_matvec(A, M_k, mask)
            result += M_k * coeff
        return result

    def wiedeman_singular_iteration(
            self, A: FieldCSRMatrix, b: FieldMatrix,
            mask: np.ndarray, verbose: bool = True
            ) -> Tuple[Optional[FieldMatrix], Optional[FieldMatrix], int]:
        
        n = A.shape[0]
        b_k = b.copy()
        y_k = self.field.Zeros(n)
        k = 0
        d_k = 0
        
        while not np.all(b_k == 0):
            if verbose:
                print(f"  Iteration {k + 1}, d_k = {d_k}")

            u_k = self.field.Random(n)

            if d_k >= n:
                return None, None, d_k
            
            seq = self.field.Zeros(2 * (n - d_k))
            w_k = b_k.copy()
            seq[0] = u_k @ w_k

            for i in range(1, 2 * (n - d_k)):
                w_k = self.masked_matvec(A, w_k, mask)
                seq[i] = u_k @ w_k
            
            f_k = galois.berlekamp_massey(seq)
            c0 = f_k.coeffs[-1]
            
            if c0 == 0:
                if f_k.degree == 0:
                    continue
                
                f_k_minus = FieldPoly(f_k.coeffs[:-1], field=self.field)
                kernel_element = self.apply_matrix_poly(f_k_minus, A, b_k, mask)
                
                if verbose:
                    print(f"Matrix is singular!")
                    print(f"Current accumulated degree: {d_k + f_k.degree}")
                
                return None, kernel_element, d_k + f_k.degree
            
            f_k = (c0 ** -1) * f_k
            
            if f_k.degree == 0:
                f_k_minus = FieldPoly([0], field=self.field)
            else:
                f_k_minus = FieldPoly(f_k.coeffs[:-1], field=self.field)
            
            y_k_update = self.apply_matrix_poly(f_k_minus, A, b_k, mask)
            y_k += y_k_update
            b_k = b + self.masked_matvec(A, y_k, mask)
            d_k += f_k.degree
            self.k += 1
            k += 1
        
        return -y_k, None, d_k

    def wiedeman_singular(
            self, A: FieldCSRMatrix,
            b: FieldMatrix, verbose: bool = True
            ) -> Tuple[Optional[FieldMatrix], Optional[np.ndarray]]:
        
        self.k = 0
        n = A.shape[0]
        m = A.shape[1]
        mask = np.ones(m, dtype=bool)
        max_iterations = m
        
        for iteration in range(max_iterations):
            if verbose:
                n_active = mask.sum()
                print(f"\n=== Elimination iteration {iteration + 1} ===")
                print(f"Active columns: {n_active}/{m}")
            
            solution_partial, kernel, d_k_global = self.wiedeman_singular_iteration(
                A, b, mask, verbose=verbose
            )
            
            if solution_partial is not None:
                solution_full = solution_partial.copy()
                solution_full[~mask] = 0
                eliminated = ~mask
                
                if verbose:
                    print(f"\n+++ Solution found! +++")
                
                return solution_full, np.where(eliminated)[0]
            
            if kernel is None:
                if verbose:
                    print(f"\n--- No solution and no kernel element found ---")
                return None, None

            kernel_active = kernel.copy()
            kernel_active[~mask] = 0

            nonzero_positions = np.asarray(kernel_active != 0).nonzero()[0]

            if len(nonzero_positions) == 0:
                if verbose:
                    print("Kernel element has no nonzero active positions")
                return None, None

            global_idx = nonzero_positions[0]
            mask[global_idx] = False
            
            if mask.sum() == 0:
                if verbose:
                    print("\n--- All columns eliminated - system has no solution ---")
                return None, None
            
            if d_k_global >= n:
                if verbose:
                    print(f"\n-- Accumulated degree {d_k_global} reached matrix dimension {n} ---")
                break
        
        if verbose:
            print(f"\n--- Max iterations reached ---")
        return None, None