import unittest
import galois
import numpy as np
import scipy as sp
from main import berlekamp_massey, wiedeman, wiedeman_sparse, field

# np.random.seed(42)
modulus = field.characteristic

class TestBerlekampMassey(unittest.TestCase):
    def test_example_1(self):
        S = field([0, 1, 1, 2, 0, 2])
        result = berlekamp_massey(S)
        expected = galois.berlekamp_massey(S)
        expected_coeffs = expected.coeffs[::-1]
        self.assertEqual(list(result.coeffs), list(expected_coeffs))

    def test_example_2(self):
        S = field([1, 2, 1, 0])
        result = berlekamp_massey(S)
        expected = galois.berlekamp_massey(S)
        expected_coeffs = expected.coeffs[::-1]
        self.assertEqual(list(result.coeffs), list(expected_coeffs))

    def test_example_3(self):
        S = field([0, 0, 1, 1, 1, 0])
        result = berlekamp_massey(S)
        expected = galois.berlekamp_massey(S)
        expected_coeffs = expected.coeffs[::-1]
        self.assertEqual(list(result.coeffs), list(expected_coeffs))

    # def test_example_4(self):
    #     S = field([0])
    #     result = berlekamp_massey(S)
    #     expected = galois.berlekamp_massey(S)
    #     expected_coeffs = expected.coeffs[::-1]
    #     self.assertEqual(list(result.coeffs), list(expected_coeffs))

    def test_example_5(self):
        S = field([0, 1, 2])
        result = berlekamp_massey(S)
        expected = galois.berlekamp_massey(S)
        expected_coeffs = expected.coeffs[::-1]
        self.assertEqual(list(result.coeffs), list(expected_coeffs))

class TestWiedeman(unittest.TestCase):
    def test_wiedeman_2x2(self):
        A = field.Random((2, 2))
        while np.linalg.det(A) == 0:
            A = field.Random((2, 2))
        b = field.Random(2)
        result = wiedeman(A, b, verbose=False)
        expected = np.linalg.solve(A, b)
        self.assertEqual(list(result), list(expected))

    def test_wiedeman_5x5(self):
        A = field.Random((5, 5))
        while np.linalg.det(A) == 0:
            A = field.Random((5, 5))
        b = field.Random(5)
        result = wiedeman(A, b, verbose=False)
        expected = np.linalg.solve(A, b)
        self.assertEqual(list(result), list(expected))

    def test_wiedeman_50x50(self):
        A = field.Random((50, 50))
        while np.linalg.det(A) == 0:
            A = field.Random((50, 50))
        b = field.Random(50)
        result = wiedeman(A, b, verbose=False)
        expected = np.linalg.solve(A, b)
        self.assertEqual(list(result), list(expected))


def generate_sparse_matrix(n, density=0.5):
    A = sp.sparse.random(n, n, density=density, format='csr', dtype=int)
    A = sp.sparse.csr_matrix(A.toarray() % modulus)
    while np.linalg.det(A.toarray().view(field)) == 0:
        A = sp.sparse.random(n, n, density=density, format='csr', dtype=int)
        A = sp.sparse.csr_matrix(A.toarray() % modulus)
    return A


class TestWiedemanSparse(unittest.TestCase):
    def test_wiedeman_sparse_2x2(self):
        A = generate_sparse_matrix(2, density=0.5)
        b = field.Random(2)
        result = wiedeman_sparse(A, b, verbose=False)
        expected = np.linalg.solve(A.toarray().view(field), b)
        self.assertEqual(list(result), list(expected))

    def test_wiedeman_sparse_10x10(self):
        A = generate_sparse_matrix(10, density=0.5)
        b = field.Random(10)
        result = wiedeman_sparse(A, b, verbose=False)
        expected = np.linalg.solve(A.toarray().view(field), b)
        self.assertEqual(list(result), list(expected))

    def test_wiedeman_sparse_50x50(self):
        A = generate_sparse_matrix(50, density=0.5)
        b = field.Random(50)
        result = wiedeman_sparse(A, b, verbose=False)
        expected = np.linalg.solve(A.toarray().view(field), b)
        self.assertEqual(list(result), list(expected))


if __name__ == '__main__':
    unittest.main()