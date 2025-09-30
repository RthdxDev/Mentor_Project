import unittest
import numpy as np
from structures import GFNumber
from algorithms import berlekamp_massey, wiedeman


class Test_Algorithms(unittest.TestCase):

    def test_berlekamp_massey_single(self):
        S = [GFNumber(1, 2)]
        result = berlekamp_massey(S, 2)
        self.assertEqual([c.number for c in result.coefficients], [1, 1])  # x + 1

    def test_berlekamp_massey_all_zero(self):
        S = [GFNumber(0, 2), GFNumber(0, 2)]
        result = berlekamp_massey(S, 2)
        self.assertEqual([c.number for c in result.coefficients], [1])

    def test_berlekamp_massey_fibonacci_mod2(self):
        # Fibonacci mod 2: 1,1,0,1,1,0,...
        S = [
            GFNumber(1, 2), GFNumber(1, 2), GFNumber(0, 2),
            GFNumber(1, 2), GFNumber(1, 2), GFNumber(0, 2)
            ]
        result = berlekamp_massey(S, 2)
        # Minimal polynomial x^2 + x + 1
        self.assertEqual([c.number for c in result.coefficients], [1, 1, 1])

    def test_berlekamp_massey_fibonacci_mod3(self):
        # Fibonacci mod 3: 0,1,1,2,0,2,...
        S = [GFNumber(0, 3), GFNumber(1, 3), GFNumber(1, 3), GFNumber(2, 3), GFNumber(0, 3), GFNumber(2, 3)]
        result = berlekamp_massey(S, 3)
        # Minimal polynomial 2x^2 + 2x + 1
        self.assertEqual([c.number for c in result.coefficients], [1, 2, 2])

    def test_2_powers_mod2(self):
        S = [GFNumber(0, 2), GFNumber(0, 2), GFNumber(0, 2), GFNumber(0, 2)]
        result = berlekamp_massey(S, 2)
        # Minimal polynomial 1
        self.assertEqual([c.number for c in result.coefficients], [1])

    def test_2_powers_mod3(self):
        S = [GFNumber(1, 3), GFNumber(2, 3), GFNumber(1, 3), GFNumber(2, 3), GFNumber(1, 3)]
        result = berlekamp_massey(S, 3)
        # Minimal polynomial x + 2
        self.assertEqual([c.number for c in result.coefficients], [1, 1])

    def test_wiedeman_2x2_system_mod2(self):
        A = np.array([[1, 0], [0, 1]], dtype=int)
        b = np.array([1, 0], dtype=int)
        modulus = 2
        x = wiedeman(A, b, modulus)
        result = (A @ x) % modulus
        np.testing.assert_array_equal(result, b)

    def test_wiedeman_2x2_system_mod3(self):
        A = np.array([[1, 0], [0, 1]], dtype=int)
        b = np.array([2, 1], dtype=int)
        modulus = 3
        x = wiedeman(A, b, modulus)
        result = (A @ x) % modulus
        np.testing.assert_array_equal(result, b)

    def test_wiedeman_3x3_system_mod2(self):
        A = np.eye(3, dtype=int)
        b = np.array([1, 0, 1], dtype=int)
        modulus = 2
        x = wiedeman(A, b, modulus)
        result = (A @ x) % modulus
        np.testing.assert_array_equal(result, b)

    def test_wiedeman_3x3_system_mod3(self):
        A = np.eye(3, dtype=int)
        b = np.array([2, 1, 0], dtype=int)
        modulus = 3
        x = wiedeman(A, b, modulus)
        result = (A @ x) % modulus
        np.testing.assert_array_equal(result, b)

    def test_wiedeman_10x10_system_mod3(self):
        A = np.eye(10, dtype=int)
        b = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0, 0], dtype=int)
        modulus = 3
        x = wiedeman(A, b, modulus)
        result = (A @ x) % modulus
        np.testing.assert_array_equal(result, b)

    def test_wiedeman_10x10_complex_system_mod3(self):
        A = np.eye(10, dtype=int)
        A[0, -1] = 1
        A[-1, 0] = 1
        b = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0, 2], dtype=int)
        modulus = 3 
        x = wiedeman(A, b, modulus)
        result = (A @ x) % modulus
        np.testing.assert_array_equal(result, b)


if __name__ == '__main__':
    unittest.main()
