import unittest
import numpy as np
from structures import GFNumber, GFPolynomial

class Test_Number_Arithmetic(unittest.TestCase):

    def test_addition_mod2(self):
        a = GFNumber(1, 2)
        b = GFNumber(1, 2)
        result = a + b
        self.assertEqual(result.number, 0)
        self.assertEqual(result.modulus, 2)

    def test_subtraction_mod2(self):
        a = GFNumber(1, 2)
        b = GFNumber(1, 2)
        result = a - b
        self.assertEqual(result.number, 0)
        self.assertEqual(result.modulus, 2)

    def test_multiplication_mod2(self):
        a = GFNumber(1, 2)
        b = GFNumber(1, 2)
        result = a * b
        self.assertEqual(result.number, 1)
        self.assertEqual(result.modulus, 2)

    def test_inverse_mod2(self):
        a = GFNumber(1, 2)
        inv = a.inverse_mul()
        self.assertEqual(inv.number, 1)
        self.assertEqual(inv.modulus, 2)

    def test_inverse_zero_mod2_raises_error(self):
        a = GFNumber(0, 2)
        with self.assertRaises(ValueError):
            a.inverse_mul()

    def test_division_mod2(self):
        a = GFNumber(1, 2)
        b = GFNumber(1, 2)
        result = a / b
        self.assertEqual(result.number, 1)
        self.assertEqual(result.modulus, 2)

    def test_addition_mod5(self):
        a = GFNumber(3, 5)
        b = GFNumber(4, 5)
        result = a + b
        self.assertEqual(result.number, 2)
        self.assertEqual(result.modulus, 5)

    def test_multiplication_mod5(self):
        a = GFNumber(3, 5)
        b = GFNumber(4, 5)
        result = a * b
        self.assertEqual(result.number, 2)
        self.assertEqual(result.modulus, 5)

    def test_inverse_mod5(self):
        a = GFNumber(2, 5)
        inv = a.inverse_mul()
        self.assertEqual(inv.number, 3)
        self.assertEqual(inv.modulus, 5)

    def test_division_mod5(self):
        a = GFNumber(4, 5)
        b = GFNumber(2, 5)
        result = a / b
        self.assertEqual(result.number, 2)
        self.assertEqual(result.modulus, 5)

    def test_negation_mod2(self):
        a = GFNumber(1, 2)
        result = -a
        self.assertEqual(result.number, 1)
        self.assertEqual(result.modulus, 2)

    def test_negation_mod5(self):
        a = GFNumber(3, 5)
        result = -a
        self.assertEqual(result.number, 2)
        self.assertEqual(result.modulus, 5)

    def test_subtraction_mod5(self):
        a = GFNumber(3, 5)
        b = GFNumber(4, 5)
        result = a - b
        self.assertEqual(result.number, 4)
        self.assertEqual(result.modulus, 5)


class Test_Polynomial(unittest.TestCase):

    def test_init_valid(self):
        coeffs = [GFNumber(1, 2), GFNumber(0, 2), GFNumber(1, 2)]  # x^2 + 1
        p = GFPolynomial(coeffs)
        self.assertEqual([c.number for c in p.coefficients], [1, 0, 1])

    def test_degree(self):
        coeffs = [GFNumber(1, 2), GFNumber(0, 2), GFNumber(1, 2)]  # degree 2
        p = GFPolynomial(coeffs)
        self.assertEqual(p.degree(), 2)

    def test_len(self):
        coeffs = [GFNumber(1, 2), GFNumber(0, 2), GFNumber(1, 2)]  # 3 coeffs
        p = GFPolynomial(coeffs)
        self.assertEqual(len(p), 3)

    def test_add_same_degree(self):
        p1 = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        p2 = GFPolynomial([GFNumber(1, 2)])  # 1
        result = p1 + p2
        self.assertEqual([c.number for c in result.coefficients], [0, 1])  # 0 + x

    def test_add_different_degree(self):
        p1 = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        p2 = GFPolynomial([GFNumber(1, 2), GFNumber(0, 2), GFNumber(1, 2)])  # 1 + x^2
        result = p1 + p2
        self.assertEqual([c.number for c in result.coefficients], [0, 1, 1])  # 0 + x + x^2

    def test_neg(self):
        p = GFPolynomial([GFNumber(1, 2), GFNumber(0, 2), GFNumber(1, 2)])  # 1 + x^2
        result = -p
        self.assertEqual([c.number for c in result.coefficients], [1, 0, 1])  # -1 - x^2 mod 2 = 1 + x^2

    def test_mul(self):
        p1 = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        p2 = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        result = p1 * p2
        self.assertEqual([c.number for c in result.coefficients], [1, 0, 1])  # 1 + x^2

    def test_add_scalar(self):
        p = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        s = GFNumber(1, 2)  # 1
        result = p + s
        self.assertEqual([c.number for c in result.coefficients], [0, 1])  # 0 + x

    def test_sub_scalar(self):
        p = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        s = GFNumber(1, 2)  # 1
        result = p - s
        self.assertEqual([c.number for c in result.coefficients], [0, 1])  # 0 + x

    def test_mul_scalar(self):
        p = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        s = GFNumber(1, 2)  # 1
        result = p * s
        self.assertEqual([c.number for c in result.coefficients], [1, 1])  # 1 + x

    def test_evaluate_at_point_mod2(self):
        p = GFPolynomial([GFNumber(1, 2), GFNumber(0, 2), GFNumber(1, 2)])  # 1 + x^2
        result = p(1)  # x=1, 1 + 1^2 = 0 mod 2
        assert isinstance(result, GFNumber)
        self.assertEqual(result.number, 0)

    def test_evaluate_at_point_mod3(self):
        p = GFPolynomial([GFNumber(2, 3), GFNumber(1, 3)])  # 2 + x
        result = p(2)  # 2 + 2 = 4 ≡ 1 mod 3
        assert isinstance(result, GFNumber)
        self.assertEqual(result.number, 1)

    def test_evaluate_constant(self):
        p = GFPolynomial([GFNumber(3, 5)])  # 3
        result = p(2)  # 3
        assert isinstance(result, GFNumber)
        self.assertEqual(result.number, 3)

    def test_evaluate_at_matrix_mod2(self):
        p = GFPolynomial([GFNumber(1, 2), GFNumber(1, 2)])  # 1 + x
        A = np.array([[1, 0], [1, 1]], dtype=int)  # matrix mod 2
        result = p(A)
        # p(A) = I + A
        expected = (np.eye(2, dtype=int) + A) % 2
        np.testing.assert_array_equal(result, expected)

    def test_evaluate_at_matrix_mod3(self):
        p = GFPolynomial([GFNumber(1, 3), GFNumber(2, 3)])  # 1 + 2x
        A = np.array([[1, 2], [0, 1]], dtype=int)  # matrix mod 3
        result = p(A)
        # p(A) = I + 2A
        expected = (np.eye(2, dtype=int) + 2 * A) % 3
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
