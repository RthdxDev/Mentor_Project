import numpy as np
from typing import Any, List, Tuple
from numpy.typing import NDArray


class GFNumber:
    modulus = 2
    
    def __init__(self, number: int, modulus: int = 2):
        self.number: int = int(number)
        self.modulus: int = modulus

    def __add__(self, other: "GFNumber") -> "GFNumber":
        if not isinstance(other, GFNumber):
            return NotImplemented
        return GFNumber((self.number + other.number) % self.modulus, self.modulus)
    
    def __neg__(self) -> "GFNumber":
        return self.inverse_add()
    
    def __sub__(self, other: "GFNumber") -> "GFNumber":
        if not isinstance(other, GFNumber):
            return NotImplemented
        return self + (-other)
    
    def __mul__(self, other: "GFNumber") -> "GFNumber":
        if not isinstance(other, GFNumber):
            return NotImplemented
        return GFNumber((self.number * other.number) % self.modulus, self.modulus)
    
    def __truediv__(self, other: "GFNumber") -> "GFNumber":
        if not isinstance(other, GFNumber):
            return NotImplemented
        return self * other.inverse_mul()
    
    def inverse_add(self) -> "GFNumber":
        return GFNumber((self.modulus - self.number) % self.modulus, self.modulus)   

    def inverse_mul(self) -> "GFNumber":
        """
        Compute inverse via Fermat's Little Theorem.
        """
        a = self.number
        p = self.modulus
        if a == 0:
            raise ValueError("Zero has no inverse")
        if a % p == 0:
            raise ValueError(f"Some trouble: a ({a}) mod p ({p}) == 0")
        return GFNumber(pow(a, p - 2, p), p)
    
    def __repr__(self) -> str:
        return f"GF_Number({self.number}, {self.modulus})"


class GFPolynomial:
    def __init__(self, coefficients: List[GFNumber] | Tuple[GFNumber]):
        """
        coefficients: List of coefficients, starting from the constant term
        """
        self.coefficients = coefficients

    def degree(self) -> int:
        return len(self.coefficients) - 1
    
    def __len__(self) -> int:
        return len(self.coefficients)

    def __add__(self, other: "GFPolynomial | GFNumber") -> "GFPolynomial":
        if isinstance(other, GFNumber):
            other = GFPolynomial([other])
        n, m = len(self), len(other)
        i, j = 0, 0
        new_coeffs = []
        while i < n and j < m:
            new_coeffs.append(self.coefficients[i] + other.coefficients[j])
            i += 1
            j += 1
        while i < n:
            new_coeffs.append(self.coefficients[i])
            i += 1
        while j < m:
            new_coeffs.append(other.coefficients[j])
            j += 1
        return GFPolynomial(new_coeffs)
    
    def __radd__(self, other: "GFPolynomial | GFNumber") -> "GFPolynomial":
        return self + other
    
    def __neg__(self) -> "GFPolynomial":
        new_coeffs = [-c for c in self.coefficients]
        return GFPolynomial(new_coeffs)
    
    def __sub__(self, other: "GFPolynomial | GFNumber") -> "GFPolynomial":
        return self + (-other)

    def __mul__(self, other: "GFPolynomial | GFNumber") -> "GFPolynomial":
        if isinstance(other, GFNumber):
            other = GFPolynomial([other])
        new_coeffs = [GFNumber(0, self.coefficients[0].modulus) for _ in range(self.degree() + other.degree() + 1)]
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                new_coeffs[i + j] += a * b
        return GFPolynomial(new_coeffs)
    
    def __rmul__(self, other: "GFPolynomial | GFNumber") -> "GFPolynomial":
        return self * other
    
    def __getitem__(self, ind):
        if not (0 <= ind < len(self.coefficients)):
            raise IndexError(f"Index out of range: index = {ind}, but {self}")
        return self.coefficients[ind]
    
    def __call__(self, A: int | NDArray[np.integer] ) -> GFNumber | NDArray[np.integer]:
        if isinstance(A, int):
            result = GFNumber(0, self.coefficients[0].modulus)
            x_power = GFNumber(1, self.coefficients[0].modulus)
            for coeff in self.coefficients:
                result += coeff * x_power
                x_power *= GFNumber(A, self.coefficients[0].modulus)
        else:
            result = np.zeros_like(A)
            x_power = np.eye(A.shape[0], dtype=int)
            for coeff in self.coefficients:
                result = (result + coeff.number * x_power) % self.coefficients[0].modulus
                x_power = (x_power @ A) % self.coefficients[0].modulus
        return result

    def copy(self) -> "GFPolynomial":
        return GFPolynomial(self.coefficients[:])
    
    def __repr__(self) -> str:
        res = ""
        for i, coeff in enumerate(self.coefficients):
            if i == 0:
                res += f"{coeff.number}"
            else:
                res += f" + {coeff.number}x^{i}"
        return f"GF_Polynomial({res})"
