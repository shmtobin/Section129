# ----------------------------
# Part a: Define one class that contains all quadrature techniques.
# ----------------------------
class Quad:
    def __init__(self, func, a, b):
        self.func = func  # Function to integrate
        self.a = a        # Lower bound of integration
        self.b = b        # Upper bound of integration

    # ----------------------------
    # Part b: Implement Midpoint Rule
    # ----------------------------
    def midpoint_rule(self):
        midpoint = (self.a + self.b) / 2
        return (self.b - self.a) * self.func(midpoint)
    
    # ----------------------------
    # Part b: Implement Trapezoidal Rule
    # ----------------------------
    def trapezoidal_rule(self):
        return (self.b - self.a) / 2 * (self.func(self.a) + self.func(self.b))
    
    # ----------------------------
    # Part b: Implement Simpson's Rule
    # ----------------------------
    def simpsons_rule(self):
        midpoint = (self.a + self.b) / 2
        return (self.b - self.a) / 6 * (self.func(self.a) + 4 * self.func(midpoint) + self.func(self.b))

# ----------------------------
# Part c: Gauss-Legendre Quadrature
# ----------------------------
    def gauss_legendre_quadrature(self, weights, nodes):
        transformed_nodes = [(self.b - self.a) / 2 * x + (self.a + self.b) / 2 for x in nodes]
        integral = (self.b - self.a) / 2 * sum(w * self.func(x) for w, x in zip(weights, transformed_nodes))
        return integral

# ----------------------------
# Part d: Legendre Polynomials
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

class GaussQuad(Quad):
    def __init__(self, func, a, b, order):
        super().__init__(func, a, b)
        self.order = order
    
    def legendre_polynomial(self, M):
        return np.polynomial.legendre.Legendre.basis(M)
    
    def plot_legendre_polynomials(self):
        x = np.linspace(-1, 1, 100)
        plt.figure(figsize=(8,6))
        for M in range(1, 6):
            P_M = self.legendre_polynomial(M)
            plt.plot(x, P_M(x), label=f'P_{M}(x)')
        plt.xlabel('x')
        plt.ylabel('P_M(x)')
        plt.title('Legendre Polynomials')
        plt.legend()
        plt.grid()
        plt.show()

# ----------------------------
# Part e: Newton’s Method for Finding Roots and Weights
# ----------------------------
    def newton_legendre_roots_weights(self, M, tol=1e-10):
        P_M = self.legendre_polynomial(M)
        P_M_deriv = P_M.deriv()
        
        roots = np.cos(np.pi * (np.arange(1, M + 1) - 0.25) / (M + 0.5))  # Initial guess
        for _ in range(100):  # Limit iterations
            roots_new = roots - P_M(roots) / P_M_deriv(roots)
            if np.max(np.abs(roots - roots_new)) < tol:
                break
            roots = roots_new
        
        weights = 2 / ((1 - roots ** 2) * (P_M_deriv(roots) ** 2))
        return roots, weights
    
    def save_roots_weights(self, M_values, filename="gauss_legendre_roots_weights.txt"):
        with open(filename, "w") as file:
            for M in M_values:
                roots, weights = self.newton_legendre_roots_weights(M)
                file.write(f"M = {M}\n")
                file.write("Roots: " + " ".join(map(str, roots)) + "\n")
                file.write("Weights: " + " ".join(map(str, weights)) + "\n\n")

# Example Function for Quadrature
def example_func(x):
    return np.sin(x)

# ----------------------------
# Part a: Apply Example to Midpoint, Trapezoidal, and Simpson's Rules
# ----------------------------
quad = Quad(example_func, 0, np.pi)

# Midpoint Rule Example
midpoint_result = quad.midpoint_rule()
print(f"Midpoint Rule: {midpoint_result}")

# Trapezoidal Rule Example
trapezoidal_result = quad.trapezoidal_rule()
print(f"Trapezoidal Rule: {trapezoidal_result}")

# Simpson's Rule Example
simpsons_result = quad.simpsons_rule()
print(f"Simpson's Rule: {simpsons_result}")

# ----------------------------
# Part c: Apply Gauss-Legendre Quadrature (Example with weights and nodes)
# ----------------------------
# Example Gauss-Legendre weights and nodes for N=2
weights = [1, 1]  # Example weights for N=2
nodes = [-1/np.sqrt(3), 1/np.sqrt(3)]  # Example nodes for N=2

gauss_quad = Quad(example_func, 0, np.pi)
gauss_result = gauss_quad.gauss_legendre_quadrature(weights, nodes)
print(f"Gauss-Legendre Quadrature Result: {gauss_result}")

# ----------------------------
# Part d: Plot Legendre Polynomials for M = [1, 2, 3, 4, 5]
# ----------------------------
gauss_quad = GaussQuad(example_func, 0, np.pi, order=5)
gauss_quad.plot_legendre_polynomials()

# ----------------------------
# Part e: Apply Newton's Method for Finding Roots and Weights
# ----------------------------
gauss_quad.save_roots_weights([1, 2, 3, 4, 5])