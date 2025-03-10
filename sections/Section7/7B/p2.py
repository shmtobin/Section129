#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from matplotlib.colors import LogNorm
from scipy.stats import norm

# ----------------------------
# a) Write down the explicit the formula in LaTeX for surface area.
# ----------------------------
"""
LaTeX Answer:

The closed-form expression for the surface area is given by

$$
A = 2\pi \beta^2 \\left( 1 + \\frac{c}{a\\,e}\\sin^{-1}(e) \\right),
\\quad e = \\sqrt{1 - \\frac{\\beta^2}{c^2}}, \\quad a = 1.
$$

Here the ellipsoid is written in standard form as 
$$
\\frac{x^2}{1^2} + \\frac{y^2}{\\beta^2} + \\frac{z^2}{c^2} = 1,
$$ 
so that \(a=1\).
"""

# ----------------------------
# b) Deterministic Quadrature: Midpoint Rule and Gaussian Quadrature
# ----------------------------
#
# We use the following one-dimensional integral representation:
#
#   I = ∫₀¹ [1/√(1 – e² u²)] du = sin⁻¹(e)/e,
#
# so that the surface area becomes
#
#   A = 2π β² (1 + (c/(a e)) I).
#
# (Recall that here a = 1.)

def integrand(u, e):
    """Integrand for the 1D representation."""
    return 1.0 / np.sqrt(1 - (e**2) * (u**2))

def exact_integral(e):
    """Exact value of the integral: sin⁻¹(e)/e."""
    return np.arcsin(e) / e

def midpoint_rule(e, N):
    """Approximate I using the midpoint rule with N subintervals."""
    u = np.linspace(0, 1, N, endpoint=False) + 0.5/N
    h = 1.0 / N
    return np.sum(integrand(u, e)) * h

def gaussian_quadrature(e, N):
    """Approximate I using N-point Gaussian quadrature on [0,1]."""
    x, w = leggauss(N)  # nodes and weights for [-1,1]
    # Transform nodes to [0,1]: u = 0.5*(x+1) and adjust weights
    u = 0.5 * (x + 1)
    w = 0.5 * w
    return np.sum(w * integrand(u, e))

def surface_area_numerical(beta, c, I):
    """Compute numerical surface area given I, beta, and c (with a=1)."""
    a = 1
    return 2 * np.pi * (beta**2) * (1 + (c / (a * e_function(beta, c))) * I)

def surface_area_exact(beta, c):
    """Compute the 'exact' surface area from the closed-form formula."""
    a = 1
    e = e_function(beta, c)
    return 2 * np.pi * (beta**2) * (1 + (c / (a * e)) * np.arcsin(e))

def e_function(beta, c):
    """Compute the eccentricity e = sqrt(1 - beta^2/c^2). Assumes c > beta."""
    return np.sqrt(1 - (beta**2) / (c**2))

# Example computation for given beta and c:
beta_example = 0.8
c_example = 1.2
e_example = e_function(beta_example, c_example)

I_mid = midpoint_rule(e_example, N=1000)
I_gauss = gaussian_quadrature(e_example, N=5)

A_mid = surface_area_numerical(beta_example, c_example, I_mid)
A_gauss = surface_area_numerical(beta_example, c_example, I_gauss)
A_ex = surface_area_exact(beta_example, c_example)

print("Example for beta = {:.2f}, c = {:.2f}".format(beta_example, c_example))
print("Surface Area (Midpoint Rule): {:.6f}".format(A_mid))
print("Surface Area (Gaussian Quadrature): {:.6f}".format(A_gauss))
print("Surface Area (Exact formula): {:.6f}".format(A_ex))

# Now, let us plot the error heatmap.
# We vary beta and c (only compute when c > beta so that e is real).

betas = np.logspace(-3, 3, 50)
cs = np.logspace(-3, 3, 50)
error_mid = np.zeros((len(betas), len(cs)))
error_gauss = np.zeros((len(betas), len(cs)))

for i, beta in enumerate(betas):
    for j, c in enumerate(cs):
        if c > beta:  # valid region for real e
            e_val = e_function(beta, c)
            I_mid_val = midpoint_rule(e_val, N=1000)
            I_gauss_val = gaussian_quadrature(e_val, N=5)
            A_mid_val = surface_area_numerical(beta, c, I_mid_val)
            A_gauss_val = surface_area_numerical(beta, c, I_gauss_val)
            A_ex_val = surface_area_exact(beta, c)
            error_mid[i, j] = np.abs(A_mid_val - A_ex_val)
            error_gauss[i, j] = np.abs(A_gauss_val - A_ex_val)
        else:
            error_mid[i, j] = np.nan
            error_gauss[i, j] = np.nan

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(error_mid, extent=[cs[0], cs[-1], betas[0], betas[-1]], origin='lower',
           aspect='auto', norm=LogNorm())
plt.colorbar(label='Absolute Error (Midpoint)')
plt.xlabel('c')
plt.ylabel(r'$\beta$')
plt.title('Error Heatmap (Midpoint Rule)')

plt.subplot(1, 2, 2)
plt.imshow(error_gauss, extent=[cs[0], cs[-1], betas[0], betas[-1]], origin='lower',
           aspect='auto', norm=LogNorm())
plt.colorbar(label='Absolute Error (Gaussian Quadrature)')
plt.xlabel('c')
plt.ylabel(r'$\beta$')
plt.title('Error Heatmap (Gaussian Quadrature)')

plt.tight_layout()
plt.show()

# Note:
# The error heatmaps show how the numerical error (difference from the closed-form value)
# depends on beta and c. Since the integrand 1/sqrt(1 - e^2 u^2) becomes more singular
# as e approaches 1, one may expect that the relative error will depend on the chosen beta and c.

# ----------------------------
# c) Monte Carlo Integration for Surface Area
# ----------------------------
#
# Here we use a simple Monte Carlo method to approximate
#
#   I = ∫₀¹ [1/√(1 – e² u²)] du,
#
# then compute A = 2π β² (1 + (c/(a e)) I).
# For this part we set 2β = c = 1, so that beta = 0.5 and c = 1.

beta_mc = 0.5
c_mc = 1.0
a = 1  # as before
e_mc = e_function(beta_mc, c_mc)  # e = sqrt(1 - (0.5)^2/1^2) = sqrt(0.75)

def monte_carlo_integral(e, N):
    """Estimate the integral I using N Monte Carlo samples (uniformly distributed in [0,1])."""
    u_samples = np.random.uniform(0, 1, N)
    return np.mean(integrand(u_samples, e))

N_values = [10, 100, 1000, 10000, 100000]
errors_mc = []

print("\nMonte Carlo Integration Results (beta=0.5, c=1):")
for N in N_values:
    I_mc = monte_carlo_integral(e_mc, N)
    A_mc = surface_area_numerical(beta_mc, c_mc, I_mc)
    A_ex_mc = surface_area_exact(beta_mc, c_mc)
    error = np.abs(A_mc - A_ex_mc)
    errors_mc.append(error)
    print("N = {:6d} | Monte Carlo A = {:.6f} | Exact A = {:.6f} | Error = {:.6f}".format(N, A_mc, A_ex_mc, error))

plt.figure()
plt.loglog(N_values, errors_mc, marker='o')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Absolute Error')
plt.title('Monte Carlo Integration Error for Surface Area\n(beta=0.5, c=1)')
plt.grid(True, which="both", ls="--")
plt.show()

# For parts c, d, and f we work with the same ellipsoid parameters:
beta_mc = 0.5
c_mc = 1.0
e_mc = e_function(beta_mc, c_mc)  # For beta=0.5 and c=1, e = sqrt(0.75)
I_exact = np.arcsin(e_mc) / e_mc  # Exact value of the 1D integral

# ========================================================================
# d) Importance Sampling with Inverse Transformation Sampling for Proposal q1 and q2
# ========================================================================
#
# We wish to approximate the integral
#   I = ∫₀¹ f(u) du  with f(u)=1/sqrt(1-e^2 u^2)
# using importance sampling. In importance sampling the estimator is:
#
#   I_est = (1/N)*∑ [ f(u_i) / p(u_i) ],
#
# where the u_i are drawn from a proposal density p(u).
#
# Here we use two proposals:
#
#   q1(u) = exp(–3u)    (properly normalized on [0,1])
#   q2(u) = sin²(5u)     (properly normalized on [0,1])
#
# For q1 the inverse CDF is available analytically, while for q2 we
# perform inverse transform sampling numerically.

# --- Proposal q1: Exponential ---
def sample_q1(N):
    """Generate N samples from q1(u) ∝ exp(-3u) on [0,1] via inverse transform."""
    U = np.random.uniform(0, 1, N)
    # CDF for q1: F(x) = (1 - exp(-3*x)) / (1 - exp(-3))
    # Inversion: x = -(1/3) * ln(1 - U*(1 - exp(-3)) )
    return - (1/3) * np.log(1 - U * (1 - np.exp(-3)))

def density_q1(x):
    """Return the probability density of q1 at x."""
    # Normalization constant: Z = ∫₀¹ exp(-3u) du = (1 - exp(-3))/3.
    return 3 * np.exp(-3*x) / (1 - np.exp(-3))

# --- Proposal q2: sin²(5u) ---
# Precompute the normalized CDF for q2
x_grid = np.linspace(0, 1, 10000)
# Unnormalized CDF: F(x) = ∫₀ˣ sin²(5t) dt = x/2 - sin(10x)/20.
F_grid = x_grid / 2 - np.sin(10 * x_grid) / 20
F_total = 0.5 - np.sin(10) / 20  # Normalization factor (F(1))
F_grid_norm = F_grid / F_total  # Normalized CDF: runs from 0 to 1

def sample_q2(N):
    """Generate N samples from q2(u) ∝ sin²(5u) on [0,1] using inverse transform via interpolation."""
    U = np.random.uniform(0, 1, N)
    return np.interp(U, F_grid_norm, x_grid)

def density_q2(x):
    """Return the probability density of q2 at x."""
    # The unnormalized density is sin^2(5x). The normalized density is:
    return (np.sin(5*x)**2) / F_total

def importance_sampling_integral(e, N, sample_func, density_func):
    """
    Estimate the integral I = ∫₀¹ 1/sqrt(1-e^2 u^2) du using N samples drawn 
    from a proposal distribution given by sample_func (with density given by density_func).
    """
    samples = sample_func(N)
    # For values outside [0,1] (should not occur), we set the integrand to 0.
    vals = integrand(samples, e) / density_func(samples)
    return np.mean(vals)

# Monte Carlo sample sizes to test:
N_values = [10, 100, 1000, 10000, 100000]

# Compute the estimator error for three methods: Uniform, q1, and q2.
def monte_carlo_uniform(e, N):
    """Standard MC integration using uniform samples on [0,1]."""
    u_samples = np.random.uniform(0, 1, N)
    return np.mean(integrand(u_samples, e))

errors_uniform = []
errors_q1 = []
errors_q2 = []

for N in N_values:
    I_uniform = monte_carlo_uniform(e_mc, N)
    I_q1 = importance_sampling_integral(e_mc, N, sample_q1, density_q1)
    I_q2 = importance_sampling_integral(e_mc, N, sample_q2, density_q2)
    errors_uniform.append(np.abs(I_uniform - I_exact))
    errors_q1.append(np.abs(I_q1 - I_exact))
    errors_q2.append(np.abs(I_q2 - I_exact))
    print(f"N = {N:6d} | Uniform I_est = {I_uniform:8.6f} | q1 I_est = {I_q1:8.6f} | q2 I_est = {I_q2:8.6f}")

plt.figure()
plt.loglog(N_values, errors_uniform, marker='o', label='Uniform')
plt.loglog(N_values, errors_q1, marker='s', label='Importance q1')
plt.loglog(N_values, errors_q2, marker='^', label='Importance q2')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Absolute Error in I')
plt.title('Error in MC Integration with Different Proposals')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# ========================================================================
# e) Box–Muller Transform for Generating Gaussian Distributed Samples
# ========================================================================
#
# Using the Box–Muller transform, we write a function to generate Gaussian 
# samples with mean µ and standard deviation σ. We then plot histograms for 
# various sample sizes.

def box_muller(mu, sigma, N):
    """
    Generate N samples from N(mu, sigma^2) using the Box-Muller transform.
    If N is odd, one extra sample is generated and trimmed.
    """
    N2 = N if N % 2 == 0 else N + 1
    U1 = np.random.uniform(0, 1, N2//2)
    U2 = np.random.uniform(0, 1, N2//2)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    samples = np.concatenate([Z0, Z1])
    samples = samples[:N]
    return mu + sigma * samples

N_list = [10, 100, 1000, 10000, 100000]
plt.figure(figsize=(12, 8))
for i, N in enumerate(N_list):
    samples = box_muller(0, 1, N)
    plt.subplot(3, 2, i+1)
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='C0', edgecolor='k')
    plt.title(f'Gaussian Samples Histogram, N = {N}')
    plt.xlabel('Value')
    plt.ylabel('Density')
plt.tight_layout()
plt.show()

# ========================================================================
# f) Monte Carlo Integration using Gaussian Proposal Distribution
# ========================================================================
#
# Now we perform the MC integration of 
#
#   I = ∫₀¹ f(u) du, with f(u)=1/sqrt(1 - e^2 u^2)
#
# using samples drawn from a Gaussian distribution N(µ, σ). Since f(u) is defined
# only on [0, 1], we set f(u)=0 outside that range. The estimator becomes:
#
#   I_est = (1/N)*Σ [ f(u_i) / φ(u_i) ],
#
# where φ(u) is the Gaussian PDF.
#
# We first take µ = 0 and σ = 1, and compare the error versus sample size.
# Then, for a fixed N = 10000, we explore various µ and σ.

def f_with_bounds(u, e):
    """
    Evaluate f(u)=1/sqrt(1-e^2 u^2) for u in [0,1] and return 0 for u outside [0,1].
    """
    u = np.array(u)
    vals = np.zeros_like(u)
    mask = (u >= 0) & (u <= 1)
    vals[mask] = integrand(u[mask], e)
    return vals

def gaussian_mc_integration(e, N, mu, sigma):
    """
    Estimate I = ∫₀¹ f(u) du using N samples drawn from N(mu, sigma^2).
    """
    samples = np.random.normal(mu, sigma, N)
    # Use the Gaussian PDF for the proposal.
    pdf_vals = norm.pdf(samples, loc=mu, scale=sigma)
    # f(u)=0 outside [0,1]
    f_vals = f_with_bounds(samples, e)
    # Avoid division by zero; note that pdf_vals > 0 almost everywhere.
    return np.mean(f_vals / pdf_vals)

# --- First: µ = 0, σ = 1 ---
errors_gaussian = []
for N in N_values:
    I_gauss_prop = gaussian_mc_integration(e_mc, N, mu=0, sigma=1)
    errors_gaussian.append(np.abs(I_gauss_prop - I_exact))
    print(f"N = {N:6d} | Gaussian Proposal (mu=0, sigma=1) I_est = {I_gauss_prop:8.6f}")

plt.figure()
plt.loglog(N_values, errors_gaussian, marker='o', color='C3')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Absolute Error in I')
plt.title('MC Integration Error with Gaussian Proposal (mu=0, sigma=1)')
plt.grid(True, which="both", ls="--")
plt.show()

# --- Next: Varying µ and σ with fixed N = 10000 ---
N_fixed = 10000
mu_vals = np.linspace(-0.5, 1.5, 50)
sigma_vals = np.linspace(0.1, 2, 50)
error_grid = np.zeros((len(sigma_vals), len(mu_vals)))  # rows: sigma, cols: mu

for i, sigma in enumerate(sigma_vals):
    for j, mu in enumerate(mu_vals):
        I_est = gaussian_mc_integration(e_mc, N_fixed, mu, sigma)
        error_grid[i, j] = np.abs(I_est - I_exact)

plt.figure(figsize=(8, 6))
plt.imshow(error_grid, extent=[mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]],
           origin='lower', aspect='auto', norm=LogNorm())
plt.colorbar(label='Absolute Error in I')
plt.xlabel('mu')
plt.ylabel('sigma')
plt.title('Error in MC Integration (Gaussian Proposal) for N=10000')
plt.show()