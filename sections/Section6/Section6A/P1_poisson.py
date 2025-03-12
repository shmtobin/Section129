# This script computes the probability density function (PDF) for the distance to the nearest star
# assuming a uniform spatial distribution of stars with number density n.
#
# The probability that there are no stars within a sphere of radius R follows a Poisson distribution:
#   P_0 = exp(- (4/3) * pi * n * R^3)
#
# The probability of finding a star in the shell between R and R + dR is:
#   dP = P_0 * (number of stars in shell)
#
# The volume of the shell is:
#   dV = 4 * pi * R^2 * dR
#
# The probability density function is therefore:
#   P(R) = 4 * pi * n * R^2 * exp(- (4/3) * pi * n * R^3)

import numpy as np
import matplotlib.pyplot as plt

def poisson_nearest_star(R, n):
    """
    Computes the probability density function P(R) for the nearest star being at distance R,
    given a uniform star density n.
    
    Parameters:
    R (float or np.array): Distance from observation point.
    n (float): Number density of stars (stars per unit volume).
    
    Returns:
    float or np.array: Probability density P(R).
    """
    coeff = 4 * np.pi * n  # Constant coefficient
    exponent = np.exp(- (4/3) * np.pi * n * R**3)  # Exponential term
    return coeff * R**2 * exponent

# Define star density (example value in stars per cubic light-year)
n = 0.01  # Adjust as needed

# Define range of distances
R_values = np.linspace(0, 10, 1000)  # Distance range from 0 to 10 light-years

# Compute probability density function
P_values = poisson_nearest_star(R_values, n)

# Plot the probability distribution
plt.figure(figsize=(8, 5))
plt.plot(R_values, P_values, label='P(R)', color='b')
plt.xlabel('Distance to Nearest Star (R)')
plt.ylabel('Probability Density P(R)')
plt.title('Poisson Distribution of Nearest Star Distance')
plt.legend()
plt.grid()
plt.savefig("plots/poisson.png", dpi=300)
plt.show()
