# 7A P1


import json
import math
import numpy as np
import matplotlib.pyplot as plt
from math import pi, e
from scipy.special import gamma

# ----------------------------
# Task 1a: Bayesian Inference on Biased Coins
# ----------------------------

def bayesian_inference_coin(dataset_filename, ax):
    """
    Loads coin-flip outcomes from dataset_filename.
    Iterates over the data in batches of 50, and at each step:
      - Computes the cumulative number of heads (M) out of tosses (N).
      - Defines the posterior as Beta(M+1, N-M+1) (with a uniform prior).
      - Computes expectation and variance from the Beta distribution.
      - Computes the Fisher variance (using p_hat*(1-p_hat)/N).
      - Plots the posterior distribution for p (the probability of heads).
    
    The function prints the batch number, cumulative counts, expectation,
    Beta variance, and Fisher variance.
    
    Parameters:
      dataset_filename (str): The JSON file with coin flip outcomes.
      ax (matplotlib.axes.Axes): Axes on which to plot the posterior curves.
    """
    # Load data from JSON file (each element is a Boolean: True for Head)
    with open(dataset_filename, 'r') as f:
        data = json.load(f)
    
    N_total = len(data)
    batch_size = 50
    num_batches = N_total // batch_size

    # Grid for p values (from 0 to 1)
    p_grid = np.linspace(0, 1, 500)

    cumulative_M = 0  # cumulative number of heads
    cumulative_N = 0  # cumulative number of tosses

    # Iterate over batches
    for i in range(num_batches):
        batch_data = data[i * batch_size:(i + 1) * batch_size]
        M_batch = sum(1 for outcome in batch_data if outcome)  # count True values
        cumulative_M += M_batch
        cumulative_N += batch_size

        # Posterior is Beta(alpha, beta) with alpha = cumulative heads + 1 and beta = cumulative tails + 1.
        alpha = cumulative_M + 1
        beta_param = cumulative_N - cumulative_M + 1

        # Evaluate the unnormalized Beta posterior on the grid:
        posterior = p_grid**(alpha - 1) * (1 - p_grid)**(beta_param - 1)
        # Normalize the posterior (so that its area is 1)
        posterior /= np.trapz(posterior, p_grid)

        # Calculate expectation and variance of the Beta distribution:
        p_expectation = alpha / (alpha + beta_param)
        p_variance = (alpha * beta_param) / (((alpha + beta_param)**2) * (alpha + beta_param + 1))

        # Compute the MLE p_hat and Fisher variance (for binomial: p*(1-p)/N)
        p_mle = cumulative_M / cumulative_N
        fisher_variance = p_mle * (1 - p_mle) / cumulative_N

        # Plot the posterior for this batch.
        ax.plot(p_grid, posterior, label=f'Batch {i+1} (N={cumulative_N})')
        # Optionally, mark the expectation with a vertical dashed line:
        ax.axvline(p_expectation, color='k', linestyle='--', alpha=0.3)

        print(f"{dataset_filename} - Batch {i+1}: N = {cumulative_N}, Heads = {cumulative_M}, "
              f"Expectation = {p_expectation:.4f}, Beta Variance = {p_variance:.4f}, "
              f"Fisher Variance = {fisher_variance:.4f}")

    ax.set_title(f'Posterior Distributions for {dataset_filename}')
    ax.set_xlabel('p (Probability of Heads)')
    ax.set_ylabel('Posterior Density')
    ax.legend(fontsize='small')

def run_bayesian_inference_all():
    """
    Runs the Bayesian inference on all three datasets and plots their posterior
    distributions side by side.
    """
    dataset_files = ['dataset_1.json', 'dataset_2.json', 'dataset_3.json']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, dataset_file in zip(axs, dataset_files):
        bayesian_inference_coin(dataset_file, ax)
    fig.tight_layout()
    plt.show()

# ----------------------------
# Task 1b: Stirling's Approximation
# ----------------------------

def stirling_approximation_plot():
    """
    Checks Stirling's approximation for factorial.
    
    1) For n from 1 to 10:
       - Computes the factorial using the Gamma function (Γ(n+1)).
       - Computes the Stirling's approximation: sqrt(2πn)*(n/e)^n.
       - Plots a scatter plot of the exact values (for integer n) and smooth curves
         for both the Gamma function and Stirling's approximation.
    
    2) Plots the difference between Stirling's approximation and the Gamma function.
    
    The plot is arranged as a 2-row, 1-column figure.
    """
    # Define a smooth range of n values from 1 to 10.
    x_cont = np.linspace(1, 10, 200)
    # Compute Stirling's approximation on the smooth grid:
    stirling_vals = np.sqrt(2 * np.pi * x_cont) * (x_cont / e)**x_cont
    # Compute Gamma(n+1) on the smooth grid:
    gamma_vals = np.array([gamma(x + 1) for x in x_cont])
    
    # For integer n values (for scatter plot)
    x_int = np.arange(1, 11)
    gamma_int = np.array([gamma(n + 1) for n in x_int])
    stirling_int = np.sqrt(2 * np.pi * x_int) * (x_int / e)**x_int

    # Create a figure with 2 subplots (2 rows, 1 column)
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Subplot 1: Plot the Gamma function and Stirling's approximation
    axs[0].scatter(x_int, gamma_int, color='red', label='Gamma(n+1) (scatter)')
    axs[0].plot(x_cont, gamma_vals, color='red', linestyle='--', label='Gamma(n+1) (smooth)')
    axs[0].plot(x_cont, stirling_vals, color='blue', linestyle='-', label="Stirling's Approximation")
    axs[0].set_title("Factorial: Gamma Function vs. Stirling's Approximation")
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('n!')
    axs[0].legend()

    # Subplot 2: Plot the difference between Stirling's approximation and Gamma(n+1)
    diff = stirling_vals - gamma_vals
    axs[1].plot(x_cont, diff, color='green')
    axs[1].set_title("Difference: Stirling's Approximation - Gamma(n+1)")
    axs[1].set_xlabel('n')
    axs[1].set_ylabel('Difference')

    fig.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    print("Running Task 1a: Bayesian Inference on Biased Coins")
    run_bayesian_inference_all()
    
    print("\nRunning Task 1b: Stirling's Approximation Check")
    stirling_approximation_plot()

# ----------------------------
# Task 1e: Bootsrapping
# ----------------------------

import json
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

def bootstrap_coin(dataset_filename, sample_sizes, n_bootstrap=100):
    """
    Loads coin-flip outcomes from dataset_filename and performs bootstrapping.
    
    For each sample size in sample_sizes, 100 bootstrap samples (with replacement)
    are drawn from the full dataset. For each bootstrap sample, the fraction of heads
    is computed. The function returns a dictionary mapping each sample size to an
    array of bootstrap estimates, and the overall coin-head probability (p_full)
    computed from the full dataset.
    
    Parameters:
      dataset_filename (str): The JSON file with coin flip outcomes.
      sample_sizes (list of int): List of sample sizes for bootstrapping.
      n_bootstrap (int): Number of bootstrap iterations for each sample size.
      
    Returns:
      bootstrap_results (dict): Keys are sample sizes; values are arrays of bootstrap estimates.
      p_full (float): Overall fraction of heads from the entire dataset.
    """
    with open(dataset_filename, 'r') as f:
        data = json.load(f)
    # Convert boolean outcomes to integers (True -> 1, False -> 0)
    data = np.array(data, dtype=int)
    N_total = len(data)
    p_full = np.sum(data) / N_total  # Full dataset estimate
    
    bootstrap_results = {}
    for size in sample_sizes:
        estimates = []
        for _ in range(n_bootstrap):
            # Sample with replacement from the dataset
            sample = np.random.choice(data, size=size, replace=True)
            p_boot = np.sum(sample) / size
            estimates.append(p_boot)
        bootstrap_results[size] = np.array(estimates)
    return bootstrap_results, p_full

def plot_bootstrap_histograms(bootstrap_results, p_full, dataset_name):
    """
    For a given dataset, plots a 3x3 grid of histograms of bootstrap estimates.
    Each subplot corresponds to one sample size from the provided bootstrap_results.
    
    The subplot title includes the sample size, bootstrap mean, and variance.
    A vertical dashed line indicates the full dataset estimate p_full.
    
    Parameters:
      bootstrap_results (dict): Keys are sample sizes; values are arrays of bootstrap estimates.
      p_full (float): The full dataset estimate of p.
      dataset_name (str): Name of the dataset (e.g., filename) for labeling.
      
    Returns:
      sample_sizes_sorted (list): Sorted list of sample sizes.
      exp_boot (list): Bootstrap mean for each sample size.
      var_boot (list): Bootstrap variance for each sample size.
    """
    sample_sizes_sorted = sorted(bootstrap_results.keys())
    exp_boot = []  # Bootstrap expectation for each sample size
    var_boot = []  # Bootstrap variance for each sample size
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs = axs.flatten()
    
    for i, size in enumerate(sample_sizes_sorted):
        estimates = bootstrap_results[size]
        mean_est = np.mean(estimates)
        var_est = np.var(estimates)
        exp_boot.append(mean_est)
        var_boot.append(var_est)
        
        ax = axs[i]
        ax.hist(estimates, bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(p_full, color='red', linestyle='--', label=f'p_full = {p_full:.3f}')
        ax.set_title(f'Sample Size = {size}\nBoot Mean = {mean_est:.3f}, Var = {var_est:.5f}')
        ax.set_xlabel('Bootstrap p')
        ax.set_ylabel('Count')
        ax.legend(fontsize='small')
    
    # Hide any extra subplots if sample_sizes_sorted has fewer than 9 entries
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    fig.suptitle(f'Bootstrap Histograms for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return sample_sizes_sorted, exp_boot, var_boot

def plot_bootstrap_summary(sample_sizes, exp_boot, var_boot, p_full, dataset_name):
    """
    Plots the bootstrap expectation values and variances versus sample size.
    
    The left subplot shows the bootstrap means along with a horizontal line for p_full.
    The right subplot shows the bootstrap variances and, for reference, the theoretical
    variance for a Bernoulli distribution, p_full*(1-p_full)/n.
    
    Parameters:
      sample_sizes (list): List of sample sizes used in bootstrapping.
      exp_boot (list): Bootstrap expectation (mean) for each sample size.
      var_boot (list): Bootstrap variance for each sample size.
      p_full (float): The full dataset estimate of p.
      dataset_name (str): Name of the dataset for labeling.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Expectation plot
    axs[0].plot(sample_sizes, exp_boot, marker='o', label='Bootstrap Mean')
    axs[0].axhline(p_full, color='red', linestyle='--', label=f'Full p = {p_full:.3f}')
    axs[0].set_xlabel('Sample Size')
    axs[0].set_ylabel('Expectation (p)')
    axs[0].set_title('Bootstrap Expectation vs. Sample Size')
    axs[0].legend()
    
    # Variance plot
    axs[1].plot(sample_sizes, var_boot, marker='o', label='Bootstrap Variance')
    # The theoretical variance for a Bernoulli sample mean is p*(1-p)/n.
    theoretical_var = [p_full * (1 - p_full) / n for n in sample_sizes]
    axs[1].plot(sample_sizes, theoretical_var, marker='x', linestyle='--',
                label='Theoretical Var (p*(1-p)/n)')
    axs[1].set_xlabel('Sample Size')
    axs[1].set_ylabel('Variance')
    axs[1].set_title('Bootstrap Variance vs. Sample Size')
    axs[1].legend()
    
    fig.suptitle(f'Bootstrap Summary for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_bootstrap_analysis():
    """
    Runs the bootstrapping analysis on three datasets.
    
    For each dataset, the function:
      1. Loads the dataset and computes the full dataset p.
      2. Performs bootstrapping for sample sizes of 5, 15, 40, 60, 90, 150, 210, 300, and 400.
      3. Plots a 3x3 histogram grid of bootstrap estimates.
      4. Plots summary curves for expectation and variance versus sample size.
    """
    dataset_files = ['dataset_1.json', 'dataset_2.json', 'dataset_3.json']
    sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]
    
    for dataset in dataset_files:
        print(f"\nProcessing {dataset}...")
        bootstrap_results, p_full = bootstrap_coin(dataset, sample_sizes, n_bootstrap=100)
        sizes, exp_boot, var_boot = plot_bootstrap_histograms(bootstrap_results, p_full, dataset)
        plot_bootstrap_summary(sizes, exp_boot, var_boot, p_full, dataset)
        print(f"Full dataset p for {dataset}: {p_full:.3f}")

if __name__ == '__main__':
    run_bootstrap_analysis()
