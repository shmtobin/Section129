# Hello :|

# Task 1: Convex hull in 2D


# Want to give credit to "Brian Faure" on Youtube whose procedure I effectively copied, 
# and the good folks over at OpenAI whose chatbot assisted considerably.


# Part 1) Build Algorithsm

# Part a) Visualize
import matplotlib.pyplot as plt
import numpy as np
import warnings # so it doesn't get mad at me
from random import randint
from math import atan2
import time
warnings.filterwarnings("ignore")
import os

# basic visualization
def scatter_plot(coords, convex_hull=None):
    xs, ys=zip(*coords) # unzip the coordinate array into lists
    plt.scatter(xs,ys) # scatter plot

    if convex_hull!=None:
        for i in range(1, len(convex_hull)+1):
            if i==len(convex_hull): i=0
            c0=convex_hull[i-1]
            c1=convex_hull[i]
            plt.plot((c0[0], c1[0]), (c0[1],c1[1], 'r'))
    plt.show()
    plt.savefig('Plots/Basic_Visualization', bbox_inches='tight')

# determining polar angle for quick sort for graham scan
def polar_angle(p0, p1=None):
    if p1 is None: p1 = anchor
    x_span = p0[0] - p1[0]
    y_span = p0[1] - p1[1]
    return atan2(y_span, x_span)

# calculates distance between points for quicksort for graham scan
def distance(p0, p1=None):
    if p1 is None: p1 = anchor
    x_span = p0[0] - p1[0]
    y_span = p0[1] - p1[1]
    return x_span**2 + y_span**2

# calculates determinant for all algorithms
def det(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

# sorts the points for graham scan algorithm
def quicksort(points):
    if len(points) <= 1:
        return points
    smaller, equal, larger = [], [], []
    pivot = polar_angle(points[randint(0, len(points) - 1)])
    for pt in points:
        angle = polar_angle(pt)
        if angle < pivot:
            smaller.append(pt)
        elif angle == pivot:
            equal.append(pt)
        else:
            larger.append(pt)
    return quicksort(smaller) + sorted(equal, key=distance) + quicksort(larger)

# 2D graham scan
def graham_scan(points, show_progress=False):
    global anchor

    # convert points to tuples to ensure comparisons work
    points = [tuple(p) for p in points]

    # find the anchor point (lowest y; if tie, leftmost x)
    anchor = min(points, key=lambda p: (p[1], p[0]))

    # sort points by polar angle with respect to the anchor
    sorted_pts = quicksort(points)

    # remove the anchor from the sorted list
    sorted_pts.remove(anchor)

    # initialize the hull with the anchor and the first sorted point
    hull = [anchor, sorted_pts[0]]
    for s in sorted_pts[1:]:
        # ensure hull remains convex
        while len(hull) >= 2 and det(hull[-2], hull[-1], s) <= 0:
            hull.pop()
        hull.append(s)

    if show_progress:
        scatter_plot(points, hull)  # assumes a scatter plot function is defined elsewhere

    return hull

# jarvis march algorithm
def jarvis(points):
    n=len(points)

    hull=[]

    leftmost_idx=np.argmin(points[:, 0])
    current_idx= leftmost_idx

    while True:
        hull.append(tuple(points[current_idx]))
        next_idx = (current_idx + 1) % n
        
        for i in range(n):
            if det(points[current_idx], points[next_idx], points[i]) > 0:
                next_idx = i

        current_idx = next_idx

        if current_idx == leftmost_idx:
            break

    return hull

# quickhull algorithm
def quickhull(points):
    # convert points to a list of tuples for consistency
    points = [tuple(p) for p in points]
    if len(points) < 3:
        return points

    # helper: which side of line from a to b is point p on?
    def side(a, b, p):
        return det(a, b, p)

    # recursively add hull points between p and q
    def add_hull(pt_set, p, q):
        index = None
        max_distance = 0
        # find the point with maximum distance from line p->q
        for i, pt in enumerate(pt_set):
            d = abs(det(p, q, pt))
            if d > max_distance:
                max_distance = d
                index = i
        if index is None:
            # no point is outside; p->q is a hull edge
            return []
        farthest = pt_set[index]
        # split set into two subsets: points to the left of (p, farthest)
        # and points to the left of (farthest, q)
        left_set = [pt for pt in pt_set if side(p, farthest, pt) > 0]
        right_set = [pt for pt in pt_set if side(farthest, q, pt) > 0]
        # recursively find hull points on these segments
        return add_hull(left_set, p, farthest) + [farthest] + add_hull(right_set, farthest, q)

    # find the leftmost and rightmost points (extremes)
    leftmost = min(points, key=lambda p: p[0])
    rightmost = max(points, key=lambda p: p[0])

    # partition the points into two subsets: above and below the line
    above = [pt for pt in points if side(leftmost, rightmost, pt) > 0]
    below = [pt for pt in points if side(rightmost, leftmost, pt) > 0]

    # build the hull (in order) by combining the extreme points and the recursively found points
    upper_hull = add_hull(above, leftmost, rightmost)
    lower_hull = add_hull(below, rightmost, leftmost)

    # The full hull is the leftmost point, then the points on the upper hull,
    # then the rightmost point, and finally the points on the lower hull.
    return [leftmost] + upper_hull + [rightmost] + lower_hull

def monotone_chain(points):
    # sort the points lexicographically (first by x-coordinate, then by y-coordinate)
    points = sorted(map(tuple, points), key=lambda p: (p[0], p[1]))

    # build the lower hullx
    lower_hull = []
    for point in points:
        while len(lower_hull) >= 2 and det(lower_hull[-2], lower_hull[-1], point) <= 0:
            lower_hull.pop()
        lower_hull.append(point)

    # build the upper hull
    upper_hull = []
    for point in reversed(points):
        while len(upper_hull) >= 2 and det(upper_hull[-2], upper_hull[-1], point) <= 0:
            upper_hull.pop()
        upper_hull.append(point)

    # combine lower and upper hull, removing the last point of each half because it's repeated
    return lower_hull[:-1] + upper_hull[:-1]

# plot scatter plot including hull algorithms
def scatter_plot_with_hulls(points, hulls):
    xs, ys = zip(*points)
    plt.scatter(xs, ys, label="Points", color="black")

    colors = ['red', 'green', 'blue', 'purple']
    labels = ['Graham Scan', 'Jarvis March', 'Quickhull', 'Monotone Chain']
    for hull, color, label in zip(hulls, colors, labels):
        hull_xs, hull_ys = zip(*hull + [hull[0]]) 
        plt.plot(hull_xs, hull_ys, color=color, label=label)

    plt.legend()
    plt.show()
    plt.savefig('Plots/Visualization_with_hulls', bbox_inches='tight')

# generates n uniformly random coordinates between 
# an upper and lower bound for a given seed
def gen_pt_cloud(n, lower, upper, seed):
    np.random.seed(seed)
    return np.random.uniform(lower, upper, size=(n, 2))

# generates n gaussian coordinates for a given seed
def gauss_pt_cloud(n, seed):
    np.random.seed(seed)
    return np.random.normal(loc=0, scale=1, size=(n, 2))

# measures the time it takes for each of hte algorithms
def measure_execution_time(points, algorithms):
    times = {}
    for name, algorithm in algorithms.items():
        start_time = time.perf_counter()  # Use high-resolution timer
        try:
            algorithm(points)  # Run the algorithm on the point cloud
        except Exception as e:
            print(f"Error with {name}: {e}")
            times[name] = np.nan  # In case of error, record NaN
            continue
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        if elapsed_time < 0:
            print(f"Warning: Negative execution time for {name}!")
            elapsed_time = 0  # Prevent negative times, set to 0
        times[name] = elapsed_time
    return times

# reading in the file
with open("mesh.dat", "r") as file:
    data = file.readlines()

# opening arrays
x = []
y = []

for line in data:
    if line.strip() == "" or line.startswith('X'):
        continue
    
    columns = line.split()
    
    try:
        x.append(float(columns[0])) 
        y.append(float(columns[1]))  
    except ValueError:
        continue

coords = np.array(list(zip(x, y)))
# visualization for a)
scatter_plot(coords)
# Part b)
# Applies all algorithms defined above
hulls = [graham_scan(coords, False), jarvis(coords), quickhull(coords), monotone_chain(coords)]
print("2D Graham Scan Hull:", hulls[0]) 
print("Jarvis March Hull:", hulls[1])
print("Quickhull hull:", hulls[2])
print("Monotone chain hull:", hulls[3])

# Part c)
# visualization for c, applies scatter plot
# with hull function defined above
scatter_plot_with_hulls(coords, hulls)


# Part 2: Time complexity of a point cloud

# Part a)
n=10
points = gauss_pt_cloud(n, 42)
print(points)

# Part b)
n_values = [10, 50, 100, 200, 400, 800, 1000]
num_trials = 5 
algorithms = {
    "Graham Scan": graham_scan,
    "Jarvis March": jarvis,
    "Quickhull": quickhull,
    "Monotone Chain": monotone_chain
}

# Plots for a)
# Prepare a dictionary for storing runtime results for each algorithm
results = {name: [] for name in algorithms.keys()}

# Loop through different point cloud sizes
for n in n_values:
    trial_times = {name: [] for name in algorithms.keys()}
    
# Testing for 0 to 1, varying n
    for trial in range(num_trials):
        # Set a different random seed for each trial
        seed_value = 42 + trial  # Varying seed for each trial
        points = gen_pt_cloud(n, 0, 1, seed_value)  # Generate points with the current seed
        
        # Measure execution time for each algorithm
        times = measure_execution_time(points, algorithms)
        
        # Store the results for this trial
        for name, time_taken in times.items():
            trial_times[name].append(time_taken)

    # Average the results for each algorithm across trials
    for name, times in trial_times.items():
        avg_time = np.mean(times)
        results[name].append(avg_time)  # Store averaged time for this n

plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(n_values, times, label=name, marker='o')
plt.xlabel("Number of Points (n)")
plt.ylabel("Average Execution Time (seconds)")
plt.title("Average Execution Time vs. Input Size for Convex Hull Algorithms Coords [0, 1]")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Plots/Algorithm_times', bbox_inches='tight')

with open("Analysis_txts/analysis_2b.txt", "w") as file:
    file.write("Time complexity of the four functions:\n")
    file.write("\n")
    file.write("For the 4 tested hull calculation methods, \n")
    file.write("Jarvis march appeared to have the worst time complexity  \n")
    file.write("which seems decently reasonable. Ideally Jarvis walk is  \n")
    file.write("O(nh), where h is the number of points on the hull, \n")
    file.write("which in theory could be as bad as O(n^2), which is worse \n")
    file.write("than Graham scan at O(n log(n)), but this would mean that  \n")
    file.write("h grows considerably faster than log(n), which doesn't sound  \n")
    file.write("very reasonable. Quickhull was next most efficient, which  \n")
    file.write("makes sense because Quickhull has time complexity O(nlog(n))  \n")
    file.write("The most efficient algorithm was Monotone Chain  \n")
    file.write("which is supposed to also be O(n log(n)), like Quickhull \n")
    file.write("so it makes sense their time complexity appears comparable.  \n")
# # c)
# Prepare a dictionary for storing runtime results for each algorithm
results_1 = {name: [] for name in algorithms.keys()}
results_2 = {name: [] for name in algorithms.keys()}

# Loop through different point cloud sizes for 0 to 1 range
for n in n_values:
    trial_times = {name: [] for name in algorithms.keys()}
    for trial in range(num_trials):
        seed_value = 42 + trial  # Varying seed for each trial
        points = gen_pt_cloud(n, 0, 1, seed_value)  # Generate points in range [0, 1]
        times = measure_execution_time(points, algorithms)
        for name, time_taken in times.items():
            trial_times[name].append(time_taken)
    for name, times in trial_times.items():
        avg_time = np.mean(times)
        results_1[name].append(avg_time)

# Loop through different point cloud sizes for -5 to 5 range
for n in n_values:
    trial_times = {name: [] for name in algorithms.keys()}
    for trial in range(num_trials):
        seed_value = 42 + trial  # Varying seed for each trial
        points = gen_pt_cloud(n, -5, 5, seed_value)  # Generate points in range [-5, 5]
        times = measure_execution_time(points, algorithms)
        for name, time_taken in times.items():
            trial_times[name].append(time_taken)
    for name, times in trial_times.items():
        avg_time = np.mean(times)
        results_2[name].append(avg_time)

# Plot both results on the same plot
plt.figure(figsize=(12, 8))
for name, times in results_1.items():
    plt.plot(n_values, times, label=f"{name} (0 to 1)", marker='o', alpha=0.8)
for name, times in results_2.items():
    plt.plot(n_values, times, label=f"{name} (-5 to 5)", marker='o', linestyle='--', alpha=0.6)
    
plt.xlabel("Number of Points (n)")
plt.ylabel("Average Execution Time (seconds)")
plt.title("Average Execution Time vs. Input Size for Convex Hull Algorithms Coords [0, 1] vs [-5, 5]")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Plots/Variance_algorithm_times', bbox_inches='tight')

# Prepare a dictionary for storing runtime results for each algorithm
results_uniform = {name: [] for name in algorithms.keys()}
results_gaussian = {name: [] for name in algorithms.keys()}

# Loop through different point cloud sizes for uniform sampling (0 to 1 range)
for n in n_values:
    trial_times = {name: [] for name in algorithms.keys()}
    for trial in range(num_trials):
        seed_value = 42 + trial  # Varying seed for each trial
        points = gen_pt_cloud(n, 0, 1, seed_value)  # Uniform sampling
        times = measure_execution_time(points, algorithms)
        for name, time_taken in times.items():
            trial_times[name].append(time_taken)
    for name, times in trial_times.items():
        avg_time = np.mean(times)
        results_uniform[name].append(avg_time)

# Loop through different point cloud sizes for Gaussian sampling
for n in n_values:
    trial_times = {name: [] for name in algorithms.keys()}
    for trial in range(num_trials):
        seed_value = 42 + trial  # Varying seed for each trial
        points = gauss_pt_cloud(n, seed_value)  # Gaussian sampling
        times = measure_execution_time(points, algorithms)
        for name, time_taken in times.items():
            trial_times[name].append(time_taken)
    for name, times in trial_times.items():
        avg_time = np.mean(times)
        results_gaussian[name].append(avg_time)

# Plot both results on the same plot
plt.figure(figsize=(12, 8))
for name, times in results_uniform.items():
    plt.plot(n_values, times, label=f"{name} (Uniform)", marker='o', alpha=0.8)
for name, times in results_gaussian.items():
    plt.plot(n_values, times, label=f"{name} (Gaussian)", marker='o', linestyle='--', alpha=0.6)
    
plt.xlabel("Number of Points (n)")
plt.ylabel("Average Execution Time (seconds)")
plt.title("Gaussian Sampling Average Execution Time vs. Input Size for Convex Hull Algorithms")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Plots/Gaussian_Algorithm_times_P1_P2b', bbox_inches='tight')

with open("Analysis_txts/analysis_2c.txt", "w") as file:
    file.write("Effect of increased variance on methods:\n")
    file.write("\n")
    file.write("For the 4 tested hull calculation methods, \n")
    file.write("all methods were slightly less efficient for \n")
    file.write("higher variance, but this effect plateaus in high n \n")
    file.write("which makes sense. \n")

# d)
num_trials = 100
n=50
results_gaussian = {name: [] for name in algorithms.keys()}

# Generate data and store runtime distributions
for trial in range(num_trials):
    seed_value = 42 + trial  # Varying seed for each trial
    points = gauss_pt_cloud(n, seed_value)  # Gaussian sampling
    times = measure_execution_time(points, algorithms)
    for name, time_taken in times.items():
        results_gaussian[name].append(time_taken)

# Plot histograms
plt.figure(figsize=(16, 12))
for i, (name, times) in enumerate(results_gaussian.items()):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid of subplots
    plt.hist(times, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(f"Runtime Distribution: {name}")
    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Plots/Algorithm_Runtime_Distributions.png', bbox_inches='tight')
plt.show()

with open("Analysis_txts/runtime_analysis.txt", "w") as f:
    for name, times in results_gaussian.items():
        best_time = np.min(times)
        worst_time = np.max(times)
        approx_distribution = "normal" 
        f.write(
            f"Algorithm: {name}\n"
            f"  Best Runtime: {best_time:.4f} seconds\n"
            f"  Worst Runtime: {worst_time:.4f} seconds\n"
            f"  Approximate Distribution: {approx_distribution}\n\n"
        )

with open("Analysis_txts/analysis_2d.txt", "w") as file:
    file.write("Distribution of times for each algorithm:\n")
    file.write("\n")
    file.write("For the 4 tested hull calculation methods, \n")
    file.write("Monotone chain was by far the fastest on average\n")
    file.write("with a high peak dominating the vast majority of runs\n")
    file.write("before 0.003 seconds, impressively.\n")
    file.write("Quickhull is the next quickest and then \n")
    file.write("Jarvis march and Graham scan appear roughly\n")