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

def polar_angle(p0,p1=None):
    if p1==None: p1=anchor
    x_span=p0[0]-p0[0]
    y_span=p0[1]-p1[1]
    return atan2(y_span, x_span)

def distance(p0, p1=None):
    if p1 is None: p1 = anchor
    x_span=p0[0]-p0[0]
    y_span=p0[1]-p1[1]
    return y_span**2 + x_span**2

def det(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) \
            - (p2[1]-p1[1])*(p3[0]-p1[0])

def quicksort(a):
    if len(a)<=1: return a
    smaller, equal, larger = [], [], []
    piv_ang = polar_angle(a[randint(0, len(a)-1)])
    for pt in a:
        pt_ang = polar_angle(pt)
        if pt_ang<piv_ang: smaller.append(pt)
        elif pt_ang==piv_ang: equal.append(pt)
        else: larger.append(pt)
    return quicksort(smaller) \
            +sorted(equal, key=distance) \
            + quicksort(larger)

def graham_scan(points, show_progress=False):
    global anchor

    min_idx = None
    for i, (x, y) in enumerate(points):
        if min_idx is None or y < points[min_idx][1]:
            min_idx = i
        if y == points[min_idx][1] and x < points[min_idx][0]:
            min_idx = i
    anchor = tuple(points[min_idx])  # Convert NumPy array to tuple
    sorted_pts = quicksort(points)
    
    # Convert all elements to tuples to avoid NumPy-related issues
    sorted_pts = [tuple(pt) for pt in sorted_pts]

    del sorted_pts[sorted_pts.index(anchor)]  # This now works

    hull = [anchor, sorted_pts[0]]
    for s in sorted_pts[1:]:
        while len(hull) >= 2 and det(hull[-2], hull[-1], s) <= 0:
            del hull[-1]  # Backtrace
        hull.append(s)

    if show_progress:
        scatter_plot(points, hull)
    
    return hull

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

def quickhull(points):
    def find_farthest_point(p1, p2, points):
        farthest_point = None
        max_distance = -1
        for point in points:
            # Calculate distance of 'point' from the line (p1, p2)
            distance = abs(det(p1, p2, point))
            if distance > max_distance:
                max_distance = distance
                farthest_point = point
        return farthest_point

    def find_hull(subset, p1, p2, hull):
        if not subset:
            return
        
        farthest_point = find_farthest_point(p1, p2, subset)
        hull.append(tuple(farthest_point))
        
        # Partition the remaining points into two subsets
        left_of_p1_far = [p for p in subset if det(p1, farthest_point, p) > 0]
        left_of_far_p2 = [p for p in subset if det(farthest_point, p2, p) > 0]
        
        # Recursively find hull points on the left of the new lines
        find_hull(left_of_p1_far, p1, farthest_point, hull)
        find_hull(left_of_far_p2, farthest_point, p2, hull)

    # Find the leftmost and rightmost points (anchor points)
    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]
    hull = [tuple(leftmost), tuple(rightmost)]
    
    # Split points into two subsets
    left_of_line = [p for p in points if det(leftmost, rightmost, p) > 0]
    right_of_line = [p for p in points if det(rightmost, leftmost, p) > 0]

    # Recursively find hull points on each side of the line
    find_hull(left_of_line, leftmost, rightmost, hull)
    find_hull(right_of_line, rightmost, leftmost, hull)
    
    # Return hull in counterclockwise order
    return sorted(hull, key=lambda p: (atan2(p[1] - leftmost[1], p[0] - leftmost[0])))

def monotone_chain(points):
    # Sort the points lexicographically (first by x-coordinate, then by y-coordinate)
    points = sorted(map(tuple, points), key=lambda p: (p[0], p[1]))

    # Build the lower hull
    lower_hull = []
    for point in points:
        while len(lower_hull) >= 2 and det(lower_hull[-2], lower_hull[-1], point) <= 0:
            lower_hull.pop()
        lower_hull.append(point)

    # Build the upper hull
    upper_hull = []
    for point in reversed(points):
        while len(upper_hull) >= 2 and det(upper_hull[-2], upper_hull[-1], point) <= 0:
            upper_hull.pop()
        upper_hull.append(point)

    # Combine lower and upper hull, removing the last point of each half because it's repeated
    return lower_hull[:-1] + upper_hull[:-1]

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

def gen_pt_cloud(n, seed):
    np.random.seed(seed)
    return np.random.uniform(0, 1, size=(n, 2))

def measure_execution_time(points, algorithms):
    times = {}
    for name, algo in algorithms.items():
        start_time = time.time()
        algo(points)
        end_time = time.time()
        times[name] = end_time - start_time
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
hull = graham_scan(coords, False)  # Call the function and store the result

# scatter_plot(coords)
# Part b)

# Graham Scan
hulls = [graham_scan(coords, False), jarvis(coords), quickhull(coords), monotone_chain(coords)]
print("2D Graham Scan Hull:", hulls[0]) 
print("Jarvis March Hull:", hulls[1])
print("Quickhull hull:", hulls[2])
print("Monotone chain hull:", hulls[3])

# Part c)
scatter_plot_with_hulls(coords, hulls)


# Part 2: Time complexity of a point cloud

# Part a)
# n=10
# points = gen_pt_cloud(n)
# print(points)

# Part b)
n_values = [10, 50, 100, 200, 400, 800, 1000]
num_trials = 5  # Number of trials per n value
algorithms = {
    "Graham Scan": graham_scan,
    "Jarvis March": jarvis,
    "Quickhull": quickhull,
    "Monotone Chain": monotone_chain
}

# Prepare a dictionary for storing runtime results for each algorithm
results = {name: [] for name in algorithms.keys()}

# Loop through different point cloud sizes
for n in n_values:
    trial_times = {name: [] for name in algorithms.keys()}
    
    # Run multiple trials for each n
    for trial in range(num_trials):
        # Set a different random seed for each trial
        seed_value = 42 + trial  # Varying seed for each trial
        points = gen_pt_cloud(n, seed_value)  # Generate points with the current seed
        
        # Measure execution time for each algorithm
        times = measure_execution_time(points, algorithms)
        
        # Store the results for this trial
        for name, time_taken in times.items():
            trial_times[name].append(time_taken)

    # Average the results for each algorithm across trials
    for name, times in trial_times.items():
        avg_time = np.mean(times)
        results[name].append(avg_time)  # Store averaged time for this n

# Plot the results
plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(n_values, times, label=name, marker='o')
plt.xlabel("Number of Points (n)")
plt.ylabel("Average Execution Time (seconds)")
plt.title("Average Execution Time vs. Input Size fo r Convex Hull Algorithms")
plt.legend()
plt.grid(True)
plt.show()

# with open("convex_hull_time_comparison.txt", "w") as file:
#     file.write("Conclusion:\n")
#     file.write("For the 4 tested hull calculation methods, \n")
#     file.write("\n")