# Task 1: Convex hull in 2D


# Want to give credit to "Brian Faure" on Youtube whose procedure I effectively copied, but learned more from this than just 
# yelling at ChatGPT to do it :)
# Part 1) Build Algorithsm
# Part a) Visualize
import matplotlib.pyplot as plt
import numpy as np
import warnings
# so it doesn't get mad at me
from random import randint
from math import atan2
warnings.filterwarnings("ignore", category=UserWarning)

def scatter_plot(coords, convex_hull=None):
    xs, ys=zip(*coords) # unzip the coordinate array into lists
    plt.scatter(xs,ys)

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

scatter_plot(coords)
# Part b)

# Graham Scan

graham_scan(coords, False)
print("2D Graham Scan Hull:", hull)  # Now hull is properly printed



# Part 2: Time complexity of a point cloud

# import numpy as np


# def gen_pt_cloud(n):
#     return np.random.uniform(0, 1, size=(n, 2))

# # Part a)
# n=100
# points = gen_pt_cloud(n)
# print(points)