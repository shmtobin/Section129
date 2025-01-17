import matplotlib.pyplot as plt
import numpy as np
import warnings
# so it doesn't get mad at me
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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

y = np.array(y)

# plotting stuff
plt.scatter(x, y, label="Mesh Data", color="b")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Data from mesh.dat")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("scatter_plot.png")
plt.close()
