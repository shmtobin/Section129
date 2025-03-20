#!/bin/bash
# Part c: Energy Minimum with Various Lattice Constants
# This script updates the lattice constant in the b-axis from 0.4 to 0.7 (10 uniformly spaced values),
# runs the SCF calculation for each value, extracts the total energy, and then plots energy vs lattice constant.
#
# Assumptions:
# - The original input file is located at qe_file/qe_file/pw.graphene.scf.in.
# - It contains a line with "lattice_b = <value>" that will be replaced.
# - The pseudopotential directory and other parameters have been set up appropriately in the input file.
#
# For each lattice constant:
#   - A new input file is created (named pw.lattice_<lattice>.in).
#   - The lattice_b parameter is updated.
#   - The SCF calculation is executed with pw.x.
#   - The output is stored in qe_file/qe_file/results_lattice/scf_lattice_<lattice>.out.
#   - The total energy is extracted and written to lattice_energy_data.dat.

# Path to the original input file
INPUT="qe_file/qe_file/pw.graphene.scf.in"
# Directory to store the lattice calculation output files
RESULTS_DIR="qe_file/qe_file/results_lattice"
mkdir -p "$RESULTS_DIR"

# Data file to store lattice constant and corresponding energy
ENERGY_FILE="lattice_energy_data.dat"
rm -f "$ENERGY_FILE"

# Define the number of lattice points
num_points=10
# Define the starting and ending lattice constant values for the b-axis
lattice_start=0.4
lattice_end=0.7

# Calculate the increment (step) value
step=$(awk -v start="$lattice_start" -v end="$lattice_end" -v n="$num_points" 'BEGIN { printf "%.5f", (end - start) / (n - 1) }')

# Loop over the number of lattice points
for ((i=0; i<num_points; i++)); do
    # Compute the current lattice constant using awk for floating point arithmetic
    lattice=$(awk -v start="$lattice_start" -v step="$step" -v i="$i" 'BEGIN { printf "%.4f", start + i*step }')
    
    # Create a new input file for this lattice constant
    new_input="qe_file/qe_file/pw.lattice_${lattice}.in"
    cp "$INPUT" "$new_input"
    
    # Update the lattice constant in the b-axis.
    # Assumes that the input file has a line containing "lattice_b = <value>".
    sed -i -r "s/(lattice_b\s*=\s*)[0-9.]+/\1${lattice}/" "$new_input"
    
    # Define an output file name for this run
    output_file="${RESULTS_DIR}/scf_lattice_${lattice}.out"
    
    echo "Running SCF calculation with lattice_b = ${lattice}..."
    # Run the SCF calculation (ensure pw.x is in your PATH)
    pw.x < "$new_input" > "$output_file"
    
    # Extract the total energy from the output file.
    # This command extracts the numeric value following "=" and before the unit "Ry".
    energy=$(grep -m1 "total energy" "$output_file" | sed -E 's/.*=\s*([-0-9\.]+).*/\1/')
    
    # Save the lattice constant and the energy into the data file
    echo "$lattice $energy" >> "$ENERGY_FILE"
done

echo "Lattice energy data has been saved in $ENERGY_FILE"

# Use Python to plot energy vs lattice constant and determine the optimal lattice size.
python3 <<EOF
import numpy as np
import matplotlib.pyplot as plt

# Load data: first column is lattice constant, second is total energy
data = np.loadtxt("$ENERGY_FILE")
if data.ndim == 1:
    lattice = data[0]
    energy = data[1]
else:
    lattice = data[:, 0]
    energy = data[:, 1]

plt.figure(figsize=(6,4))
plt.plot(lattice, energy, marker='o', linestyle='-')
plt.xlabel("Lattice constant b-axis")
plt.ylabel("Total Energy (Ry)")
plt.title("Energy vs Lattice Constant")
plt.grid(True)
plt.savefig("lattice_energy_convergence.png")
plt.show()

# Determine the optimal lattice constant (the one with minimum energy)
min_index = np.argmin(energy)
optimal_lattice = lattice[min_index]
min_energy = energy[min_index]
print(f"Optimal lattice constant: {optimal_lattice}, with energy: {min_energy}")
EOF

echo "Plot saved as lattice_energy_convergence.png"