#!/bin/bash
# Part b: Energy Convergence
# This script extracts the total energy (ground state energy) from each SCF output file
# and plots energy versus ecutwfc using Python.
#
# Assumptions:
# - The SCF output files are located in qe_file/qe_file/results and are named as scf_<ecut>.out.
# - Each output file contains a line similar to:
#       "!    total energy              =   -XXX.XXXX Ry"
# - We extract only the numeric value.
#
# The script creates an energy_data.dat file with two columns: ecutwfc and total energy,
# then calls a Python snippet to plot the data and save it as energy_convergence.png.

energy_file="energy_data.dat"
rm -f "$energy_file"

results_dir="qe_file/qe_file/results"
# List of ecutwfc values that were used in part a
ecutwfc_vals=(10 15 20 25 30 35 40 60)

# Loop over each ecutwfc value to extract energy from the corresponding output file
for ecut in "${ecutwfc_vals[@]}"; do
    output_file="${results_dir}/scf_${ecut}.out"
    if [[ -f "$output_file" ]]; then
        # Extract only the first matching line with "total energy" and then pick field 5
        energy=$(grep -m1 "total energy" "$output_file" | awk '{print $5}')
        # Remove any non-numeric characters (e.g., "Ry")
        energy=$(echo "$energy" | sed 's/[^0-9\.\-]//g')
        echo "$ecut $energy" >> "$energy_file"
    else
        echo "Warning: File $output_file not found."
    fi
done

echo "Energy data has been extracted to $energy_file"

# Use Python to plot the energy convergence curve.
python3 <<'EOF'
import numpy as np
import matplotlib.pyplot as plt

# Load the energy data from the file
data = np.loadtxt("energy_data.dat")
# If only one line of data exists, np.loadtxt returns a 1D array.
if data.ndim == 1:
    ecut = data[0]
    energy = data[1]
else:
    ecut, energy = data[:,0], data[:,1]

plt.figure()
plt.plot(ecut, energy, marker='o', linestyle='-')
plt.xlabel("Wavefunction Cutoff Energy (ecutwfc) [Ry]")
plt.ylabel("Total Energy [Ry]")
plt.title("Energy Convergence with ecutwfc")
plt.grid(True)
plt.savefig("energy_convergence.png")
plt.show()
EOF

echo "Plot saved as energy_convergence.png"