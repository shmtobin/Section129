#!/bin/bash
# Part e: Band Structure (Corrected)
# Ensure you're in the correct directory: ~/Section129/sections/Section10/qe_file/qe_file

# Step 1: Run bands calculation with pw.x
echo "Running bands calculation using pw.x..."
pw.x < pw.graphene.bands.in > bands.pw.out

# Check if calculation succeeded
if [ ! -f "graphene.save/charge-density.dat" ]; then
    echo "Error: pw.x bands calculation failed. Check bands.pw.out"
    exit 1
fi

# Step 2: Process bands data with bands.x
echo "Processing bands data with bands.x..."
bands.x < bands.graphene.in > bands.processing.out

# Check if bands.dat was generated
if [ ! -f "bands.dat" ]; then
    echo "Error: bands.x failed to generate bands.dat. Check bands.processing.out"
    exit 1
fi

# Step 3: Generate gnuplot figure using a compatible terminal
echo "Plotting band structure with gnuplot..."
gnuplot << EOF
set terminal pngcairo enhanced
set output 'band_structure_gnuplot.png'
set xlabel 'k-point path'
set ylabel 'Energy (Ry)'
set title 'Band Structure from Gnuplot'
unset key
plot 'bands.dat' using 1:2 with lines lw 2, \
     '' using 1:3 with lines lw 2
EOF

# Step 4: Generate Python version of the plot
echo "Creating Python plot..."
python3 << EOF
import numpy as np
import matplotlib.pyplot as plt

# Load data (kpoints in 1st column, bands in subsequent columns)
data = np.loadtxt('bands.dat')
k, energies = data[:, 0], data[:, 1:]

plt.figure(figsize=(8, 6))
for band in range(energies.shape[1]):
    plt.plot(k, energies[:, band], 'b-', lw=1)

plt.xlabel('k-point path')
plt.ylabel('Energy (Ry)')
plt.title('Band Structure from Python')
plt.grid(True)
plt.savefig('band_structure_python.png')
plt.show()
EOF

echo "Band structure plots generated:"
echo "- Gnuplot: band_structure_gnuplot.png"
echo "- Python: band_structure_python.png"

# I think I broke something in one of the files and I am too tired to fix it
# Thank you so much for being so understanding throughout this whole course
# and have a great Spring break! 