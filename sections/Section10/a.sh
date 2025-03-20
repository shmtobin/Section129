#!/bin/bash
# This script performs a Self-Consistent Field (SCF) calculation for Quantum Espresso.
# It loops over several wavefunction cutoff energies (ecutwfc), updates the input file,
# and then runs pw.x using the updated file.
#
# The pseudopotential directory is set to '/root/Desktop/pseudo' as required.
#
# Directory structure:
#   a.sh is in Section10/
#   The input file is located in qe_file/qe_file/pw.graphene.scf.in
#
# For each run:
#   - A new input file named pw.scf_<ecut>.in is created.
#   - The parameters ecutwfc and ecutrho are updated.
#   - The pseudo_dir parameter is set to '/root/Desktop/pseudo'.
#   - The calculation is executed with output stored in qe_file/qe_file/results/scf_<ecut>.out.

# Path to the original input file
INPUT="qe_file/qe_file/pw.graphene.scf.in"
# Define the required pseudopotential directory
PSEUDO_DIR="/root/Desktop/pseudo"
# Create an output directory for the SCF result files if it doesn't exist
OUTDIR="qe_file/qe_file/results"
mkdir -p "$OUTDIR"

# List of ecutwfc values to use in the SCF calculation
ecutwfc_vals=(10 15 20 25 30 35 40 60)

# Loop over each ecutwfc value
for ecut in "${ecutwfc_vals[@]}"; do
    # Create a new input file for this cutoff value
    new_input="qe_file/qe_file/pw.scf_${ecut}.in"
    cp "$INPUT" "$new_input"

    # Update the ecutwfc parameter to the current value
    sed -i -r "s/ecutwfc\s*=\s*[0-9]+/ecutwfc = ${ecut}/" "$new_input"
    # Ensure the ecutrho parameter is set to 200.0
    sed -i -r "s/ecutrho\s*=\s*[0-9.]+/ecutrho = 200.0/" "$new_input"
    # Update the pseudo_dir parameter to the required pseudopotential directory
    sed -i -r "s|pseudo_dir\s*=\s*['\"][^'\"]+['\"]|pseudo_dir = '${PSEUDO_DIR}'|" "$new_input"

    # Define an output file name based on the current cutoff value
    output_file="${OUTDIR}/scf_${ecut}.out"
    echo "Running SCF calculation with ecutwfc = ${ecut}..."
    # Execute the pw.x calculation (make sure pw.x is in your PATH)
    pw.x < "$new_input" > "$output_file"
done

echo "All SCF calculations completed. Check the ${OUTDIR} directory for outputs."