#!/bin/bash
# Part d: KS-orbitals (wavefunctions)
# This script updates the pp.in file to obtain the KS-orbitals below the highest occupied
# valence band and the lowest unoccupied conduction band orbital at the Gamma point.
#
# Assumptions:
# - The original pp.in file is located in qe_file/qe_file/pp.in.
# - It contains lines with "kband(1) =" and "kband(2) =".
# - In this example, we update these to kband(1)=5 and kband(2)=6.
#   (Adjust these values as needed for your system.)
#
# The script creates a modified input file (pp_ks.in) and then runs pp.x to obtain the KS orbitals.

# Path to the original pp.in file
PP_INPUT="qe_file/qe_file/pp.in"
# Temporary modified input file for KS orbital calculation
PP_KS="qe_file/qe_file/pp_ks.in"

# Copy the original file to create the KS-specific input file
cp "$PP_INPUT" "$PP_KS"

# Update the band indices:
# Change the value for kband(1) to 5 and kband(2) to 6.
sed -i -r 's/(kband\(1\)\s*=\s*)[0-9]+/\1 5/' "$PP_KS"
sed -i -r 's/(kband\(2\)\s*=\s*)[0-9]+/\1 6/' "$PP_KS"

echo "Updated pp.in for KS orbitals with kband(1)=5 and kband(2)=6."

# Run pp.x with the modified input file (ensure pp.x is in your PATH)
echo "Running pp.x to obtain KS orbitals..."
pp.x < "$PP_KS" > qe_file/qe_file/pp_ks.out

echo "KS orbitals calculation completed."
echo "Check qe_file/qe_file/pp_ks.out for the output details."