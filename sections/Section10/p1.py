#!/usr/bin/env python3
import os
import re
import subprocess

# ----------------------------
# Part a: Self-Consistent Field (SCF) Calculation
# ----------------------------

def update_input(content, new_ecutwfc, new_ecutrho):
    """
    Update the input file content with the new ecutwfc, ecutrho,
    and update the pseudopotential directory to '/root/Desktop/pseudo'.
    """
    # Use case-insensitive regex replacements
    content = re.sub(r'(ecutwfc\s*=\s*)(\S+)', r'\1{}'.format(new_ecutwfc), content, flags=re.IGNORECASE)
    content = re.sub(r'(ecutrho\s*=\s*)(\S+)', r'\1{}'.format(new_ecutrho), content, flags=re.IGNORECASE)
    content = re.sub(r"(pseudo_dir\s*=\s*)['\"][^'\"]+['\"]", r"\1'/root/Desktop/pseudo'", content, flags=re.IGNORECASE)
    return content

# Define the list of ecutwfc values and the fixed ecutrho value
ecutwfc_values = [10, 15, 20, 25, 30, 35, 40, 60]
ecutrho_value = 200.0

# Path to the original input file (adjust the path if necessary)
input_file = os.path.expanduser("~/Section129/sections/Section10/qe_file/qe_file/pw.graphene.scf.in")

# Check that the input file exists
if not os.path.isfile(input_file):
    print(f"Error: Input file '{input_file}' not found!")
    exit(1)

# Warn if the pseudopotential directory does not exist
pseudo_dir = "/root/Desktop/pseudo"
if not os.path.isdir(pseudo_dir):
    print(f"Warning: Pseudopotential directory '{pseudo_dir}' does not exist. Please ensure it exists and contains the necessary files.")

# Read the original input file
with open(input_file, "r") as f:
    original_content = f.read()

# Loop over each specified ecutwfc value, update the input file, and run the calculation
for ecut in ecutwfc_values:
    # Update the file content with the current cutoff values
    updated_content = update_input(original_content, ecut, ecutrho_value)
    
    # Create a new input filename that includes the ecutwfc value
    new_input_filename = f"pw_scF_ecutwfc_{ecut}.in"
    with open(new_input_filename, "w") as f:
        f.write(updated_content)
    
    # Define the corresponding output filename
    output_filename = f"pw_scF_ecutwfc_{ecut}.out"
    
    print(f"Running calculation for ecutwfc = {ecut}.")
    # Run the calculation using subprocess with file handles
    with open(new_input_filename, "r") as infile, open(output_filename, "w") as outfile:
        result = subprocess.run(["pw.x"], stdin=infile, stdout=outfile, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"Calculation for ecutwfc = {ecut} failed with return code {result.returncode}")
        print("Error output from pw.x:")
        print(result.stderr.decode())
        print(f"Please check the input file '{new_input_filename}' for correct formatting and verify that the pseudopotential files are available in '{pseudo_dir}'.")
    else:
        print(f"Calculation completed for ecutwfc = {ecut}. Output saved in '{output_filename}'.")
