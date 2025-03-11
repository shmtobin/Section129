# ----------------------------
# Part a
# ----------------------------
def simulate_tm(binary1, binary2):
    """
    Simulate a Turing machine that performs binary multiplication.
    
    The tape configuration is built as:
      [left blanks] + [multiplicand] + " #" + [multiplier] + " $" + [accumulator] + [right blanks]
    
    Initially the multiplier appears in full; as each bit (from rightmost to leftmost)
    is processed, if the bit is '1' the multiplicand (appropriately shifted) is added to the
    accumulator, and that multiplier bit is replaced by an 'X'. Each tape configuration is
    stored as a string.
    
    The accumulator is padded to a fixed width equal to len(binary1)+len(binary2), which is the
    worst-case product length.
    """
    left_blanks = "B" * 5
    right_blanks = "B" * 5
    width = len(binary1) + len(binary2)  # fixed width for the accumulator

    def int_to_bin_str(n):
        return format(n, 'b').zfill(width)

    tape_states = []
    acc = 0

    # Initial tape configuration (state q0)
    tape = left_blanks + binary1 + " #" + binary2 + " $" + int_to_bin_str(acc) + right_blanks
    tape_states.append(tape)

    # Convert multiplier to a mutable list of characters.
    multiplier_list = list(binary2)

    # Process each bit of the multiplier from rightmost (least significant) to leftmost.
    for i in range(len(binary2)):
        idx = len(binary2) - 1 - i  # index of the current bit (right-to-left)
        if multiplier_list[idx] == '1':
            add_val = int(binary1, 2) << i  # shift multiplicand by i positions
            acc += add_val
        # Mark the processed bit as 'X'
        multiplier_list[idx] = 'X'
        # Update tape configuration with the new multiplier state and accumulator value.
        tape = left_blanks + binary1 + " #" + ''.join(multiplier_list) + " $" + int_to_bin_str(acc) + right_blanks
        tape_states.append(tape)

    return tape_states

def write_tape_to_file(binary1, binary2, tape_states):
    """
    Write each tape configuration (with an artificial state label) to a .dat file.
    The filename is chosen to reflect the multiplication, e.g.,
    TM_101001010111_x_101000101.dat.
    """
    filename = f"TM_{binary1}_x_{binary2}.dat"
    with open(filename, "w") as f:
        for i, state in enumerate(tape_states):
            f.write(f"q{i}: {state}\n")
    print(f"File '{filename}' written with {len(tape_states)} tape configurations.")

# ----------------------------
# Part b
# ----------------------------
if __name__ == "__main__":
    # Generate tape file for the multiplication: 101001010111 × 101000101
    binary1 = "101001010111"
    binary2 = "101000101"
    tape_states = simulate_tm(binary1, binary2)
    write_tape_to_file(binary1, binary2, tape_states)

    # Generate tape file for the multiplication: 101111 × 101001
    binary1 = "101111"
    binary2 = "101001"
    tape_states = simulate_tm(binary1, binary2)
    write_tape_to_file(binary1, binary2, tape_states)

# ----------------------------
# Part c
# ----------------------------
def write_optimized_tm_diagram():
    """
    Write a text file that contains a simplified state transition diagram for an optimized
    Turing machine for binary multiplication.
    
    In the optimized version, we assume the following (illustrative) state structure:
      - q0: initial state,
      - q1: reading and processing the multiplicand,
      - q2: detecting the separator '#' and transitioning to multiplier processing,
      - q3: reading a multiplier digit and deciding whether to add,
      - q4: performing the (shifted) addition when a '1' is read,
      - q5: marking the multiplier digit as processed,
      - q6: moving the tape head back to prepare for the next operation,
      - q7: final adjustments before halting,
      - qhalt: halting state.
      
    This optimized TM merges several redundant states found in the original (non‐optimized)
    design. Thus, excluding qhalt there are 8 states; including qhalt the TM has 9 states.
    """
    diagram_text = """
Optimized Turing Machine State Transition Diagram:

q0: Initial state. Read left blanks and move right to start processing.
q1: Process the multiplicand digits.
q2: On reading the '#' separator, switch to multiplier processing.
q3: Read a multiplier digit. If the digit is '1', transition to q4; if '0', skip to q5.
q4: Add the multiplicand (shifted appropriately) to the accumulator.
q5: Mark the processed multiplier digit (change it to a marker, e.g. 'X') and prepare for next digit.
q6: Move the tape head back to the beginning (or appropriate position) for the next cycle.
q7: Final adjustments before termination.
qhalt: Halt state.

Total number of states (excluding qhalt): 8
Including the halt state, the optimized Turing Machine has 9 states.
    """
    with open("optimized_TM.txt", "w") as f:
        f.write(diagram_text.strip())
    print("Optimized TM diagram written to 'optimized_TM.txt' (9 states total).")

if __name__ == "__main__":
    write_optimized_tm_diagram()

# ----------------------------
# Part d
# ----------------------------
import math

def compute_tm_complexity(L_a, L_b):
    """
    Compute the simulated Turing machine's computation complexity based on binary multiplication.
    
    We model the number of steps n as follows:
      n = 1 + L_b + (L_a + L_b) * (# of 1's in the multiplier)
      
    Explanation:
      - 1 step for the initial configuration.
      - L_b steps: one step per multiplier digit (processing/marking).
      - For each '1' in the multiplier, an addition is performed that takes (L_a + L_b) steps.
      
    Since the multiplicand does not affect the number of steps (it only shifts the addition cost),
    the variability in n comes solely from the multiplier.
    
    For a given pair of lengths L_a and L_b, all possible multiplier bit-strings of length L_b are
    considered. There are 2^(L_a + L_b) total (multiplicand, multiplier) pairs, but n depends only
    on the multiplier.
    
    Returns:
        best: Minimum steps (n_min) over all multipliers.
        worst: Maximum steps (n_max) over all multipliers.
        average: Average steps ⟨n⟩.
        histogram: Dictionary mapping computed step count n to its frequency.
    """
    total_pairs = 2 ** (L_a + L_b)
    histogram = {}
    total_steps = 0
    # For a multiplier of length L_b, let k be the number of 1's.
    # There are comb(L_b, k) multipliers with k ones.
    for k in range(L_b + 1):
        n = 1 + L_b + (L_a + L_b) * k
        frequency = math.comb(L_b, k) * (2 ** L_a)  # multiplicand choices contribute equally.
        histogram[n] = frequency
        total_steps += n * frequency
    best = min(histogram.keys())
    worst = max(histogram.keys())
    average = total_steps / total_pairs
    return best, worst, average, histogram

def write_tm_complexity_summary(pairs):
    """
    For each pair (L_a, L_b) provided in 'pairs', compute the TM complexity statistics and write
    a summary to "tm_complexity_summary.txt". The summary includes:
      - Best (minimum) number of steps.
      - Worst (maximum) number of steps.
      - Average number of steps (⟨n⟩).
      - A histogram mapping each computed step count to its frequency.
    """
    with open("tm_complexity_summary.txt", "w") as f:
        for L_a, L_b in pairs:
            best, worst, avg, hist = compute_tm_complexity(L_a, L_b)
            f.write(f"For La,b = [{L_a}, {L_b}]:\n")
            f.write(f"  Best (min n): {best}\n")
            f.write(f"  Worst (max n): {worst}\n")
            f.write(f"  Average (⟨n⟩): {avg:.2f}\n")
            f.write("  Histogram (n : frequency):\n")
            for n_val in sorted(hist.keys()):
                f.write(f"    {n_val} : {hist[n_val]}\n")
            f.write("\n")
    print("TM complexity summary written to 'tm_complexity_summary.txt'.")

# List of (L_a, L_b) pairs to test
test_pairs = [(2, 3), (3, 2), (3, 5), (5, 3), (3, 12), (12, 3)]

if __name__ == "__main__":
    write_tm_complexity_summary(test_pairs)

# ----------------------------
# Part e
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap_deterministic():
    """
    Generate a 2D heatmap for the average complexity ⟨n⟩ of the deterministic TM
    (from part d) over grid points La,b = [a, b] for a, b ∈ [2, 30].

    The average complexity is computed using:
       n = 1 + L_b + (L_a + L_b) * (# of 1's in the multiplier)
    where the average is taken over all 2^(L_b) possible multiplier bit strings.
    Note that this model is not symmetric (swapping L_a and L_b changes the constant term).
    """
    a_vals = np.arange(2, 31)
    b_vals = np.arange(2, 31)
    avg_matrix = np.zeros((len(a_vals), len(b_vals)))
    
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            _, _, avg, _ = compute_tm_complexity(a, b)
            avg_matrix[i, j] = avg
    
    plt.figure(figsize=(8, 6))
    # Using imshow; x-axis is L_b (multiplier length), y-axis is L_a (multiplicand length)
    heatmap = plt.imshow(avg_matrix, origin='lower', extent=[2, 30, 2, 30], aspect='auto', cmap='viridis')
    plt.colorbar(heatmap, label='Average Steps ⟨n⟩')
    plt.xlabel('L_b (Multiplier Length)')
    plt.ylabel('L_a (Multiplicand Length)')
    plt.title('Heatmap of Average Complexity (Deterministic TM)')
    plt.savefig("tm_complexity_heatmap.png")
    plt.close()
    print("Deterministic TM complexity heatmap saved as 'tm_complexity_heatmap.png'.")

if __name__ == "__main__":
    generate_heatmap_deterministic()

# ----------------------------
# Part f
# ----------------------------
def compute_probabilistic_tm_complexity(L_a, L_b):
    """
    Compute the complexity for a nondeterministic (probabilistic) TM design that
    reduces the computational complexity and removes the asymmetry between L_a and L_b.
    
    In this design the TM first nondeterministically chooses the smaller input as the multiplier.
    Let L_min = min(L_a, L_b) and L_max = max(L_a, L_b). Then we assume:
      - If the chosen multiplier (of length L_min) is all zeros:
           n = 1 + L_min    (base cost for processing)
      - Otherwise (if at least one '1' exists):
           n = 1 + L_min + (L_a + L_b)   (base cost plus one concurrent addition)
    
    Thus:
      best = 1 + L_min
      worst = 1 + L_min + (L_a + L_b)
      average = [ (1/2^(L_min))*(1 + L_min) + (1 - 1/2^(L_min))*(1 + L_min + (L_a + L_b)) ]
    
    The histogram is simplified to these two outcomes.
    """
    L_min = min(L_a, L_b)
    L_max = max(L_a, L_b)
    best = 1 + L_min
    worst = 1 + L_min + (L_a + L_b)
    total_multiplier_cases = 2 ** L_min
    # For each possibility of the chosen multiplier, the multiplicand has 2^(L_max) choices.
    freq_best = (1) * (2 ** L_max)         # Only one all-zero multiplier possibility.
    freq_worst = (total_multiplier_cases - 1) * (2 ** L_max)
    total_cases = total_multiplier_cases * (2 ** L_max)
    average = (best * freq_best + worst * freq_worst) / total_cases
    histogram = {best: freq_best, worst: freq_worst}
    return best, worst, average, histogram

def write_nondet_tm_complexity_summary(pairs):
    """
    For each pair (L_a, L_b) in 'pairs', compute the nondeterministic TM complexity statistics
    and write a summary to "nondet_tm_complexity_summary.txt".
    """
    with open("nondet_tm_complexity_summary.txt", "w") as f:
        for L_a, L_b in pairs:
            best, worst, avg, hist = compute_probabilistic_tm_complexity(L_a, L_b)
            f.write(f"For La,b = [{L_a}, {L_b}]:\n")
            f.write(f"  Best (min n): {best}\n")
            f.write(f"  Worst (max n): {worst}\n")
            f.write(f"  Average (⟨n⟩): {avg:.2f}\n")
            f.write("  Histogram (n : frequency):\n")
            for n_val in sorted(hist.keys()):
                f.write(f"    {n_val} : {hist[n_val]}\n")
            f.write("\n")
    print("Nondeterministic TM complexity summary written to 'nondet_tm_complexity_summary.txt'.")

def simulate_probabilistic_tm(binary1, binary2):
    """
    Simulate a nondeterministic (probabilistic) Turing machine for binary multiplication.
    
    This design first chooses the shorter binary string as the multiplier (to process concurrently),
    thus reducing the sequential dependency. The tape configuration is built as:
      [left blanks] + [multiplicand] + " #" + [chosen multiplier] + " $" + [accumulator] + [right blanks]
    
    If the chosen multiplier contains at least one '1', a single (concurrent) addition is performed.
    """
    # Choose the smaller string as the multiplier.
    if len(binary1) <= len(binary2):
        multiplier = binary1
        multiplicand = binary2
    else:
        multiplier = binary2
        multiplicand = binary1
    
    left_blanks = "B" * 5
    right_blanks = "B" * 5
    width = len(multiplicand) + len(multiplier)
    
    def int_to_bin_str(n):
        return format(n, 'b').zfill(width)
    
    tape_states = []
    # Initial tape configuration.
    tape = left_blanks + multiplicand + " #" + multiplier + " $" + int_to_bin_str(0) + right_blanks
    tape_states.append(tape)
    
    if '1' in multiplier:
        # In the probabilistic design, all addition operations (for bits that are '1')
        # are performed concurrently in a single step.
        acc = int(multiplicand, 2) * int(multiplier, 2)
        # Mark the multiplier as processed (using 'X' as a marker).
        tape = left_blanks + multiplicand + " #X" + " $" + int_to_bin_str(acc) + right_blanks
        tape_states.append(tape)
    else:
        # If the multiplier is all zeros, no addition is needed.
        tape_states.append(tape)
    return tape_states

if __name__ == "__main__":
    # Test the nondeterministic TM complexity summary on a set of (L_a, L_b) pairs.
    nondet_test_pairs = [(2, 3), (3, 2), (3, 5), (5, 3), (3, 12), (12, 3)]
    write_nondet_tm_complexity_summary(nondet_test_pairs)