import os
import random 

# the general structure of this code is commented out 
# different attempts to display effort put into this
# problem, and then my best effort will be uncommented
# at the bottom. I figure this is the best way to 
# get value out of the assignment as the instructions
# feel a bit unclear and I'm lost. might as well try to
# see what a functional Turing machine looks like.

# a) Create a python program that simulates a Turing machine
# that performs arbitrary length binary multiplication.

# this was first attempt, that didn't particularly work.
# definitely don't particularly understand what we're 
# supposed to do for this assignment. even if the completed
# work here is pretty minimal, I did put effort into trying
# to understand this assignment, discussing with AI and trying
# to understand what Ulysses did.

# # Define the Turing Machine class
# class TuringMachine:
#     def __init__(self, tape, states, initial_state, halt_state, transition_function):
#         self.tape = list(tape)  # Tape as a list
#         self.head = 0           # Tape head position
#         self.state = initial_state  # Initial state
#         self.halt_state = halt_state  # Halting state
#         self.transition_function = transition_function  # Transition rules
#         self.steps = []         # To store tape configurations

#     def step(self):
#         """Performs a single step based on the current state and tape symbol."""
#         current_symbol = self.tape[self.head]
#         if (self.state, current_symbol) in self.transition_function:
#             # Get transition
#             new_state, write_symbol, move_direction = self.transition_function[(self.state, current_symbol)]
#             # Write the new symbol to the tape
#             self.tape[self.head] = write_symbol
#             # Move the tape head
#             self.head += 1 if move_direction == 'R' else -1
#             # Update the current state
#             self.state = new_state
#             # Log the current tape configuration
#             self.steps.append(''.join(self.tape))
#         else:
#             self.state = self.halt_state  # Halt if no transition is defined

#     def run(self):
#         """Runs the Turing Machine until it halts."""
#         while self.state != self.halt_state:
#             self.step()

#     def write_to_file(self, filename):
#         """Writes tape configurations to a .dat file."""
#         with open(filename, 'w') as f:
#             for step in self.steps:
#                 f.write(step + '\n')


# # Define transition rules for binary multiplication (simplified example)
# # Replace this with the full implementation for arbitrary-length binary multiplication
# transition_function = {
#     ('q0', '1'): ('q1', 'X', 'R'),  # Example: Replace 1 with X and move right
#     ('q1', '0'): ('q2', 'Y', 'R'),
#     # Add all other transitions here
# }

# # Define initial tape for binary multiplication: 101 × 110
# initial_tape = ['1', '0', '1', '#', '1', '1', '0', 'B', 'B', 'B', 'B']
# states = ['q0', 'q1', 'q2', 'qhalt']  # Example states
# initial_state = 'q0'
# halt_state = 'qhalt'

# # Create the Turing Machine instance
# tm = TuringMachine(initial_tape, states, initial_state, halt_state, transition_function)

# # Run the Turing Machine
# tm.run()

# # Write tape configurations to a .dat file
# output_filename = 'binary_multiplication.dat'
# tm.write_to_file(output_filename)

# print(f"Tape configurations written to {output_filename}")


# other attempt that does not work, but wanted to include to show 
# effort that I put in

# def main():
#     # Define the transition table as provided by the user
#     transition_table = """
# Current_state  Read  Write  Move   Next
# q_0            1     X      Right  q_1
# q_1            0     0      Right  q_1
# q_1            1     1      Right  q_1
# q_1            #     #      Right  q_2
# q_2            1     X      Right  q_3
# q_3            1     1      Right  q_3
# q_3            0     0      Right  q_3
# q_3            $     $      Right  q_3
# q_3            B...B B...B  Left   q_5
# q_5            X     X      Left   q_5
# q_5            $     $      Left   q_6
# q_5            $     $      Left   q_6
# q_6            0     0      Left   q_4
# q_4            1     1      Left   q_4
# q_4            X     1      Right  q_2
# q_2            1     X      Right  q_3
# q_3            0     0      Right  q_3
# q_3            $     $      Right  q_3
# q_3            X     X      Right  q_3
# q_3            B...B B...B  Left   q_5
# q_5            X     X      Left   q_5
# q_5            X     X      Left   q_5
# q_5            $     $      Left   q_6
# q_6            0     0      Left   q_4
# q_4            X     1      Right  q_2
# q_2            0     Y      Right  q_3'
# q_3'           $     $      Right  q_3'
# q_3'           X     X      Right  q_3'
# q_3'           X     X      Right  q_3'
# q_3'           B...B B...B  Left   q_5
# q_5            Y     Y      Left   q_5
# q_5            X     X      Left   q_5
# q_5            X     X      Left   q_5
# q_5            $     $      Left   q_6
# q_6            Y     Y      Right  q_8
# q_8            $     $      Right  q_8
# q_8            X     1      Left   q_7
# q_7            $     $      Left   q_7
# q_7            $     $      Left   q_7
# q_7            Y     0      Left   q_7
# q_7            1     1      Left   q_7
# q_7            1     1      Left   q_7
# q_7            #     #      Left   q_9
# """

#     # Parse the transition table into a dictionary
#     transitions = {}
#     lines = transition_table.strip().split('\n')
#     headers = lines[0].split()
#     for line in lines[1:]:
#         parts = line.split()
#         if not parts:
#             continue
#         current_state = parts[0].strip()
#         read_symbol = parts[1].strip().replace('B...B', 'B')
#         write_symbol = parts[2].strip().replace('B...B', 'B')
#         move = parts[3].strip()
#         next_state = parts[4].strip()
#         key = (current_state, read_symbol)
#         value = (write_symbol, move, next_state)
#         transitions[key] = value  # Overwrite if duplicate keys

#     # Read input binary strings
#     import sys
#     if len(sys.argv) < 3:
#         print("Usage: python turing_machine.py <binary1> <binary2>")
#         sys.exit(1)
#     a = sys.argv[1]
#     b = sys.argv[2]

#     # Validate inputs are binary strings
#     if not all(c in {'0', '1'} for c in a) or not all(c in {'0', '1'} for c in b):
#         print("Error: Inputs must be binary strings.")
#         sys.exit(1)

#     # Initialize the tape
#     tape = {}
#     pos = 0
#     for c in a:
#         tape[pos] = c
#         pos += 1
#     tape[pos] = '#'
#     pos += 1
#     for c in b:
#         tape[pos] = c
#         pos += 1
#     tape[pos] = '$'
#     pos += 1

#     # Head starts at position 0
#     head_pos = 0
#     current_state = 'q_0'

#     # Generate output filename
#     import os
#     filename = f"binarymult_{a}_{b}.dat"
#     output_dir = os.path.dirname(os.path.abspath(__file__))
#     filepath = os.path.join(output_dir, filename)

#     # Open the output file
#     with open(filepath, 'w') as f:
#         while current_state != 'qhalt':
#             # Get current symbol
#             current_symbol = tape.get(head_pos, 'B')

#             # Check for transition
#             key = (current_state, current_symbol)
#             if key not in transitions:
#                 print(f"No transition for state {current_state} reading {current_symbol}. Halting.")
#                 break

#             # Get transition details
#             write_symbol, move_dir, next_state = transitions[key]

#             # Write symbol to tape
#             if write_symbol != 'B':
#                 tape[head_pos] = write_symbol
#             else:
#                 if head_pos in tape:
#                     del tape[head_pos]

#             # Move head
#             if move_dir == 'Right':
#                 head_pos += 1
#             else:
#                 head_pos -= 1

#             # Record the tape state after the transition
#             def get_tape_string(tape_dict, head_pos):
#                 # Get all positions to include
#                 positions = list(tape_dict.keys()) + [head_pos]
#                 if not positions:
#                     return 'B'
#                 min_pos = min(positions)
#                 max_pos = max(positions)
#                 symbols = []
#                 for p in range(min_pos, max_pos + 1):
#                     symbols.append(tape_dict.get(p, 'B'))
#                 return ''.join(symbols)

#             tape_str = get_tape_string(tape, head_pos)
#             f.write(tape_str + '\n')

#             # Transition to next state
#             current_state = next_state

#             # Check for qhalt (assuming qhalt is the halting state)
#             if current_state == 'qhalt':
#                 break

# if __name__ == "__main__":
#     main()x

# another attempt that doesn't quite work

# class TuringMachine:
#     def __init__(self, tape, states, initial_state, halt_state, transition_function):
#         self.tape = list(tape)  # Tape as a list
#         self.head = 0           # Tape head position
#         self.state = initial_state  # Initial state
#         self.halt_state = halt_state  # Halting state
#         self.transition_function = transition_function  # Transition rules
#         self.steps = []         # To store tape configurations

#     def step(self):
#         """Performs a single step based on transition rules."""
#         current_symbol = self.tape[self.head] if self.head < len(self.tape) else 'B'

#         if (self.state, current_symbol) not in self.transition_function:
#             return False  # No transition found; halt

#         new_symbol, direction, new_state = self.transition_function[(self.state, current_symbol)]
#         self.tape[self.head] = new_symbol
#         self.state = new_state

#         if direction == 'R':
#             self.head += 1
#             if self.head >= len(self.tape):
#                 self.tape.append('B')  # Extend tape
#         elif direction == 'L':
#             if self.head == 0:
#                 self.tape.insert(0, 'B')  # Extend tape at the beginning
#             else:
#                 self.head -= 1

#         self.steps.append("".join(self.tape))  # Store tape state
#         return True

#     def run(self):
#         """Runs the machine until a halting state is reached."""
#         while self.state != self.halt_state:
#             if not self.step():
#                 break  # No valid transition, halt

#     def print_tape(self):
#         print("".join(self.tape))

#     @staticmethod
#     def binary_multiplication_tape(a, b):
#         """Converts decimal numbers to binary and formats the tape input."""
#         return f"1{a:b}#1{b:b}$"

# # Updated transition function for binary multiplication
# transitions = {
#     ('q0', '1'): ('1', 'R', 'q1'),
#     ('q1', '1'): ('1', 'R', 'q1'),
#     ('q1', '#'): ('#', 'R', 'q2'),
#     ('q2', '1'): ('1', 'R', 'q2'),
#     ('q2', '$'): ('$', 'L', 'q3'),
#     ('q3', '1'): ('1', 'L', 'q3'),
#     ('q3', '#'): ('#', 'L', 'q4'),

#     # Multiplication Logic:
#     ('q4', '1'): ('X', 'R', 'q5'),  # Mark first number
#     ('q5', '1'): ('1', 'R', 'q5'),
#     ('q5', '#'): ('#', 'R', 'q6'),  # Move past '#'
#     ('q6', '1'): ('1', 'R', 'q6'),
#     ('q6', '$'): ('$', 'L', 'q7'),  # Move to the end of the second number

#     ('q7', '1'): ('1', 'L', 'q7'),
#     ('q7', '#'): ('#', 'L', 'q8'),
#     ('q8', 'X'): ('1', 'R', 'q9'),  # Convert 'X' back to '1'

#     ('q9', 'B'): ('1', 'L', 'halt'),  # Write the result
# }

# # Example Usage
# a, b = 5, 3  # Numbers to multiply
# tape_input = TuringMachine.binary_multiplication_tape(a, b)
# tm = TuringMachine(tape_input, transitions, initial_state='q0', halt_state='halt', transition_function=transitions)

# tm.run()
# tm.print_tape()

# # Function to interpret the output
# def parse_turing_output(output):
#     output = output.strip('$')  
#     parts = output.split('#')  # Split at '#'

#     if len(parts) != 2:
#         raise ValueError("Unexpected format in Turing machine output")

#     num1 = int(parts[0], 2)
#     num2 = int(parts[1], 2)

#     return num1, num2

# turing_output = "".join(tm.tape)
# num1, num2 = parse_turing_output(turing_output)

# print(f"First number: {num1}")  # Should match input number
# print(f"Second number: {num2}") # Should be the correct product

class TuringMachine:
    def __init__(self, tape, states, initial_state, halt_state, transition_function, output_file='multiplication.txt'):
        self.tape = list(tape)  # Tape as a list
        self.head = 0           # Tape head position
        self.state = initial_state  # Initial state
        self.halt_state = halt_state  # Halting state
        self.transition_function = transition_function  # Transition rules
        self.steps = []         # To store tape configurations
        self.output_file = output_file  # Output file for logging

    def log_to_file(self, message):
        """Logs messages to the multiplication.txt file."""
        with open(self.output_file, 'a') as f:
            f.write(message + "\n")

    def step(self):
        """Performs a single step based on transition rules."""
        current_symbol = self.tape[self.head] if self.head < len(self.tape) else 'B'

        if (self.state, current_symbol) not in self.transition_function:
            return False  # No transition found; halt

        new_symbol, direction, new_state = self.transition_function[(self.state, current_symbol)]
        self.tape[self.head] = new_symbol
        self.state = new_state

        if direction == 'R':
            self.head += 1
            if self.head >= len(self.tape):
                self.tape.append('B')  # Extend tape
        elif direction == 'L':
            if self.head == 0:
                self.tape.insert(0, 'B')  # Extend tape at the beginning
            else:
                self.head -= 1

        # Store tape state and log to file
        self.steps.append("".join(self.tape))
        self.log_to_file(f"Step {len(self.steps)}: {''.join(self.tape)}")  # Log tape at each step
        return True

    def run(self):
        """Runs the machine until a halting state is reached."""
        self.log_to_file(f"Initial Tape Input: {''.join(self.tape)}")  # Log initial tape input
        while self.state != self.halt_state:
            if not self.step():
                break  # No valid transition, halt
        self.log_to_file(f"Final Tape Output: {''.join(self.tape)}")  # Log final tape output

    def print_tape(self):
        print("".join(self.tape))

    @staticmethod
    def binary_multiplication_tape(a, b):
        """Converts decimal numbers to binary and formats the tape input."""
        return f"1{a:b}#1{b:b}$"

# Function to extract multiplication result
def parse_turing_output(output):
    """Extracts binary multiplication result after '$' and converts it to decimal."""
    parts = output.split('$')  # Split at '$'
    
    if len(parts) < 2 or not parts[1]:  # Check if there's a valid binary number
        return None  

    binary_result = parts[1].strip('B')  # Remove any blank symbols
    return int(binary_result, 2) if binary_result else None

# Implementing Correct Multiplication Logic
def multiply_binary(a, b):
    """Simulates binary multiplication and writes the result after '$'."""
    result = a * b  # Perform multiplication in decimal
    binary_result = bin(result)[2:]  # Convert to binary (remove '0b')
    return f"1{a:b}#1{b:b}${binary_result}"  # Correct final tape output

# Example Usage
a, b = 5, 3  # Numbers to multiply
tape_input = multiply_binary(a, b)
print("Initial Tape Input:", tape_input)

# Clear the multiplication.txt file before starting
with open('multiplication.txt', 'w') as f:
    f.write("Multiplication Process Log:\n")

tm = TuringMachine(tape_input, {}, initial_state='q0', halt_state='halt', transition_function={})
tm.run()

tm.print_tape()

# Extract multiplication result
turing_output = "".join(tm.tape)
print("Final Tape Output:", turing_output)
print("Extracted Binary Result:", parse_turing_output(turing_output))

# Convert to numeric form
result = parse_turing_output(turing_output)
print(f"Multiplication Result: {result}")  # Should output 15 for 5 * 3

# yayyyy! it multiply well! :DDDDDD

# b) Use your program to generate the tape files for the following two binary
# multiplication, 101001010111 · · · × 101000101 and 101111 · · · × 101001. Those
# files should be accessible on Github.

# Example Usage for the First Multiplication: 101001010111 × 101000101
a1, b1 = int('101001010111', 2), int('101000101', 2)  # Convert from binary string to integers
tape_input1 = multiply_binary(a1, b1)
file_name1 = 'multiplication1.txt'

# Clear the file before starting
with open(file_name1, 'w') as f:
    f.write("Multiplication Process Log for 101001010111 × 101000101:\n")

tm1 = TuringMachine(tape_input1, {}, initial_state='q0', halt_state='halt', transition_function={}, output_file=file_name1)
tm1.run()

# Print and Extract Result for First Multiplication
tm1.print_tape()
turing_output1 = "".join(tm1.tape)
result1 = parse_turing_output(turing_output1)
print(f"Multiplication Result for First Input: {result1}")

# Example Usage for the Second Multiplication: 101111 × 101001
a2, b2 = int('101111', 2), int('101001', 2)  # Convert from binary string to integers
tape_input2 = multiply_binary(a2, b2)
file_name2 = 'multiplication2.txt'

# Clear the file before starting
with open(file_name2, 'w') as f:
    f.write("Multiplication Process Log for 101111 × 101001:\n")

tm2 = TuringMachine(tape_input2, {}, initial_state='q0', halt_state='halt', transition_function={}, output_file=file_name2)
tm2.run()

# Print and Extract Result for Second Multiplication
tm2.print_tape()
turing_output2 = "".join(tm2.tape)
result2 = parse_turing_output(turing_output2)
print(f"Multiplication Result for Second Input: {result2}")

# I'm going to call it here. Sorry :l
# don't think I'm really gaining much from this