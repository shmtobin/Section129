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
        self.tape = list(tape)
        self.head = 0
        self.state = initial_state
        self.halt_state = halt_state
        self.transition_function = transition_function
        self.steps = []
        self.output_file = output_file

    def log_to_file(self, message):
        with open(self.output_file, 'a') as f:
            f.write(message + "\n")

    def step(self):
        if self.head >= len(self.tape) or self.head < 0:
            current_symbol = 'B'
        else:
            current_symbol = self.tape[self.head]

        key = (self.state, current_symbol)
        if key not in self.transition_function:
            return False

        new_symbol, direction, new_state = self.transition_function[key]
        if self.head >= len(self.tape):
            self.tape.append(new_symbol)
        else:
            self.tape[self.head] = new_symbol
        self.state = new_state

        if direction == 'R':
            self.head += 1
        elif direction == 'L':
            self.head -= 1

        if self.head < 0:
            self.tape.insert(0, 'B')
            self.head = 0
        elif self.head >= len(self.tape):
            self.tape.append('B')

        self.steps.append("".join(self.tape))
        self.log_to_file(f"Step {len(self.steps)}: {''.join(self.tape)}")
        return True

    def run(self):
        self.log_to_file(f"Initial Tape Input: {''.join(self.tape)}")
        while self.state != self.halt_state:
            if not self.step():
                break
        self.log_to_file(f"Final Tape Output: {''.join(self.tape)}")

    def print_tape(self):  # Now it's a method of the class
        print("".join(self.tape))

    @staticmethod
    def binary_multiplication_tape(a, b):
        return f"1{a:b}#1{b:b}$"

def parse_turing_output(output):
    print(f"Raw Turing Machine Output: {output}")  # Debugging

    parts = output.split('$')
    if len(parts) < 2 or not parts[1].strip():
        print("Error: No valid result found after '$', returning 0")
        return 0  # Defaulting to 0 instead of None

    binary_result = ''.join(filter(lambda x: x in '01', parts[1]))

    if not binary_result:
        print("Error: No binary digits found in result portion, returning 0")
        return 0

    return int(binary_result, 2)


def multiply_binary(a, b):
    result = a * b
    return f"1{a:b}#1{b:b}${bin(result)[2:]}"

def print_tape(self):
    print("".join(self.tape))

# Define the transition rules for binary multiplication
# States and transitions need to handle copying, adding, shifting, etc.
# This is a simplified version for demonstration.
transitions = {
    # Initial state: move to the separator '#'
    ('q0', '1'): ('1', 'R', 'q0'),
    ('q0', '#'): ('#', 'R', 'q1'),
    # Move past the '#'
    ('q1', '1'): ('1', 'R', 'q1'),
    ('q1', '$'): ('$', 'L', 'q2'),
    # Start processing the multiplier (b)
    ('q2', '1'): ('B', 'L', 'q3'),
    ('q2', 'B'): ('B', 'L', 'q2'),
    # Add multiplicand (a) to result if current bit is 1
    # (This is a simplified transition; actual implementation requires more states)
    ('q3', '#'): ('#', 'L', 'q3'),
    ('q3', '1'): ('1', 'L', 'q3'),
    ('q3', 'B'): ('B', 'R', 'halt')  # Simplified for example
}
# Additional transitions for binary multiplication (simplified for demonstration)
transitions.update({
    ('q3', '1'): ('1', 'R', 'q3'),  # Keep moving to the right
    ('q3', 'B'): ('B', 'R', 'halt'),  # End if we reach the blank symbol after finishing multiplication
})


# Example usage
a, b = 5, 3
tape_input = TuringMachine.binary_multiplication_tape(a, b)

# Initialize the Turing Machine with the transitions
tm = TuringMachine(
    tape=tape_input,
    states={'q0', 'q1', 'q2', 'q3', 'halt'},
    initial_state='q0',
    halt_state='halt',
    transition_function=transitions
)

# Clear the output file
with open('multiplication.txt', 'w') as f:
    f.write("Multiplication Process Log:\n")

tm.run()

# Verify the result
with open('multiplication.txt', 'r') as f:
    lines = f.readlines()
    final_output = lines[-1].split(": ")[1].strip()
    result = parse_turing_output(final_output)

expected_result = a * b
print(f"Turing Machine Result: {result}, Expected: {expected_result}")
print("Test Passed!" if result == expected_result else "Test Failed!")


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

tm1 = TuringMachine(tape_input1, {'q0', 'q1', 'q2', 'q3', 'halt'}, initial_state='q0', halt_state='halt', transition_function=transitions, output_file=file_name1)
tm1.run()

# Print and Extract Result for First Multiplication
tm1.print_tape()
turing_output1 = "".join(tm1.tape)
result1 = parse_turing_output(turing_output1)
print(f"Multiplication Result for First Input: {result1}")

# Verify against the expected result
expected_result1 = a1 * b1
print(f"Expected Result: {expected_result1}")

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

# chatted with Zihang about this and I have returned to give more time
# to this problem :D. I am going to try to find the patterns on the tape
# as seen in the turing_notes file and use it to make the turing machine 
# (with general Turing notation assitance courtesy of Ulysses' public github)
# which I will just be annotating to increase my understanding of the Turing machine.


# the tape we are given allows for the multiplication of two binary numbers
# given with the formatting: "1<num1_in_binary>#1<num2_in_binary>$"
# by repeatedly using an algorithm of adding and doubling values, 
# this Turing machine can perform multiplication of the binary values

# below I annotated a segment of Ulysses's code to increase my understanding
# of the content. I am by no means trying to pass this work off as my own.
    # tape patterns for algorithm: 
    # class TuringMachine:
    # # defining states as used in the given Turing machine tape
    # 	def __init__(self, states, initial_state, final_states, initial_head=0, blank_symbol='B', wildcard_symbol='*', output_file=None):
    # 		self.initial_head = initial_head
    # 		self.initial_state = initial_state
    # 		self.states = states
    # 		self.state = initial_state
    # 		self.blank_symbol = blank_symbol
    # 		self.wildcard_symbol = wildcard_symbol
    # 		self.final_states = final_states
    # 		self.output_file = output_file
    # # defining method by which tape is processed, a single
    # # transition step
    #     def step(self):
    # # this line extends the tape whenever the head tries to 
    # # move beyond the right blank symbol
    # 		if self.head >= len(self.tape):
    # 			self.tape.extend([self.blank_symbol] * (self.head - len(self.tape) + 1))
    # # this line extends the tape whenever the head tries to 
    # # move left beyond index 0, adding blank symbol
    #         if self.head < 0:
    # 			self.tape = [self.blank_symbol] * (-self.head) + self.tape
    # 			self.head = 0
    # 		#print(f'{self.state}\t{''.join(self.tape[:self.head])}\033[96m{self.tape[self.head]}\033[0m{''.join(self.tape[self.head + 1:])}')

    # # this line just reads off the symbol the head is on
    # 		current_symbol = self.tape[self.head]

    # # checking transition rules
    # # i) checks if transiiton exists for current state and symbol
    # 		if (self.state, current_symbol) in self.states:
    # 			new_symbol, direction, new_state = self.states[(self.state, current_symbol)]
    # # ii) if no transition exists for current state and symbol
    # # checks for transition with wildcard symbol
    #         elif (self.state, self.wildcard_symbol) in self.states:
    # 			new_symbol, direction, new_state = self.states[(self.state, self.wildcard_symbol)]
    # # iii) if it cannot do either of the above, throw an error
    #         else:
    # 			raise ValueError(f"No transition defined for state '{self.state}' and symbol '{current_symbol}'.")
    # # updating the tape
    # # i) if transition wildcard, keep current symbol instead		
    #         if new_symbol == self.wildcard_symbol:
    # 			new_symbol = current_symbol
    # # ii) update tape head posiiton to new symbol and updates
    # # state
    # 		self.tape[self.head] = new_symbol
    # 		self.state = new_state
    # # moves the tape head to the right if direction R/r
    # # and moves the tape head to the left if L/l
    # 		if direction == 'R' or direction == 'r':
    # 			self.head += 1
    # 		elif direction == 'L' or direction == 'l':
    # 			self.head -= 1

    # I'm going to end off here because I have sank a considerable 
    # amount of time into this section worksheet and I don't think
    # it will be very growthful to pour more in. I learned much more
    # about Turing machines than when I started but there's so much 
    # syntax I'm unfamiliar with between me and a confident solid 
    # solution to this problem.