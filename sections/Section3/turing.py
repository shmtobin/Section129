import os

# Define the Turing Machine class
class TuringMachine:
    def __init__(self, tape, states, initial_state, halt_state, transition_function):
        self.tape = list(tape)  # Tape as a list
        self.head = 0           # Tape head position
        self.state = initial_state  # Initial state
        self.halt_state = halt_state  # Halting state
        self.transition_function = transition_function  # Transition rules
        self.steps = []         # To store tape configurations

    def step(self):
        """Performs a single step based on the current state and tape symbol."""
        current_symbol = self.tape[self.head]
        if (self.state, current_symbol) in self.transition_function:
            # Get transition
            new_state, write_symbol, move_direction = self.transition_function[(self.state, current_symbol)]
            # Write the new symbol to the tape
            self.tape[self.head] = write_symbol
            # Move the tape head
            self.head += 1 if move_direction == 'R' else -1
            # Update the current state
            self.state = new_state
            # Log the current tape configuration
            self.steps.append(''.join(self.tape))
        else:
            self.state = self.halt_state  # Halt if no transition is defined

    def run(self):
        """Runs the Turing Machine until it halts."""
        while self.state != self.halt_state:
            self.step()

    def write_to_file(self, filename):
        """Writes tape configurations to a .dat file."""
        with open(filename, 'w') as f:
            for step in self.steps:
                f.write(step + '\n')


# Define transition rules for binary multiplication (simplified example)
# Replace this with the full implementation for arbitrary-length binary multiplication
transition_function = {
    ('q0', '1'): ('q1', 'X', 'R'),  # Example: Replace 1 with X and move right
    ('q1', '0'): ('q2', 'Y', 'R'),
    # Add all other transitions here
}

# Define initial tape for binary multiplication: 101 × 110
initial_tape = ['1', '0', '1', '#', '1', '1', '0', 'B', 'B', 'B', 'B']
states = ['q0', 'q1', 'q2', 'qhalt']  # Example states
initial_state = 'q0'
halt_state = 'qhalt'

# Create the Turing Machine instance
tm = TuringMachine(initial_tape, states, initial_state, halt_state, transition_function)

# Run the Turing Machine
tm.run()

# Write tape configurations to a .dat file
output_filename = 'binary_multiplication.dat'
tm.write_to_file(output_filename)

print(f"Tape configurations written to {output_filename}")