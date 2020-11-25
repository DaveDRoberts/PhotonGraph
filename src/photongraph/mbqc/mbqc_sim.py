from .open_graphstates import OpenGraphState


class MBQCSimulator:
    """
    Takes in an open graph state with an assignment of input, output qubits and
    measurement angles. Simulator will need to calculate gflow.

    Features:
        - Generate an animation illustrating measurement sequences and the
          evolution of the computation.
        - Choose different simulation methods:
            + A symbolic simulation
            + Logical Heisenberg picturen simulation of logical operators and
              stabilizers
            + Translate MBQC pattern to a quantum circuit and utilise optimized
              quantum circuit simulators.

    """

    def __init__(self, og_state):
        """

        Args:
            og_state (OpenGraphState):
        """

    def simulate(self):
        """

        Returns:

        """

        return NotImplementedError
