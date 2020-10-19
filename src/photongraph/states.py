

class StateVector:

    def __init__(self, vector, qudit_num, qudit_dim):
        """


        Args:
            vector:
            qudit_num:
            qudit_dim:
        """
        self._vector = vector
        self._qudit_num = qudit_num
        self._qudit_dim = qudit_dim

    def __repr__(self):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def _basis_states(self):
        """
        Generates the basis states for the state vector in the standard
        order.

        Returns:

        """
        return NotImplementedError

    def evolve(self, U):
        return NotImplementedError

    def inner_product(self, state):
        return NotImplementedError()

    def normalize(self):
        return NotImplementedError

    def schmidt_measure(self):
        """
        Computes the Schmidt measure for the state vector

        Returns:

        """
        return NotImplementedError

    def state_check(self, check_type):
        """
        Checks if state is LME, RU

        Returns:

        """
        return NotImplementedError

    def graph_state(self):
        """
        This return s GraphState object if the state vector is a graph
        state.

        Returns:

        """
        return NotImplementedError

    def logical_fock_states(self):
        """
        Generate an ordered dictionary
        Returns:

        """
        return NotImplementedError

    @property
    def dim(self):
        return self._qudit_dim

    @property
    def num(self):
        return self._qudit_num

    @property
    def vector(self):
        return self._vector



