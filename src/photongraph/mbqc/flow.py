class Flow:
    """
    Holds the information which defines the flow on an open graph state.

    """

    def __init__(self, corr_sets, partial_order, flow_type):
        """

        Args:
            corr_sets:
            partial_order:
            flow_type:
        """
        self._corr_sets = corr_sets
        self._partial_order = partial_order
        self._flow_type = flow_type

    def __repr__(self):
        flow_str = ""
        flow_str += "Correction Sets: \n"

        for qubit, corr_set in self._corr_sets.items():
            flow_str += "f({}) = {} \n".format(str(qubit), str(corr_set))

        flow_str += "\nPartial Order: \n"

        for round in list(self._partial_order.keys()):
            flow_str += str(self._partial_order[round]) + "<"

        flow_str = flow_str[:-1]
        flow_str += "\n\n"
        flow_str += "Depth: {}".format(5)

        return flow_str

    @property
    def flow_type(self):
        return self._flow_type

    def corr_set(self, qubit):
        """
        Returns the correcting set of qubits for a specified qubit.

        Args:
            qubit (int): Qubit.

        Returns:
            set: Correction qubits.
        """
        return self._corr_sets[qubit]


