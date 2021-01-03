from ..graphs.graphstates import QubitGraphState


class OpenGraphState(QubitGraphState):
    """
    Represents an open graph state which is simply a graph state where some
    qudits have been assigned to be input or output qubits. Further, each qubit
    (except exclusively output) is assigned a measurement angle (0 - 2*pi) and
    plane (XY, XZ, YZ).

    """
    def __init__(self, edge_list, qubits=()):
        """

        Args:
            edge_list:
            qubits:
        """
        super().__init__(edge_list, qubits=())
        self._inputs = set()
        self._outputs = set()
        self._meas_planes = dict()
        self._meas_angles = dict()

    @property
    def inputs(self):
        """set: Input qudits of open graph state"""
        return self._inputs

    @property
    def outputs(self):
        """set: Output qudits of open graph state"""
        return self._outputs

    def __calc_flow(self, flow_type):
        """
        Determines if the underlying graph state and input/output configuration
        have the chosen flow type (flow, gflow and extended gflow). If it does


        Args:
            flow_type (str):

        Returns:
            bool, dict: Result of

        """
        return NotImplementedError

    def __visualise_flow(self, flow_type):
        """
        Visualises the specified flow.

        Args:
            flow_type (str):

        Returns:

        """

        return NotImplementedError

    def print_flow(self, flow_type):
        """
        Prints the specified flow in a readable format.

        Args:
            flow_type (str):

        Returns:

        """

        return NotImplementedError

    def draw(self, with_flow=False):
        """

        Args:
            with_flow (bool):

        """

        return NotImplementedError