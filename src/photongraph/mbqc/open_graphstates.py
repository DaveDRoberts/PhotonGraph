from ..graphs.graphstates import QubitGraphState
from .flows import flow


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

    def draw(self, ax=None, with_flow=False, animation=False, save_fig=False):
        """
        This method is similar to that from the super class except qubits are drawn
        as a grid with all input qubits on the left and all output qubits on the right.
        If a qubit is both an input and output then it is put with the output qubits.

        Label each non-output qubit with measurement angle and plane.

        Qubit shape code:
            Input - Square
            Output  - Octogon
            Input/Output  - Square+octgon
            Auxillary - Circle

        Flow colour code:
            Purple arrow from qubit to its correction qubit(s).
            Grey dashed "hyper-edge" to encompass all qubits in the same measurement round
            and labelled with round number.

        If animation, then a matplotlib animation is generated which illustrates the sequence of measurement rounds.
        Note that in general, qubit measurements will not

        Args:
            ax (matplotlib.ax):
            with_flow (bool):
            animation (bool):

        """

        return NotImplementedError

    def zx_diagram(self):
        """

        Returns:

        """

        raise NotImplementedError

    def qc_qiskit(self):
        """
        Generates the corresponding quantum circuit for qiskit.

        Returns:

        """

        raise NotImplementedError()

    def qc_cirq(self):
        """
        Generates the corresponding quantum circuit for qiskit.

        Returns:

        """

        raise NotImplementedError()
