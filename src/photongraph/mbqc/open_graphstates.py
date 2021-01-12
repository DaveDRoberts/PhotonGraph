from ..graphs.graphstates import QubitGraphState
from .flow import Flow
from collections import defaultdict


class OpenGraphState(QubitGraphState):
    """
    Represents an open graph state which is simply a graph state where some
    qudits have been assigned to be input or output qubits. Further, each qubit
    (except exclusively output) is assigned a measurement angle (0 - 2*pi) and
    plane (XY, XZ, YZ).

    """
    def __init__(self, edge_list, inputs = set(), outputs = set()):
        """

        Args:
            edge_list (list):
            inputs (set):
            outputs (set):
            meas_planes (dict):
            meas_angles (dict):
        """
        super().__init__(edge_list)
        self._inputs = inputs
        self._outputs = outputs
        self._meas_planes = {qubit:"XY" for qubit in self.qubits}
        self._meas_angles = {qubit: 0.0 for qubit in self.qubits}

    @property
    def inputs(self):
        """set: Input qudits of open graph state"""
        return self._inputs

    @property
    def outputs(self):
        """set: Output qudits of open graph state"""
        return self._outputs

    @inputs.setter
    def inputs(self, qubits):
        """

        Args:
            qubits:

        Returns:

        """

        self._inputs = set(qubits)

    @outputs.setter
    def outputs(self, qubits):
        """

        Args:
            qubits:

        Returns:

        """

        self._outputs = set(qubits)

    @property
    def meas_planes(self):
        return self._meas_planes

    @property
    def meas_angles(self):
        return self._meas_angles

    def set_meas_plane(self, qubit, plane):
        """

        Args:
            qubit (int):
            plane (str):

        Returns:

        """
        assert plane.upper() in ["XY", "YZ", "XZ"]
        assert qubit in self.qubits

        self._meas_planes[qubit] = plane

    def set_meas_angle(self, qubit, angle):
        """

        Args:
            qubit (int):
            angle (float):

        Returns:

        """

        assert qubit in self.qubits

        self._meas_angles[qubit] = angle

    def __flow(self):
        """
        Determines if an open graph state has a causal flow. Only applies to XY plane measurements!

        Code developed from pseudocode algorithm found
        dx.doi.org/10.1007/978-3-540-70575-8_70 for finding a maximally delayed
        flow.

        Returns:
            Flow

        Todo: Check that all measurement planes are XY

        """

        l = {}
        corr_sets = {}

        for v in self.outputs:
            l[v] = 0

        inputs = set(self.inputs)
        processed = set(self.outputs)
        vertices = set(self.qubits)
        correct = set(self.outputs).difference(inputs)

        depth = 1
        while True:

            correct_prime = set()
            processed_prime = set()

            for v in correct:
                ngh_v = set(self.neighbourhood(v))
                v_diff_o = vertices.difference(processed)
                u = ngh_v.intersection(v_diff_o)

                if len(u) == 1:
                    corr_sets[list(u)[0]] = v
                    l[list(u)[0]] = depth
                    processed_prime.update(u)
                    correct_prime.add(v)

            if not processed_prime:
                if processed == vertices:
                    partial_order_rev = defaultdict(list)
                    # group qubits by iteration round of algorithm
                    for key, value in sorted(l.items()):
                        partial_order_rev[value].append(key)
                    # reverse the order to get the partial order
                    partial_order = {}
                    for i, r in enumerate(sorted(list(partial_order_rev.keys()), reverse=True)):
                        partial_order[i] = partial_order_rev[r]

                    return Flow(corr_sets,partial_order, "flow")

                raise Exception("This open graph state doesn't have flow.")

            else:

                processed.update(processed_prime)
                correct.difference_update(correct_prime)
                V_diff_In = vertices.difference(inputs)
                p_prime_inter = processed_prime.intersection(V_diff_In)
                correct.update(p_prime_inter)
                depth += 1

    def __gflow(self):
        """

        Returns:

        """
        raise NotImplementedError()

    def __ext_gflow(self):
        """

        Returns:

        """
        raise NotImplementedError()

    def __visualise_flow(self, flow_type):
        """
        Visualises the specified flow.

        Args:
            flow_type (str):

        Returns:

        """

        raise NotImplementedError()

    def flow(self, flow_type):
        """

        Args:
            flow_type (str):

        Returns:

        """
        if flow_type == "flow":
            return self.__flow()
        elif flow_type == "gflow":
            return self.__gflow()
        elif flow_type == "ext_gflow":
            return self.__ext_gflow()

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

            return NotImplementedError()

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


