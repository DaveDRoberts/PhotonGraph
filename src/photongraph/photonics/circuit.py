import strawberryfields as sf
import thewalrus.quantum as twq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import string
from ..utils import sort_tuples_by_ele, common_member, \
    qudit_qubit_encoding, logical_fock_states_lists
from .ops import PPS, Inter
from ..graphs import qubit_REW_state_check, qubit_hyperedges


class Circuit:
    """Represents a photonic circuit specified by a number of spatial modes.

    Paragraph description

    TODO: Write doc strings!

    Attributes:


    """

    def __init__(self, mode_num):
        """

        Args:
            mode_num (int): number of spatial modes
        """
        self._mode_num = mode_num
        self._op_reg = {}
        self._cov_matrix = np.array([])
        self._compiled = False

    @staticmethod
    def __sort_ops(group):
        """
        Sorts operators which a group by the first mode they act on.

        Args:
            group:

        Returns:
            list:
        """
        ops_first_modes = []
        for op in group:
            first_mode = op.modes[0]
            ops_first_modes.append((op, first_mode))

        ops_first_modes_sorted = sort_tuples_by_ele(ops_first_modes, 1)
        new_group = [op for op, f_mode in ops_first_modes_sorted]

        return new_group

    def add_op(self, group_id, op ):
        """
        This adds an op to the op register.
        This will check each group starting from the first to see if there are
        ops acting on the same modes as the op. If not, the op is added to that
        group else, the process continues through the groups, if there are no
        available groups then the a new group is created. Ops are ordered by
        the first mode they act on. Each time the op_reg is updated the group
        which has been changed gets resorted.

        Allow for the optional parameter of a specific group to be added to the
        op. However, would need to check if there was already any ops which
        shared the same modes.

        TODO: Allow this function to add multiple ops at the same time (to the
              same group)
        TODO: Check that the new op doesn't share any modes with the current op

        """

        self._compiled = False

        if group_id not in self._op_reg.keys():
            # check if operator register is empty, if so initialise with an
            # empty group

            if not self._op_reg:
                self._op_reg[('group_0', 0)] = []
            # check the modes of the op
            modes = op.modes
            groups = list(self._op_reg.keys())
            groups_sorted = sort_tuples_by_ele(groups, 1)
            added = False
            for group in groups_sorted:
                occupied_modes = []
                for _op in self._op_reg[group]:
                    occupied_modes.append(_op.modes)
                # check if op has modes in occ_modes
                if common_member(modes, occupied_modes):
                    pass
                else:
                    # self._op_reg[group].append(op)
                    # sort group after new op has been added
                    old_group = self._op_reg[group]
                    old_group.append(op)
                    new_group = self.__sort_ops(group=old_group)
                    self._op_reg[group] = new_group
                    added = True
                    break
            # if op couldn't be added to the previous groups
            # create a new group
            if not added:
                n = len(groups)
                self._op_reg[('group_'+str(n),n)] = [op]
        else:
            # ops, if it does, do not add the op and raise an error
            old_group = self._op_reg[group_id]
            old_group.append(op)
            new_group = self.__sort_ops(old_group)
            self._op_reg[group_id] = new_group

    def remove_op(self, group_id, op_pos):
        """
        This removes an op from the op register.

        Args:
            group_id (tuple): Contains a name and order number
                              e.g. ('group_0', 0)
            op_pos (int): The index of an op in a group

        """
        self._compiled = False
        op_reg = self._op_reg
        op_group = op_reg[group_id]
        del op_group[op_pos]

        self._op_reg[group_id] = op_group

    def config_op(self, group_id, op_pos, **op_params):
        """
        This updates the parameters of an op of a particular group in the
        operator register.

        TODO: Check that the group exists

        Args:
            group_id (tuple):
            op_pos (int):
            op_params (tuple):

        """
        self._compiled = False

        op_reg = self._op_reg
        op = op_reg[group_id][op_pos]

        op.update(**op_params)

        self._op_reg[group_id][op_pos] = op

    def config_op_group(self, group, *group_op_params):
        """
        This updates a group from the operator register with the specified
        parameters.

        Args:
            group (str):
            op_params (tuple): a tuple for each op

        TODO: Check that this functions correctly

        Returns:

        """
        self._compiled = False
        op_reg = self._op_reg

        for op_pos, op_params in enumerate(group_op_params):
            op = op_reg[group][op_pos]
            op.update(op_params)
            self._op_reg[group][op_pos] = op

    def __program(self):
        """
        Takes the current op_reg and generates the SF program.

        Args:

        Returns:
            (sf.program):
        """
        prog = sf.Program(self._mode_num)
        op_reg = self._op_reg

        groups = list(op_reg.keys())
        groups_sorted = sort_tuples_by_ele(groups, 1)

        for group in groups_sorted:
            for op in op_reg[group]:
                prog.append(op.sf_op(), op.modes)

        return prog

    def compile(self):
        """
        Compiles the photonic circuit.
        Build circuit from op_reg
        Generates the covariance matrix describing the photonic state.
        """
        prog = self.__program()
        eng = sf.Engine(backend="gaussian")
        circuit_sim = eng.run(prog)
        self._cov_matrix = circuit_sim.state.cov()
        self._compiled = True

    def draw(self):
        """
        Draws a schematic representation of the circuit.

        """
        raise NotImplementedError()

    def print(self):
        """
        Prints out the operator register in a readable way.

        """
        raise NotImplementedError()

    @property
    def cov_matrix(self):
        return self._cov_matrix


class PostGSG(Circuit):

    """
    Postselected graph state generator photonic circuit.

    """

    def __init__(self, qudit_num, qudit_dim):

        self._qudit_num = qudit_num
        self._qudit_dim = qudit_dim
        super().__init__(qudit_num*qudit_dim)
        self.__build_op_reg()
        self._qudit_state = {}
        self._qubit_state = {}

    def __build_op_reg(self):
        """
        This builds the operator register when an instance is created.


        """

        qn = self._qudit_num
        qd = self._qudit_dim

        self._op_reg[('sources', 0)] = []
        for j in range(qn // 2):
            for i in range(qd):
                self._op_reg[('sources', 0)].append(PPS((qd*2*j+i, qd*2*j+i+qd)))

        self._op_reg[('preFLU', 1)] = []
        for i in range(qn):
            self._op_reg[('preFLU', 1)].append(
                Inter(np.arange(qd*i, qd*(i+1), dtype=int), np.eye(qd)))

        self._op_reg[('fusions', 2)] = []

        self._op_reg[('postFLU', 3)] = []
        for i in range(qn):
            self._op_reg[('postFLU', 3)].append(
                Inter(np.arange(qd*i, qd*(i+1), dtype=int), np.eye(qd)))

    def run(self, mp=False):
        """
        Determines the logical output state using the covariance matrix

        Args:
            mp (bool): Use multiprocessing

        """
        assert self._compiled

        qudit_dim = self._qudit_dim
        qudit_num = self._qudit_num
        mode_num = self._mode_num
        cov_matrix = self._cov_matrix

        qudit_basis_states, fock_states = logical_fock_states_lists(qudit_dim,
                                                                    qudit_num)

        def calc_amp(fock_state):
            return twq.pure_state_amplitude(np.zeros(2 * mode_num), cov_matrix,
                                        fock_state)

        qudit_state = {}
        for i, qs in enumerate(qudit_basis_states):
            amp = calc_amp(fock_states[i])
            if not np.isclose(amp, 0):
                qudit_state[qs] = np.round(amp, 10)

        self._qudit_state = qudit_state

        qudit_qubit_map = qudit_qubit_encoding(self._qudit_dim,
                                               self._qudit_num)
        qubit_state = {qudit_qubit_map[qs]: amp for qs, amp in
                         self._qudit_state.items()}

        self._qubit_state = qubit_state

    def __postselect_logical_state(self, form, Z_projectors, reduced):
        """



            Args:
                form:
                Z_projectors (dict):
                reduced (bool):

            Returns:
                (dict):

        """

        ps_logical_state = {}
        reduced_ps_logical_state = {}
        if form == 'qubit':
            logical_state = self._qubit_state
        elif form== 'qudit':
            logical_state = self._qudit_state

        for qs, amp in logical_state.items():
            if np.all([True if qs[q] == s else False for q, s in
                       Z_projectors.items()]):
                ps_logical_state[qs] = amp
                reduced_qubit_state = tuple([q for i, q in enumerate(qs) if
                                             i not in Z_projectors.keys()])
                reduced_ps_logical_state[reduced_qubit_state] = amp

        if reduced:
            return reduced_ps_logical_state
        else:
            return ps_logical_state

    @staticmethod
    def __print_state(form, logical_state):
        """

        Args:
            state (dict):

        Returns:

        """
        # print out if the state is a graph state
        if form == 'qubit':
            if qubit_REW_state_check(logical_state):
                print("Logical state is a qubit graph state.")
            else:
                print("Logical state is NOT a qubit graph state.")
        elif form == 'qudit':
            print("Qudit graph state checker not implemented yet.")

        for state, amp in logical_state.items():
            state_str = "|" + ''.join(
                "%s " % ','.join(map(str, str(x))) for x in state)[:-1] + ">"

            amp_str = str(amp)

            print(state_str + "  :  " + amp_str)

    def logical_output_state(self, form="qubit", Z_projectors = {}):
        """
        Prints out the logical output state in one of two forms, qubit or qudit.

        Args:
            form (str):
            Z_projectors (dict):

        Examples:

        """
        assert form in ['qubit', 'qudit']

        ps_logical_state = self.__postselect_logical_state(form, Z_projectors,
                                                           reduced=False)
        self.__print_state(form, ps_logical_state)

    def display_gs(self, qudit_type_order, form='qubit', Z_projectors={},
                            inc_ps_qubits = True, label_type="let_num"):
        """
        Displays postselected graph state

        todo: Need implement the code for displaying hypergraphs and qudit
              graphs, and qudit multigraphs.

        Args:
            qudit_type_order (str): e.g. 'bbrrrrbb'
            form (str):
            Z_projectors (dict):
            inc_ps_qubits (bool):
            label_type (str): Options 'num', 'let_num', 'cat_num'

        """
        # need to perform checks on form and Z projectors

        qudit_dim = self._qudit_dim
        qudit_num = self._qudit_num

        qubit_num = int(np.log2(qudit_dim ** qudit_num))
        non_ps_qubit_num = qubit_num - len(list(Z_projectors.keys()))

        if label_type == "let_num":
            qudit_letters = [letter for letter in
                             string.ascii_uppercase[:qubit_num]]
            qubit_numbers = [str(number) for number in
                             range(int(np.log2(qudit_dim)))]
            labels = ['$' + let + num + '$' for let in qudit_letters for num in
                      qubit_numbers]
            qubit_labels = {k: labels[k] for k in range(len(labels))}
            qubit_colours = {i: colour for i, colour in
                             enumerate(qudit_type_order)}

        elif label_type == "num":
            qubit_labels = {k: '$' + str(k) + '$' for k in range(qubit_num)}
            qubit_colours = {k: 'teal' for k in range(qubit_num)}

        elif label_type == "cat_num":
            qubit_labels = {k: '$' + str(k + 1) + '$' for k in
                            range(qubit_num)}
            qubit_colours = {k: 'teal' for k in range(qubit_num)}

        ps_qubit_state = self.__postselect_logical_state(form, Z_projectors,
                                                         True)

        graph_edges = np.array(
            [edge for edge in qubit_hyperedges(ps_qubit_state) if
             len(edge) > 1])

        all_qubits = range(qubit_num)
        ps_qubits = list(Z_projectors.keys())
        non_ps_qubits = [q for q in all_qubits if q not in ps_qubits]

        ro_qubits = non_ps_qubits + ps_qubits
        ro_qubits_labels = {i: qubit_labels[v] for i, v in
                            enumerate(ro_qubits)}
        ro_qubits_colours = dict(
            (i, qubit_colours[v]) if v not in ps_qubits else (i, 'grey') for
            i, v in enumerate(ro_qubits))

        graph = nx.Graph()
        graph.add_edges_from(graph_edges)

        # Include qubits which don't have any edges
        if inc_ps_qubits:
            graph.update(nodes=range(qubit_num))
        else:
            graph.update(nodes=range(non_ps_qubit_num))
            ro_qubits_labels = {k: v for k, v in ro_qubits_labels.items() if
                                k in range(non_ps_qubit_num)}
            ro_qubits_colours = {k: v for k, v in ro_qubits_colours.items() if
                                 k in range(non_ps_qubit_num)}

        fig = plt.figure(1, figsize=(9, 8))

        nx.draw_networkx(graph, pos=nx.circular_layout(graph),
                         labels=ro_qubits_labels, font_size=12,
                         node_color=[ro_qubits_colours[i] for i in
                                     graph.nodes()], node_size=1000,
                         edgecolors='black', edge_color='black',
                         font_color='black', width=4, linewidths=3)

        plt.show()
