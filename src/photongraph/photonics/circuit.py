import strawberryfields as sf
import thewalrus.quantum as twq
import numpy as np
import networkx as nx
import hypernetx as hnx
import matplotlib.pyplot as plt
import string
from ..utils import sort_tuples_by_ele, common_member, \
    qudit_qubit_encoding, logical_fock_states_lists, efficiency_calc, \
    efficiency_scale_factor
from .ops import PPS, Inter
from ..graphs import state_check, graph_state_edges


class Circuit:
    """Represents a photonic circuit specified by a number of spatial modes.

    Uses Strawberry Fields (SF) for photonic simulation.

    """

    def __init__(self, mode_num):
        """

        Args:
            mode_num (int): number of modes
        """
        self._mode_num = mode_num
        self._op_reg = {}
        self._cov_matrix = np.array([])
        self._compiled = False

    def __repr__(self):
        return 'Circuit({})'.format(self._mode_num)

    def __str__(self):
        return 'Circuit({})'.format(self._mode_num)

    @staticmethod
    def __sort_ops(group):
        """
        Sorts operators which are grouped by the first mode they act on.

        Args:
            group (str):

        Returns:
            list:
        """
        ops_first_modes = []
        for op in group:
            first_mode = op.modes[0]
            ops_first_modes.append((op, first_mode))

        ops_first_modes_sorted = sort_tuples_by_ele(ops_first_modes, 1)
        new_group = [op for op, _ in ops_first_modes_sorted]

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
        This updates the parameters of an op in a particular group of the
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
            sf.program:
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
    def compiled(self):
        """bool: Status of compilation."""
        return self._compiled

    @property
    def mode_num(self):
        """int: Number modes in circuit."""
        return self._mode_num

    @property
    def op_reg(self):
        """dict: Contains operator register."""
        return self._op_reg

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
        self._qubit_num = int(np.log2(qudit_dim ** qudit_num))
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

        Todo: Specify qubit/qudit encoding
        Todo: After calculating all the amplitudes, call the normalise
              method on StateVector.

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
        # Use map to apply calc_amp
        def calc_amp(fock_state):
            return twq.pure_state_amplitude(np.zeros(2 * mode_num), cov_matrix,
                                        fock_state)

        qudit_state_un = {}
        for i, qs in enumerate(qudit_basis_states):
            amp = calc_amp(fock_states[i])
            if not np.isclose(np.abs(amp), 0):
                qudit_state_un[qs] = np.round(amp, 10)

        # normalise the probability amplitudes of the logical states
        norm_const = np.sqrt(np.sum(np.square(np.abs(np.array(list(qudit_state_un.values()))))))

        qudit_state = {state:amp/norm_const for state, amp in
                              qudit_state_un.items()}

        self._qudit_state = qudit_state

        qudit_qubit_map = qudit_qubit_encoding(self._qudit_dim,
                                               self._qudit_num)
        qubit_state = {qudit_qubit_map[qs]: amp for qs, amp in
                         self._qudit_state.items()}

        self._qubit_state = qubit_state

    def __postselect_logical_state(self, form, Z_projectors, qudit_perm):
        """



            Args:
                form:
                Z_projectors (dict):
                qubit_perm (list): permutes the labels of qubits

            Returns:
                (dict):

        """

        ps_logical_state_un = {}

        if form == 'qubit':
            logical_state = self._qubit_state
            for qs, amp in logical_state.items():
                if np.all([True if qs[int(q)] == s else False for q, s in
                           Z_projectors.items()]):
                    perm_qs = np.array(list(qs))
                    og_order = list(range(self._qubit_num))
                    perm_qs[og_order] = perm_qs[qudit_perm]
                    ps_logical_state_un[tuple(perm_qs)] = amp

        elif form == 'qudit':
            logical_state = self._qudit_state
            for qs, amp in logical_state.items():
                if np.all([True if qs[int(q)] == s else False for q, s in
                           Z_projectors.items()]):
                    perm_qs = np.array(list(qs))
                    og_order = list(range(self._qudit_num))
                    perm_qs[og_order] = perm_qs[qudit_perm]
                    ps_logical_state_un[tuple(perm_qs)] = amp

        norm_const = np.sqrt(
            np.sum(np.square(np.abs(np.array(list(ps_logical_state_un.values()))))))

        ps_logical_state = {state: amp / norm_const for state, amp in
                       ps_logical_state_un.items()}

        return ps_logical_state

    def __print_state(self, form, logical_state, Z_projectors):
        """

        Args:
            state (dict):

        Returns:

        """
        # print out if the state is a graph state
        if form == 'qubit':
            reduced_ps_logical_state = {}
            for qs, amp in logical_state.items():
                reduced_qubit_state = tuple([int(q) for i, q in enumerate(qs)
                                             if i not in Z_projectors.keys()])
                reduced_ps_logical_state[reduced_qubit_state] = amp

            ps_qubit_num = len(list(reduced_ps_logical_state.keys())[0])

            if state_check(2, ps_qubit_num, reduced_ps_logical_state, "RU"):
                print("Logical state is a qubit graph state.")
            else:
                print("Logical state is NOT a qubit graph state.")
                print("Number of basis states: ", len(logical_state.keys()),
                      "/", str(2**len(list(logical_state.items())[0][0])))
        elif form == 'qudit':
            d = self._qudit_dim
            reduced_ps_logical_state = {}
            for qs, amp in logical_state.items():
                reduced_qubit_state = tuple([int(q) for i, q in enumerate(qs)
                                             if i not in Z_projectors.keys()])
                reduced_ps_logical_state[reduced_qubit_state] = amp

            ps_qudit_num = len(list(reduced_ps_logical_state.keys())[0])

            if state_check(d, ps_qudit_num, reduced_ps_logical_state, "RU"):
                print("Logical state is a qudit RU state.")
            else:
                print("Logical state is NOT a qudit RU state.")
                print("Number of basis states: ", len(logical_state.keys()),
                      "/", str(d ** len(list(logical_state.items())[0][0])))

        for state, amp in logical_state.items():
            state_str = "|" + ''.join(
                "%s " % ','.join(map(str, str(x))) for x in state)[:-1] + ">"

            amp_str = str(amp)

            print(state_str + "  :  " + amp_str)

    def logical_output_state(self, form="qubit", Z_projectors={},
                             qudit_perm=None):
        """
        Prints out the logical output state in one of two forms, qubit or qudit.

        Args:
            form (str):
            Z_projectors (dict):
            qubit_perm (list): permutes the labels of qubits

        Examples:

        """
        assert form in ['qubit', 'qudit'], "Logical output must be either " \
                                           "qubit or qudit."

        if not qudit_perm:
            if form =="qubit":
                qudit_perm = list(range(self._qubit_num))
            elif form =="qudit":
                qudit_perm = list(range(self._qudit_num))

        ps_logical_state = self.__postselect_logical_state(form, Z_projectors,
                                                           qudit_perm)
        self.__print_state(form, ps_logical_state, Z_projectors)

    def display_gs(self, qudit_type_order, form='qubit', Z_projectors={},
                            inc_ps_qubits=True, label_type="let_num",
                   qubit_perm=None):
        """
        Displays postselected graph state

        Args:
            qudit_type_order (str): e.g. 'bbrrrrbb'
            form (str):
            Z_projectors (dict):
            inc_ps_qubits (bool):
            label_type (str): Options 'num', 'let_num', 'cat_num'
            qubit_perm (list): permutes the labels of qubits
        """
        # need to perform checks on form and Z projectors

        qudit_dim = self._qudit_dim

        qubit_num = self._qubit_num
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
                                                         qubit_perm)

        reduced_ps_qubit_state = {}
        for qs, amp in ps_qubit_state.items():
            reduced_qubit_state = tuple([q for i, q in enumerate(qs)
                                         if str(i) not in Z_projectors.keys()])
            reduced_ps_qubit_state[reduced_qubit_state] = amp

        all_qubits = range(qubit_num)
        ps_qubits = list(map(int, Z_projectors.keys()))
        non_ps_qubits = [q for q in all_qubits if q not in ps_qubits]

        ro_qubits = non_ps_qubits + ps_qubits
        ro_qubits_labels = {i: qubit_labels[v] for i, v in
                            enumerate(ro_qubits)}
        ro_qubits_colours = dict(
            (i, qubit_colours[v]) if v not in ps_qubits else (i, 'grey') for
            i, v in enumerate(ro_qubits))

        edges = []
        hyperedges = []
        for edge in graph_state_edges(2, non_ps_qubit_num, reduced_ps_qubit_state).keys():
            if len(edge) == 2:
                edges.append(edge)
            elif len(edge)>2:
                # plotting a single Z edge causes plotting problems
                hyperedges.append(tuple(edge))

        graph = nx.Graph()
        graph.add_edges_from(edges)

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

        # generate node positions
        node_pos = nx.circular_layout(graph)

        nx.draw_networkx(graph, pos=node_pos,
                         labels=ro_qubits_labels, font_size=12,
                         node_color=[ro_qubits_colours[i] for i in
                                     graph.nodes()], node_size=1000,
                         edgecolors='black', edge_color='black',
                         font_color='black', width=4, linewidths=3)


        hg_nodes = set([q for he in hyperedges for q in he])

        hg_node_pos = {node: pos for node, pos in node_pos.items()
                            if node in hg_nodes}

        hg_edge_dict = {str(i) + "he-": he for i, he in enumerate(hyperedges)}

        hg = hnx.Hypergraph(hg_edge_dict)
        #hg._add_nodes_from(list(all_qubits))
        hnx.draw(hg, pos=hg_node_pos, with_edge_labels=False,
                 with_node_labels=False,
                 edges_kwargs={'dr': 0.06, 'linewidth': 3})

        plt.show()

    def coincidence_rate(self, loss_params, fock_states=(),
                         photon_cutoff=1, pulse_rate=0.5*10**9,
                         units="s"):
        """
        Calculates the m-fold coincidence rate for a collection of
        Fock states which which have at least one photon in each subset
        of modes which corresponds to a qudit.

        The concidence rate is given by the sum of Fock state
        probabilities multiplied by the pulse rate of the laser.

        The Fock states can be specifed instead - this can be a much
        faster way of determining the m-fold coincidence rate if there
        are only a few with a non-zero amplitude.

        Args:


        Returns:
            string: Formatted string displaying coincidence rate

        """

        eta = efficiency_calc(loss_params)

        if not fock_states:
            _, fock_states = logical_fock_states_lists(self._qudit_dim,
                                                       self._qudit_num)

        coin_prob = 0.0
        for fock_state in fock_states:

            num_of_modes = len(fock_state)
            prob_amp = twq.pure_state_amplitude(
                np.zeros(2 * num_of_modes), self._cov_matrix, fock_state)
            prob = (np.abs(prob_amp)) ** 2

            photon_occ = [fock_state[i] for i in
                          np.array(fock_state).nonzero()[0]]
            scaled_prob = prob * efficiency_scale_factor(photon_occ,
                                                         eta)
            coin_prob += scaled_prob

        coin_rate = pulse_rate * coin_prob

        if units == "s":
            return '{} Hz'.format(round(coin_rate, 8))
        elif units == "m":
            return '{} min^-1'.format(round(coin_rate * 60, 4))
        elif units == "h":
            return '{} hour^-1'.format(round(coin_rate * 3600, 4))
        elif units == "d":
            return '{} days^-1'.format(round(coin_rate * 3600 * 24, 4))