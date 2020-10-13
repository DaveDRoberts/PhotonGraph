import numpy as np
import itertools as it
from collections import defaultdict
import networkx as nx
import hypernetx as hnx


class QuditStateVector:
    """
    Class to represent the state vector of a n d-dimensional qudit in the
    computational basis.
    Subclasses a dict where keys are the basis states (tuples) and values
    are complex values representing the probability amplitude.


    """
    def __init__(self):
        pass


class Edge:
    """
    Defines a graph edge in the most general sense. Technically this class
    describes a weighted hyperedge i.e. it can contain 1 or more vertices
    (qudits) and have an associated weight which is modulo qudit dimension.


    """
    def __init__(self, qudits, weight, qudit_dim):
        # super().__init__(qudits)
        self._qudits = frozenset(qudits)
        self._weight = weight
        self._qudit_dim = qudit_dim

    def __repr__(self):
        return 'Edge({}, {})'.format(set(self._qudits), self.weight)

    def __str__(self):
        return 'Edge({}, {})'.format(set(self._qudits), self.weight)

    @property
    def qudit_dim(self):
        return self._qudit_dim

    @property
    def weight(self):
        return self._weight

    @property
    def qudits(self):
        return self._qudits

    @property
    def cardinality(self):
        return len(self._qudits)

    def same_qudits(self, edge):
        assert isinstance(edge, Edge)

        return self._qudits == edge.qudits

    def add_weight(self, new_weight):
        self._weight = (self._weight + new_weight) % self._qudit_dim

    def mul_weight(self, m):
        self._weight = (self._weight * m) % self._qudit_dim

    def edge_diff(self, edge):
        """
        Returns an Edge with the same weight but with qudits removed.
        Weight of resultant edge is unchanged.

        Args:
            edge:

        Returns:

        """
        return Edge(self._qudits.difference(edge.qudits), self._weight,
                    self._qudit_dim)

    def edge_union(self, edge):
        """


        Args:
            edge:

        Returns:

        """

        new_weight = self._weight*edge.weight % self._qudit_dim
        return Edge(self._qudits.union(edge.qudits), new_weight,
                    self._qudit_dim)


class GraphState:
    """

    Include an attribute dictionary of dictionaries which NX graphs have

    Make an abstract base class for graph transformations then generate
    specific ones for a graph state e.g.

    Create a graph state visualisation function that will utilise methods from
    NX and HNX e.g draw cardinality-2 edges as straight lines but all other
    cardinality edges as "rubber bands" wth HNX.

    Include method to generate GraphState from QuantumStateVector

    Adding a new edge requires checking if there's an edge which contains the
    same qudits, if so, add the weight of the new one module qudit_dim, else,
    add the new edge.

    """

    def __init__(self, edge_dict, qudit_dim):
        """
        Need to specify the qudit dimension, number of qudits

        """

        self._qudit_dim = qudit_dim
        self._edges = {}
        self._qudits = set([q for k in edge_dict.keys() for q in k])
        self._gen_edges_from_dict(edge_dict)
        self._incidence_dict = {}
        self._update_inc_dict()

    def _gen_edges_from_dict(self, edge_dict):
        """

        Args:
            edge_dict:

        Returns:

        """

        d = self._qudit_dim
        for qudits, weight in edge_dict.items():
            edge = Edge(qudits, weight, d)
            self._edges[edge.qudits] = edge

    def _update_inc_dict(self):
        """
        Creates a new incidence dict each time it is called and generates a
        new one using the edge list.

        Iterate through the set of edges
        i) create a key for each new qudit
        ii) the value for each key is a list containing all the edges that
        qudit belongs to.
        iii)

        Returns:

        """
        inc_dict = defaultdict(list)
        for edge in self._edges.values():
            for qudit in edge.qudits:
                inc_dict[qudit].append(edge)

        self._incidence_dict = dict(inc_dict)

    def _update_edges(self, new_edges):
        """
        Updates the current edge list by adding new_edges according to
        rule: if edge already exists add weights module qudit_dim,
        otherwise add the edge.

        Args:
            new_edges (dict):

        Returns:

        """

        for new_edge in new_edges.values():
            e_edge = self._edges.get(new_edge.qudits)
            if e_edge:
                e_edge.add_weight(new_edge.weight)
                if e_edge.weight == 0:
                    del self._edges[new_edge.qudits]

            else:
                self._edges[new_edge.qudits] = new_edge

        self._update_inc_dict()

    def _qudit_adjacency(self, qudit):
        """
        The adjacency of a qudit is the generalisation of the neighbourhood
        for edges of cardinality greater than 2.

        Select the qudit from the incidence dict, this will retrieve all its
        edges. Then take the union of


        Need to create a set qudit's edges where the qudit is removed from each
        one.

        Args:
            qudit:

        Returns:

        """
        d = self._qudit_dim
        adjacency = []

        for edge in self._incidence_dict[qudit]:
            adj_edge = edge.edge_diff(Edge(frozenset([qudit]), 1, d))
            adjacency.append(adj_edge)

        return adjacency

    @property
    def edges(self):
        return self._edges.values()

    @property
    def qudits(self):
        return self._qudits

    @property
    def stab_gens(self):
        return NotImplementedError()

    def remove_edge(self, edge_qubits):
        """
        Need to find the edge with the specified qubits and remove it

        Args:
            edge_qubits:

        Returns:

        """

        return NotImplementedError()

    def add_edges(self, edge_dict):
        """
        Will need to check if there is already an edge with the same qudits, if
        so, just add the weight of this edge.

        if the resultant weight is 0 then remove the edge.

        Args:
            edge:

        Returns:

        """
        return NotImplementedError()

    def adjacency(self, qudit):
        """

        Args:
            qudit:

        Returns:

        """
        return self._qudit_adjacency(qudit)

    def state_vector(self):

        return NotImplementedError()

    def Z_projection(self, qudit_projs):
        """
        This removes all edges involving the selected qubits.
        Should probably remove the projected qudits from the graph
        Args:
            qudit_projs (dict): Keys are qudits, values are the single-qudit
            basis state to project onto.

        Returns:

        """
        return NotImplementedError()

    def fusion(self, gs, qudit_a, qudit_b):
        """
        Performs type-1 fusion between qudit_a of this graph with
        qudit_b of gs.

        I'm not sure how this operation generalises for qudit
        hypergraph states. But I'll assume that when I fuse qubit_b with
        qudit_a that qudit_a inherits the all of the edges of qudit_b
        and qudit_b becomes a neighbour of qudit_a with edge weight 1.

        Args:
            gs:
            qudit_a:
            qudit_b:

        Returns:

        """
        assert isinstance(gs, GraphState)

        # find largest qudit number - add 1 and add that value to each
        # qudit number in the second graph state

        # Maybe create a method to renumber the qudits
        # add all the renumbered edges to this graph state
        # for qudit_a to inherit the edges of qudit_b, just replace
        # qudit_b with qudit_a in all of qudit_b's edges.
        # Then add an edge between qudit_a and qubit_b.



        return NotImplementedError()

    def LC(self, qudit):
        """
        Performs the generalisation of local complementation

        Args:
            qudit:

        Returns:

        """
        new_edges = {}
        q_adj = self._qudit_adjacency(qudit)

        q_adj_pair = [(e1, e2) for e1, e2 in it.combinations(q_adj, r=2)
                      if not (e1.qudits == e2.qudits)]

        for e1, e2 in q_adj_pair:
            new_edge = e1.edge_union(e2)
            new_edges[new_edge.qudits] = new_edge

        self._update_edges(new_edges)

    def edge_LC(self, qudit_a, qudit_b):
        """

        Args:
            edge_a:
            edge_b:

        Returns:

        """

        self.LC(qudit_a)
        self.LC(qudit_b)
        self.LC(qudit_a)

    def EM(self, edge_qudits, m):
        """
        Multiplies the weight of specified edge by m. The result is
        modulo qudit dimension.

        Args:
            edge_qudits:
            m (int):

        Returns:

        """

        qudits = frozenset(edge_qudits)
        self._edges[qudits].mul_weight(m)

    def ctrl_perm(self, targ, ctrls=frozenset(), w=1):
        """
        First generate the adjacency for the target qubit. For each
        edge in the adjacency create a new edge with the same weight and
        the target qubits added. These adges are added modulo qudit dim
        to the graph


        Args:
            ctrls (set): Contains all of the control qudits
            targ (int): Target qubit
            w (int): Multiplicity/weight of the operation i.e. the
            number of times the operation is applied

        Returns:

        """
        # check that target and control qubits belong to graph

        new_edges = {}
        d = self._qudit_dim
        targ_adj = self._qudit_adjacency(targ)

        for edge in targ_adj:
            new_qudits = frozenset(edge.qudits.union(ctrls))
            new_weight = (w*edge.weight) % d
            new_edge = Edge(new_qudits, new_weight, d)
            new_edges[new_qudits] = new_edge

        self._update_edges(new_edges)

    def draw(self, **params):
        """


        Args:
            **params:

        Returns:

        """

        nx_graph = nx.Graph()

        hnx_edge_labels = {}
        hnx_edge_dict = {}

        for i, edge in enumerate(self._edges.values()):
            eq = tuple(edge.qudits)
            if len(eq) == 2:
                nx_graph.add_edge(eq[0], eq[1], weight=edge.weight)
            else:
                hnx_edge_dict['he' + str(i)] = eq
                hnx_edge_labels['he' + str(i)] = edge.weight

        hnx_graph = hnx.Hypergraph(hnx_edge_dict)
        # nx_graph.update(nodes=range(len(self._qudits)))
        nx_graph.update(nodes=self._qudits)
        #node_pos = nx.circular_layout(nx_graph)
        node_pos = nx.circular_layout(sorted(self._qudits))

        nx_edge_labels = nx.get_edge_attributes(nx_graph, 'weight')

        nx.draw_networkx(nx_graph, pos=node_pos,
                         font_size=12,
                         node_color='teal', node_size=1000,
                         edgecolors='black', edge_color='black',
                         font_color='black', width=4, linewidths=3)

        if self._qudit_dim > 2:
            nx.draw_networkx_edge_labels(nx_graph, node_pos,
                                         edge_labels=nx_edge_labels)

        hg_nodes = set([q for e in hnx_edge_dict.values() for q in e])
        hg_node_pos = {node: pos for node, pos in node_pos.items()
                       if node in hg_nodes}

        hnx.draw(hnx_graph, pos=hg_node_pos,
                 edge_labels=hnx_edge_labels,
                 with_edge_labels=self._qudit_dim > 2,
                 with_node_labels=False,
                 edges_kwargs={'dr': 0.08, 'linewidth': 3},
                 edge_labels_kwargs={}, label_alpha=1)


def state_check(qudit_dim, qudit_num, state_vector, check_type):
    """
    Checks if a qudit state is a LME, RU or REW state (qubit only).

    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits
        state_vector (dict): Each key (tuple) is a basis state and values are
        the associated amplitudes.
        check_type (str): Either LME or RU

    Returns:
        bool: Result of check

    """

    assert check_type in ('LME', 'RU')

    basis_matrix = np.array(
        list(it.product(*[list(range(qudit_dim))] * qudit_num)))

    if len(state_vector) == qudit_dim ** qudit_num:
        amp_zero = np.round(state_vector[tuple(basis_matrix[0])], 10)
        amp_zero_tilda = np.round(1 / amp_zero, 10)
        all_amps_normed = np.round(np.array(
            [state_vector[tuple(bs)] * amp_zero_tilda for bs in basis_matrix]),
                                   8)
        phis = np.angle(all_amps_normed)
        equal_sup_check = np.all(np.isclose(np.abs(all_amps_normed),
                                            np.ones(qudit_dim ** qudit_num)))
        weights_um = np.array(np.round(phis * (qudit_dim / (2 * np.pi)), 6))

        RU_check, weights = np.modf(np.mod(weights_um, qudit_dim))

        if check_type == "LME":

            if equal_sup_check:
                return True
            else:
                return False

        elif check_type == "RU":
            if equal_sup_check and not (RU_check.any()):
                return True
            else:
                return False
    else:
        return False


def graph_state_edges(qudit_dim, qudit_num, state_vector):
    """
    For a given qudit state vector the weighted edges of the graph state
    are generated if it is a graph state.

    First, check if the all basis states are present in the state vector.
    Then check if the modulus of all amplitudes are equal, and if complex
    phases are all integer multiples of w=j2*pi/d. These are to check if
    the state is a roots of unity (RU) state (for qubits this is a real
    equally weighted (REW) state).

    There is a one-to-one correspondence between hypergraph states and REW
    states. However, not all RU states are qudit graph states.

    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits
        state_vector (dict): Each key (tuple) is a basis state and values are
        the associated amplitudes.

    Returns:
        dict: Each value is an edge (tuple) and the values are weight of
        the edge. "Normal" qubit edges are always weight 1.

    """

    # generates a matrix where each row is a basis state
    basis_matrix = np.array(
        list(it.product(*[list(range(qudit_dim))] * qudit_num)))

    if not len(state_vector) == qudit_dim ** qudit_num:
        return {}
    # This removes any global phase by dividing all basis
    # state amps by the amp of the all-zero state
    amp_zero = np.round(state_vector[tuple(basis_matrix[0])], 8)
    if not amp_zero:
        return {}
    amp_zero_tilda = np.round(1 / amp_zero, 8)

    all_amps_normed = np.array(
        [state_vector[tuple(bs)] * amp_zero_tilda for bs in basis_matrix])
    phis = np.angle(all_amps_normed)

    weights_um = np.array(np.round(phis * (qudit_dim / (2 * np.pi)), 6))

    # check if state is a RU state, if not return empty {}
    RU_check, weights = np.modf(np.mod(weights_um, qudit_dim))
    equal_sup = np.all(np.isclose(np.abs(all_amps_normed), np.ones(qudit_dim ** qudit_num)))
    if RU_check.any() or (not equal_sup):
        return {}

    weights = weights.astype(int)

    state_vector_w = {tuple(bs): weight for bs, weight in
                      zip(basis_matrix, weights)}

    bm_states = list(it.product((0, 1), repeat=qudit_num))[1:]
    basis_manifold = defaultdict(list)
    for bm_state in bm_states:
        basis_manifold[np.array(bm_state).sum()].append(bm_state)

    k = 1

    edges = {}

    while True:

        new_edges = {}

        for bm_state in basis_manifold[k]:
            bm_state_w = state_vector_w[bm_state]
            # checks if the basis state has a non-zero weight
            if bm_state_w:
                new_edge = tuple(np.array(bm_state).nonzero()[0])
                new_edges[new_edge] = bm_state_w
                edges[new_edge] = bm_state_w

        if new_edges:
            for edge, edge_weight in new_edges.items():
                gZ = edge_weight*np.prod(basis_matrix[:, edge], axis=1).flatten()
                weights = np.mod(np.subtract(weights, gZ), qudit_dim)

            state_vector_w = {tuple(bs): weight for bs, weight in
                              zip(basis_matrix, weights)}

        # check if all signs are 0 i.e. + except for the |0>^tensor(n) state
        if np.array(list(weights)).sum() == 0:
            break
        else:
            k = k + 1
        # this means that the RU state is not a graph state
        if k > qudit_num:
            return {}

    return edges


def state_vector_from_edges(qudit_dim, qudit_num, edges):
    """
    Takes the edges of a graph state and generates the associated state vector.


    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits
        edges (dict): Edges of graph state

    Returns:
        dict: state vector
    """

    # generates a matrix where each row is a basis state
    basis_matrix = np.array(
        list(it.product(*[list(range(qudit_dim))] * qudit_num)))

    new_weights = np.zeros(qudit_dim ** qudit_num, dtype=int)

    for edge, weight in edges.items():
        gZ = weight * np.prod(basis_matrix[:, edge], axis=1).flatten()
        new_weights = np.mod(np.add(new_weights, gZ), qudit_dim)

    state_vector_w = {tuple(bs): new_weight for bs, new_weight in
                      zip(basis_matrix, new_weights)}

    # convert from weight form to standard form
    amp = 1 / (np.sqrt(qudit_dim) ** qudit_num)
    state_vector = {
        state: np.round(amp * np.exp(2j * np.pi * weight / qudit_dim), 8) for
        state, weight in state_vector_w.items()}

    return state_vector
