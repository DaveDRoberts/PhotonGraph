from collections import defaultdict
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import hypernetx as hnx

from .edges import Edge
from ..states.statevector import StateVector
from ..utils import check_integer


class GraphState:
    """
    Represents the most general form of graph state: a qudit hypergraph state.
    Qudits have arbitrary prime dimension, edges are weighted with an
    integer value module qudit dimension and can have a cardinality >=1.

    Notes:
        The theory of hypergraphs and transformations is covered in detail in
        Mariami Gachechiladze's thesis: Quantum Hypergraph States and the Theory
        of Multiparticle Entanglement.

        There are two key papers on qudit hypergraph states: Qudit Hypergraph
        States - Steinhoff et al and Qudit Hypergraph States and their
        Properties - Xiong et al.

    """

    def __init__(self, weighted_edge_dict, qudit_dim, qudits=()):
        """

        Args:
            weighted_edge_dict (dict): Each key is a tuple of qubits and value
                                       is edge weight.
            qudit_dim (int): Dimension of each qudit.
            qudits (iterable): Qudits of graph state

        """

        check_integer(qudit_dim, 2)

        self._qudit_dim = qudit_dim
        self._edges = {}
        self._qudits = set([q for k in weighted_edge_dict.keys() for q in k])
        self._qudits.update(set(qudits))
        self._edges = self._gen_edges(weighted_edge_dict)
        self._incidence_dict = {}
        self.__update_inc_dict()

    def __eq__(self, other):

        assert isinstance(other, self.__class__)
        same_qudits = self.qudits == other.qudits
        same_edges = self._edges == other._edges

        return same_qudits and same_edges

    def __repr__(self):
        return 'GraphState(d = {}, n = {})'.format(self._qudit_dim,
                                                   len(self._qudits))

    def __str__(self):
        return 'GraphState(d = {}, n = {})'.format(self._qudit_dim,
                                                   len(self._qudits))

    def _gen_edges(self, weighted_edge_dict):
        """
        Takes in dict where each key is a tuple of qubits and value is the edge
        weight e.g. {(0, 1):1, (2,3):1, (0,1,2):2}.

        Args:
            weighted_edge_dict (dict): Each key is a tuple of qubits and value
                                       is edge weight.

        Returns:
            dict: Each key is a tuple of qubits and value is a Edge object.

        """

        d = self._qudit_dim

        edges = {}
        for qudits, weight in weighted_edge_dict.items():
            edge = Edge(qudits, int(weight % d), d)
            edges[edge.qudits] = edge

        return edges

    def __update_inc_dict(self):
        """
        Creates a new incidence dict each time it is called.

        """
        inc_dict = defaultdict(list)
        for edge in self._edges.values():
            for qudit in edge.qudits:
                inc_dict[qudit].append(edge)

        self._incidence_dict = dict(inc_dict)

    def _update_edges(self, new_edges):
        """
        Updates the edges by adding new_edges according to the rule: if edge
        already exists add weights modulo qudit_dim otherwise add the edge.

        Args:
            new_edges (dict): Each key is a tuple of qubits and value is edge
                              weight.

        """

        for new_edge in new_edges.values():
            e_edge = self._edges.get(new_edge.qudits)
            if e_edge:
                e_edge.add_weight(new_edge.weight)
                if e_edge.weight == 0:
                    del self._edges[new_edge.qudits]

            else:
                if new_edge.weight:
                    self._edges[new_edge.qudits] = new_edge

        self.__update_inc_dict()

    def __qudit_adjacency(self, qudit):
        """
        The adjacency of a qudit is the generalisation of the neighbourhood
        for edges of cardinality greater than 2. The adjacency of a qudit is
        generated by taking all the edges it belongs to and removing it. The
        resultant edges form the adjacency of the qudit. The weight of each edge
        remains unchanged.

        e.g. if we have the edges Edge([0,1,2],1,3), Edge([1,2],2,3),
             Edge([1,3,4],1,3). Then the adjacency of qudit 1 is
             A(1) = [Edge([0,2],1,3), Edge([2],2,3), Edge([3,4],1,3)]

        Args:
            qudit (int):

        Returns:
            list: Contains Edge objects
        """
        d = self._qudit_dim
        adjacency = []

        for edge in self._incidence_dict[qudit]:
            adj_edge_qudits = edge.qudits.difference({qudit})
            if adj_edge_qudits:
                adj_edge = Edge(adj_edge_qudits, edge.weight, d)
                adjacency.append(adj_edge)

        return adjacency

    @staticmethod
    def _check_cardinality(edges):
        """

        Args:
            edges:

        Returns:
            bool:
        """

        for edge in edges:
            if edge.cardinality != 2:
                return False

        return True

    @property
    def qudit_dim(self):
        """int: Dimension of each qudit"""
        return self._qudit_dim

    @property
    def qudit_num(self):
        """int: Number of qudits in graph state"""
        return int(len(self._qudits))

    @property
    def qudits(self):
        """set: Contains all qudits in the graph state."""
        return self._qudits

    @property
    def edges(self):
        """list: Each element is an Edge object."""
        return list(self._edges.values())

    @property
    def incidence_dict(self):
        """dict: Keys are tuples of qudits and values are all the edges they
        belong to e.g. {0:[Edge([0,1],1,2), Edge([0,2],1,2)],
                        1:[Edge([0,1],1,2)],
                        2:[Edge([0,2],1,2)]}

        """
        return self._incidence_dict

    @property
    def stabilizer_gens(self):
        """dict: Stabilizer generators of the graph state. Each key is a qudit
        and its value is a list of tuples where each tuple has the form
        ('op label', qudits, weight) e.g. ('X', [0], 1), ('CZ', [1,2], 1).

        """

        _stab_gens = {}

        for qudit in self._qudits:
            qudit_stab_gen = [('X', [qudit], 1)]
            for edge in self.__qudit_adjacency(qudit):
                qs = edge.qudits
                label = "C" * (len(qs) - 1) + "Z"
                edge.mul_weight(-1)
                qudit_stab_gen.append((label, list(qs), edge.weight))
            _stab_gens[qudit] = qudit_stab_gen

        return _stab_gens

    @property
    def stabilizers(self):
        """
        Using the stabilizer generators generates all of the stabilisers for the graph state.

        Returns:

        """
        raise NotImplementedError

    @property
    def state_vector(self):
        """pg.StateVector: Returns the state vector associated with the graph
        state."""

        n = len(self._qudits)
        d = self._qudit_dim
        qudit_index_map = {q:i for i, q in
                           enumerate(sorted(tuple(self._qudits)))}

        basis_matrix = np.array(list(it.product(*[list(range(d))] * n)))

        new_weights = np.zeros(d ** n, dtype=int)

        for edge in self._edges.values():
            weight = edge.weight
            qudits = edge.qudits
            qudit_indices = [qudit_index_map[q] for q in qudits]
            gZ = weight * np.prod(basis_matrix[:, qudit_indices],
                                  axis=1).flatten()

            new_weights = np.mod(np.add(new_weights, gZ), d)

        amp = 1 / (np.sqrt(d) ** n)

        vector = np.round(amp * np.exp(2j * np.pi * new_weights / d), 8)

        return StateVector(n, d, vector, list(self._qudits))

    def adj_matrix(self):
        """
        This returns the adjacency matrix of the graph state provided all of the edges have a cardinality of 2.

        Raises:
            Exception if there are any edges with cardinality not equal to 2.

        Returns:

        """
        raise NotImplementedError()

    def adjacency(self, qudit):
        """
        Returns the adjacency of a qudit.

        Args:
            qudit (int): Qudit in graph state

        Returns:
            (list): Contains Edge objects
        """
        return self.__qudit_adjacency(qudit)

    def neighbourhood(self, qudit):
        """
        Returns the neighbours of a qudit.

        Args:
            qudit (int): Qudit in graph state

        Returns:
            (list): Contains qudits.
        """
        return list(set([q for edge in self.__qudit_adjacency(qudit)
                         for q in edge.qudits]))

    def add_edges(self, weighted_edge_dict):
        """
        Adds specified edges to graph state. If graph state contains any edges
        which encompass the same qudits their weights are added modulo qudit
        dimension, otherwise the edge is added. If the any edges have weight 0
        they are removed from the graph state.

        Args:
            weighted_edge_dict: Each key is a tuple of qubits and value is edge
                                weight.

        """
        new_edges = self._gen_edges(weighted_edge_dict)
        self._update_edges(new_edges)

    def EPC(self, qudit):
        """
        Local Edge-pair Complementation (EPC) the generalisation of local
        complementation for hypergraph states. The local part is typically
        dropped since, in general, the transformation involves non-local unitary
        operations.

        Args:
            qudit (int): Qudit specified by its number.

        Notes:
            Local complementation (LC) for a qubit graph state involves
            complementing the neighbourhood of the chosen qubit.

            For qudit graph states this generalises to creating a set of edges
            where the weights of each edge are the product of weights of the
            chosen qudit's neighbouring edges. These edges are then added to the
            original graph where the weights are added modulo qudit dimension
            e.g. Consider a 3-qutrit graph state with edges Edge([0,1],2,3) and
            Edge([0,2],1,3), LC on qudit 0 adds the following edge
            Edge([1,2],2,3).

            For qubit hypergraph states LEPC on a qubit works as follows:
            1) Generate adjacency of qubit
            2) Generate adjacency pairs by creating sets of all possible pairs
               of edges in the adjacency
            3) Generate a multiset of edges by taking the union of qubit sets in
               each adjacency pair. Only edges with odd multiplicity remain.
            4) The edges in the previous multiset are complemented in the graph.

            It has not been explicitly shown in the literature how EPC
            generalises for qudit hypergraph states. But a reasonable,
            non-rigorous approach is to combine the generalisations for qudit
            graph states and qubit hypergraph states.

            For qudit hypergraphs, edges are represented as tuples containing a
            set of qubits and a weight i.e. ({q1,q2,q3}, w).

            For qudit hypergraph states:
            1) Generate adjacency of qudit
            2) Generate adjacency pairs by creating sets of all possible pairs
               of edges in the adjacency
            3) Generate a multiset of edges by taking the union between qudit
               sets and multiplying weights for each adjacency pair.
            4) The edges are added to the graph state: if the an edge
               encompassing the same qudits is already present their weights are
               added, if not, the edge is added. If any edges have weight 0 they
               are removed from the graph state.

        """
        new_edges = {}
        q_adj = self.__qudit_adjacency(qudit)

        q_adj_pair = [(e1, e2) for e1, e2 in it.combinations(q_adj, r=2)
                      if not (e1.qudits == e2.qudits)]

        for e1, e2 in q_adj_pair:
            new_edge = e1.edge_union(e2)
            new_edges[new_edge.qudits] = new_edge

        self._update_edges(new_edges)

    def pivot(self, qudit_a, qudit_b):
        """
        Pivot a.k.a. edge-local complementation (confusing, right?). This
        involves alternate application of EPC between two qudits.

        Args:
            qudit_a: First qudit.
            qudit_b: Second Qudit.

        """

        self.EPC(qudit_a)
        self.EPC(qudit_b)
        self.EPC(qudit_a)

    def EM(self, qudit, m):
        """
        Edge-multiplication is a local Clifford operation with a
        corresponding graphical transformation: the weight of each edge
        connected to the specified qudit is multiplied by m modulo qudit
        dimension.

        Args:
            qudit (int): Qudit the operation is applied to.
            m (int): Multiplication factor.

        """
        assert isinstance(m, int)

        edges = self._incidence_dict[qudit]
        for edge in edges:
            edge.mul_weight(m)

    def ctrl_perm(self, targ, ctrls=frozenset()):
        """
        Performs a permutation operation which corresponds to CkX gates with k
        control qudits.

        Args:
            targ (int): Target qudit.
            ctrls (set): Control qudits

        Notes:
            This operation involves:
            1) Generate the adjacency of the target qudit
            2) Create a set of edges by adding all of the control qudits to
               each edge qudit set in the adjacency.
            3) The edges are added to the graph state: if the an edge
               encompassing the same qudits is already present their weights are
               added, if not, the edge is added. If any edges have weight 0 they
               are removed from the graph state.

        Todo: Check that target and control qubits belong to graph state.

        """

        new_edges = {}
        d = self._qudit_dim
        targ_adj = self.__qudit_adjacency(targ)

        for edge in targ_adj:
            new_qudits = frozenset(edge.qudits.union(ctrls))
            new_weight = edge.weight % d
            new_edge = Edge(new_qudits, new_weight, d)
            new_edges[new_qudits] = new_edge

        self._update_edges(new_edges)

    def measure_X(self, qudit, adj_qudit, state=0):
        """
        Performs a Pauli-X measurement on a specified qudit. For this to
        correspond to a graphical transformation an adjacent qudit must be
        specified. For standard graph states it doesn't really matter which
        adjacent qubit is chosen as the resultant graph states are all
        LU-equivalent.

        Args:
            qudit (int): Qudit to be measured.
            adj_qudit (int): Qudit adjacent/neighbour to measured qudit.
            state (int): Basis states are assigned integers 0 to d-1. If
            state==-1 then state is chosen randomly.

        """
        raise NotImplementedError()

    def measure_Y(self, qudit, state=0):
        """
        Performs a Pauli-Y measurement on a specified qudit.

        Args:
            qudit (int): Qudit to be measured.
            state (int): Basis states are assigned integers 0 to d-1. If
                         state==-1 then state is chosen randomly.

        """

        raise NotImplementedError()

    def measure_Z(self, qudit, state=0):
        """
        Performs a Pauli-Z measurement on a specified qudit.

        Args:
            qudit (int): Qudit to be measured.
            state (int): Basis states are assigned integers 0 to d-1.

        """

        valid_states = [i for i in range(self._qudit_dim)]
        assert state in valid_states, \
            "State must be one of {}".format(valid_states)

        assert len(self._qudits) > 1, 'Graph state must have more than 1 qudit.'

        qudit_adj = self.adjacency(qudit)
        edges_to_remove = self.incidence_dict[qudit]

        for edge in edges_to_remove:
            del self._edges[edge.qudits]

        new_edges = {}

        for edge in qudit_adj:
            edge.mul_weight(state)
            new_edges[edge.qudits] = edge

        self._update_edges(new_edges)
        self._qudits.remove(qudit)

    def fusion(self, gs, qudit_a, qudit_b):
        """
        Performs type-1 fusion between qudit_a of this graph with
        qudit_b of gs.

        Args:
            gs (pg.GraphState): A graph state.
            qudit_a (int): Qudit from this graph state.
            qudit_b (int): Qudit from input graph state.

        Notes:
            I'm not sure how this operation generalises for qudit
            hypergraph states. But I'll assume that when I fuse qubit_b with
            qudit_a that qudit_a inherits the all of the edges of qudit_b
            and qudit_b becomes a neighbour of qudit_a with edge weight 1.

        """
        assert isinstance(gs, GraphState)

        return NotImplementedError()

    def draw(self, ax=None, **params):
        """
        Visualise graph state.


        Args:
            ax (matplotlib.ax)
            **params:

        TODO: Allow visualisation parameters to be passed in.
        TODO: Fix cropping issues: could apply a white hyperedge to all qudits,
              this wouldn't be seen and might fix the cropping issue.

        """

        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=100)
            fig.tight_layout()

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

        # includes an unconnected qudits
        nx_graph.update(nodes=self._qudits)
        hnx_graph._add_nodes_from(self._qudits)
        # ensures qudits are always plotted in the same order
        node_pos = nx.circular_layout(sorted(self._qudits), scale=1)

        nx_edge_labels = nx.get_edge_attributes(nx_graph, 'weight')

        nx.draw_networkx(nx_graph, pos=node_pos, ax=ax,
                         font_size=12,
                         node_color='teal', node_size=1000,
                         edgecolors='black', edge_color='black',
                         font_color='black', width=4, linewidths=3)

        if self._qudit_dim > 2:
            nx.draw_networkx_edge_labels(nx_graph, node_pos, ax=ax,
                                         edge_labels=nx_edge_labels)

        hnx.draw(hnx_graph, pos=node_pos,
                 edge_labels=hnx_edge_labels, ax=ax,
                 with_edge_labels=self._qudit_dim > 2,
                 with_node_labels=False,
                 edges_kwargs={'dr': 0.06, 'linewidth': 3},
                 edge_labels_kwargs={}, label_alpha=1)


class QubitGraphState(GraphState):
    """
    Subclass for qubit graph states i.e. qudits of dimension 2

    Note that methods stabilizer_gens_strings, nx_graph and graph_hash only apply for
    graph states where every edge has a cardinality of 2.

    """

    def __init__(self, edge_list, qubits=()):
        """

        """

        weighted_edge_dict = {}
        for edge in edge_list:
            weighted_edge_dict[edge] = 1
        super().__init__(weighted_edge_dict, qudit_dim=2, qudits=qubits)

    def __repr__(self):
        return 'QubitGraphState(n = {})'.format(len(self._qudits))

    def __str__(self):
        return 'QubitGraphState(n = {})'.format(len(self._qudits))

    def add_edges(self, edge_list):
        """
        Adds edges to graph state.

        Args:
            edge_list:

        Returns:

        """
        weighted_edge_dict = {}
        for edge in edge_list:
            weighted_edge_dict[edge] = 1
        new_edges = super()._gen_edges(weighted_edge_dict)
        super()._update_edges(new_edges)

    @property
    def qubit_num(self):
        """int: Number of qubits in graph state"""
        return int(len(self._qudits))

    @property
    def qubits(self):
        """set: Contains all qubits in the graph state."""
        return self._qudits

    def stabilizer_gens_strings(self):
        """dict: Stabilizer generators of the graph state. Each key is a qudit
        and its value is a list of tuples where each tuple has the form
        ('op label', qudits, weight) e.g. ('X', [0], 1), ('CZ', [1,2], 1).

        Same as base class except there is the option to return stabilizer generators as strings provided all edges
        have a cardinality of 2.

        Note: Qubits are labelled left to right the opposite convention for Qiskit!

        """
        assert self._check_cardinality(self.edges), "All edges must have a cardinality of 2."

        stab_gens = super().stabilizer_gens
        qubit_num = len(stab_gens.keys())
        stab_strs = []
        # create a string of I's
        for q, stab_gen in stab_gens.items():
            stab_list = ['I'] * qubit_num
            ops = ['X', 'Z']
            for op, qs, w in stab_gen:
                if (op in ops) and len(qs) == 1 and w == 1:
                    stab_list[qs[0]] = op
                else:
                    raise Exception("Each operation of the stabilizer " +
                                    "generator should be a tuple of the "
                                    "form (label, qubits, weight) where "
                                    "label is 'X' or 'Z' and len(qubits)==1"
                                    "and weight ==1")

            stab_str = "".join(stab_list)
            stab_strs.append(stab_str)

        return stab_strs

    @property
    def nx_graph(self):
        """
        Generates a NetworkX graph.

        Raises:
             Exception if graph state contains edges with cardinality not equal to 2.

        Returns:

        """
        assert self._check_cardinality(self.edges), "All edges must have a cardinality of 2."

        _nx_graph = nx.Graph()
        for edge in self.edges:
            eq = tuple(edge.qudits)
            _nx_graph.add_edge(eq[0], eq[1])

        _nx_graph.update(nodes=self.qubits)

        return _nx_graph

    def graph_hash(self):
        """
        Will use the GSC and Pynauty python packages to generate a hash for the graph state
        Raises:
             Exception if graph state contains edges with cardinality not equal to 2.

        Returns:

        """

        # return hash_graph(self.nx_graph())
        raise NotImplementedError()




