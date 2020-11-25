from ..utils import check_integer


class Edge:
    """
    Defines a weighted edge which can encompass one or more qudits i.e. it is
    a weighted qudit hyperedge. Weight is an integer modulo qudit dimension.

    Attributes:
        _qudits (frozenset): Qudits encompassed by edge.
        _weight (int): Weight of the edge.
        _qudit_dim (int): Dimension of each qudit.

    Notes:
        Investigate non-integer weight edges for more general weighted
        hypergraph states.

    """

    def __init__(self, qudits, weight, qudit_dim):
        """

        Args:
            qudits (iterable): Any iterable containing qudits- qudits are
                               numbered by integers
            weight (int): Weight of edge.
            qudit_dim (int): Dimension of qudits encompassed by edge.
        """

        check_integer(weight, 0)
        check_integer(qudit_dim, 2)

        self._qudits = frozenset(qudits)
        self._weight = weight
        self._qudit_dim = qudit_dim

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        same_weight = self.weight == other.weight
        same_qudit_dim = self.qudit_dim == other.qudit_dim
        same_qudits = self.qudits == other.qudits

        return same_weight and same_qudit_dim and same_qudits

    def __repr__(self):
        return 'Edge({}, {})'.format(set(self._qudits), self.weight)

    def __str__(self):
        return 'Edge({}, {})'.format(set(self._qudits), self.weight)

    @property
    def qudit_dim(self):
        """int: Dimension of qudits encompassed by edge."""
        return self._qudit_dim

    @property
    def weight(self):
        """int: Weight of the edge."""
        return self._weight

    @property
    def qudits(self):
        """frozenset: Contains all qudits encompassed by edge."""
        return self._qudits

    @property
    def cardinality(self):
        """int: Number of qudits encompassed by edge."""
        return len(self._qudits)

    def same_qudits(self, edge):
        """
        Checks if an edge encompasses the same qudits.

        Args:
            edge (photongraph.Edge): An edge.

        Returns:
            bool: True if edge encompasses the same qudits.
        """

        assert isinstance(edge, Edge)

        return self._qudits == edge.qudits

    def add_weight(self, new_weight):
        """
        Adds a new weight to the edge's current weight modulo qudit dimension.

        Args:
            new_weight (int): A weight.

        """
        self._weight = (self._weight + new_weight) % self._qudit_dim

    def mul_weight(self, mul):
        """
        Multiplies edge's current weight by an integer modulo qudit dimension.

        Args:
            mul (int): Multiplier.

        Returns:

        """
        self._weight = (self._weight * mul) % self._qudit_dim

    def edge_union(self, edge):
        """
        Returns an edge with qudits from both edges with a weight which is the
        product of the two constituent edges.


        Args:
            edge:

        Returns:
            photongraph.Edge:

        Example:
            >>> e1 = Edge([0,1], 2,3)
            >>> e2 = Edge([2,3], 2,3)
            >>> e1.edge_union(e2)
            Edge([0, 1, 2, 3],1, 3)
        """
        assert edge.qudit_dim == self._qudit_dim, "Edge must have same qudit " \
                                                  "dimension"

        new_weight = self._weight*edge.weight % self._qudit_dim
        return Edge(self._qudits.union(edge.qudits), new_weight,
                    self._qudit_dim)