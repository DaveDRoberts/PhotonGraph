import strawberryfields as sf
import thewalrus.quantum as twq
import numpy as np
from ..utils import sort_tuples_by_ele, common_member, efficiency_calc, \
    efficiency_scale_factor, logical_fock_states, basis_matrix
from .ops import PPS, Inter
from ..states.statevector import StateVector


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
            group (str): Identifies a group of operators.

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

    def __build_op_reg(self):
        """
        Builds the operator register when an instance is created.

        """

        n = self._qudit_num
        d = self._qudit_dim

        self._op_reg[('sources', 0)] = []
        for j in range(n // 2):
            for i in range(d):
                self._op_reg[('sources', 0)].append(PPS((d*2*j+i, d*2*j+i+d)))

        self._op_reg[('preFLU', 1)] = []
        for i in range(n):
            self._op_reg[('preFLU', 1)].append(
                Inter(np.arange(d*i, d*(i+1), dtype=int), np.eye(d)))

        self._op_reg[('fusions', 2)] = []

        self._op_reg[('postFLU', 3)] = []
        for i in range(n):
            self._op_reg[('postFLU', 3)].append(
                Inter(np.arange(d*i, d*(i+1), dtype=int), np.eye(d)))

    @property
    def qudit_num(self):
        """int: Number of qudits."""
        return self._qudit_num

    @property
    def qubit_num(self):
        """int: Number of qubits."""
        return self._qubit_num

    @property
    def qudit_dim(self):
        """int: Dimension of qudits."""
        return self._qudit_dim

    def run(self, encoding='qubit'):
        """
        Determines the logical output state using the covariance matrix.

        Args:
            encoding (str): Either native qudit encoding or qubit one.

        Returns:
            StateVector: Logical postselected output state vector.
        """
        assert self._compiled
        assert encoding in ['qudit', 'qubit'], \
            "Encoding must be either 'qudit' or 'qubit'."

        qudit_dim = self._qudit_dim
        qudit_num = self._qudit_num
        mode_num = self._mode_num
        cov_matrix = self._cov_matrix

        basis_mat = basis_matrix(qudit_dim, qudit_num)
        lfs = logical_fock_states(qudit_dim, qudit_num)
        fock_states = [lfs[tuple(basis_state)][0] for basis_state in basis_mat]

        amps = list(map(lambda fs: twq.pure_state_amplitude(
            np.zeros(2 * mode_num), cov_matrix, fs), fock_states))

        vector = np.array(amps, dtype='complex128')

        if encoding == 'qudit':
            state_vector = StateVector(qudit_num, qudit_dim, vector)
            state_vector.normalize()
            return state_vector
        elif encoding == 'qubit':
            qubit_num = int(np.log2(qudit_dim ** qudit_num))
            qubit_dim = 2
            state_vector = StateVector(qubit_num, qubit_dim, vector)
            state_vector.normalize()
            return state_vector

    def coincidence_rate(self, loss_params, fock_states=(),
                         pulse_rate=0.5*10**9, units="s"):
        """
        Calculates the m-fold coincidence rate for a collection of
        Fock states which have at least one photon in each subset
        of modes that corresponds to a qudit.

        The concidence rate is given by the sum of Fock state probabilities
        multiplied by the pulse rate of the laser. The Fock states can be
        specifed instead - this can be a much faster way of determining the
        m-fold coincidence rate if there are only a few with a non-zero
        amplitude.

        Args:
            loss_params (dict): Loss parameters.
            fock_states (iterable): Fock states to calculate prob amps.
            pulse_rate (float): Pulse rate of laser in Hz.
            units (str): Rate units for coincidence detection.

        Returns:
            string: Formatted string of coincidence rate.

        """
        assert self.compiled, "Circuit must be compiled first."

        qudit_dim = self._qudit_dim
        qudit_num = self._qudit_num
        mode_num = self._mode_num
        eta = efficiency_calc(loss_params)

        if not fock_states:
            basis_mat = basis_matrix(qudit_dim, qudit_num)
            lfs = logical_fock_states(qudit_dim, qudit_num)
            fock_states = [lfs[tuple(basis_state)][0] for basis_state in
                           basis_mat]

        coin_prob = 0.0
        for fock_state in fock_states:
            prob_amp = twq.pure_state_amplitude(
                np.zeros(2 * mode_num), self._cov_matrix, fock_state)
            prob = (np.abs(prob_amp)) ** 2

            photon_occ = [fock_state[i] for i in
                          np.array(fock_state).nonzero()[0]]
            scaled_prob = prob * efficiency_scale_factor(photon_occ, eta)
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