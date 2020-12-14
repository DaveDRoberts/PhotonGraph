import numpy as np
import strawberryfields as sf
import thewalrus.quantum as twq

from .param_circuit import ParamCircuit
from ..states.statevector import StateVector
from .pc_utils import param_inter, decomp_inter, basis_matrix, \
    logical_fock_states, efficiency_calc, efficiency_scale_factor


class PostGSG(ParamCircuit):
    """
    Postselected graph state generator.


    """

    def __init__(self, qudit_num, qudit_dim):
        """

        Args:
            qudit_num:
            qudit_dim:
        """
        super().__init__(qudit_num * qudit_dim)
        self._qudit_num = qudit_num
        self._qudit_dim = qudit_dim
        self._qubit_num = int(np.log2(qudit_dim ** qudit_num))
        self._prog = self._program()
        self._init_param_vals()

    def __repr__(self):
        return 'ParamCircuit({})'.format(self._mode_num)

    def __str__(self):
        return 'ParamCircuit({})'.format(self._mode_num)

    def _program(self):
        """
        Generates the parametrised SF program for a postselected graph state
        generator photonic circuit.

        """
        n = self._qudit_num
        d = self._qudit_dim

        prog = sf.Program(self._mode_num)

        # add photon-pair sources
        for j in range(n // 2):
            for i in range(d):
                tmsv_r = prog.params('tmsv_r_{}'.format(d*j + i))
                tmsv_phi = prog.params('tmsv_phi_{}'.format(d*j + i))
                prog.append(sf.ops.S2gate(r=tmsv_r, phi=tmsv_phi),
                            [d * 2 * j + i, d * 2 * j + i + d])

        # add pre-fusion local unitary operations
        for i in range(n):
            param_inter(prog, np.arange(d * i, d * (i + 1), dtype=int),
                        'preFLU_{}'.format(chr(ord('@')+i+1)))

        # add fusion MZIs
        self._fusions(prog)

        # add post-fusion local unitary operations
        for i in range(n):
            param_inter(prog, np.arange(d * i, d * (i + 1), dtype=int),
                        'postFLU_{}'.format(chr(ord('@')+i+1)))

        return prog

    def _fusions(self):
        """
        Adds fusion MZIs to program. This needs to be overidden since the
        locations of these don't follow a consistent pattern.

        """
        pass

    def _init_param_vals(self):
        """
        Initialises circuit parameters.

        Default parameter values:
            - All sources are turned ON with TMSV parameter (r=0.2, phi=0.0)
            - All fusions are turned OFF, set MZI to BAR i.e. theta=pi
            - All local unitary interferometers are set to identity.

        """

        free_params = self._prog.free_params

        # set all sources to be on, with the same TMSV param
        for param_id in free_params.keys():

            if 'tmsv_r' in param_id:
                self._param_vals[param_id] = 0.2
            if 'tmsv_phi' in param_id:
                self._param_vals[param_id] = 0.0
            elif "fusion" in param_id:
                self._param_vals[param_id] = np.pi

        n = self._qudit_num

        for i in range(n):
            decomp = decomp_inter(np.eye(n),
                                  'preFLU_{}'.format(chr(ord('@') + i + 1)))
            for param_id, param_val in decomp.items():
                self._param_vals[param_id] = param_val

        for i in range(n):
            decomp = decomp_inter(np.eye(n),
                                  'postFLU_{}'.format(chr(ord('@') + i + 1)))
            for param_id, param_val in decomp.items():
                self._param_vals[param_id] = param_val

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

    @property
    def program(self):
        """                 """
        return self._prog

    def run(self, encoding='qubit'):
        """
        Determines the postselected logical output state using the covariance
        matrix.

        Args:
            encoding (str): Either native qudit encoding or qubit one.

        Returns:
            StateVector: Logical postselected output state vector.
        """

        assert encoding in ['qudit', 'qubit'], \
            "Encoding must be either 'qudit' or 'qubit'."

        d = self._qudit_dim
        n_qd = self._qudit_num
        n_qb = self._qubit_num
        m = self._mode_num
        res = self._run(self._prog, backend='gaussian')
        cov_matrix = res.state.cov()

        basis_mat = basis_matrix(d, n_qd)
        lfs = logical_fock_states(d, n_qd)
        fock_states = [lfs[tuple(basis_state)][0] for basis_state in basis_mat]

        amps = list(map(lambda fs: twq.pure_state_amplitude(
            np.zeros(2 * m), cov_matrix, fs), fock_states))

        vector = np.round(np.array(amps, dtype=np.complex128), 12)

        if encoding == 'qudit':
            state_vector = StateVector(n_qd, d, vector)
            state_vector.normalize()
            return state_vector
        elif encoding == 'qubit':
            state_vector = StateVector(n_qb, 2, vector)
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

        res = self._run(self._prog, backend='gaussian')
        cov_matrix = res.state.cov()

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
                np.zeros(2 * mode_num), cov_matrix, fock_state)
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


class PostGSG4P4D(PostGSG):
    """
    Postselected graph state generator with 4 qudits (photons) and qudit
    dimension of 4.

    """

    def __init__(self):
        super().__init__(4, 4)

    def _init_param_vals(self):
        """
        Initialises circuit parameters.

        Default parameter values:
            - The first and last source of each qudit-pair generator are turned
              ON with TMSV parameter (r=0.2, phi=0.0) i.e. sources 0 and 3, and
              sources 4 and 7.
            - Only fusion_B0_C0 is turned on
            - All local unitary interferometers are set to identity.
            - The postselected logical state is 8-qubit GHZ state.

        """

        free_params = self._prog.free_params

        for param_id in free_params.keys():
            if param_id in ['tmsv_r_0', 'tmsv_r_3', 'tmsv_r_4', 'tmsv_r_7']:
                self._param_vals[param_id] = 0.2
            elif param_id in ['tmsv_r_1', 'tmsv_r_2', 'tmsv_r_5', 'tmsv_r_6']:
                self._param_vals[param_id] = 0.0
            elif "tmsv_phi" in param_id:
                self._param_vals[param_id] = 0.0
            elif param_id in ['fusion_B0_C0']:
                self._param_vals[param_id] = 0.0
            elif param_id in ['fusion_B1_C1', 'fusion_A0_D0', 'fusion_A2_D2']:
                self._param_vals[param_id] = np.pi

        n = self._qudit_num

        for i in range(n):
            decomp = decomp_inter(np.eye(n),
                                  'preFLU_{}'.format(chr(ord('@') + i + 1)))
            for param_id, param_val in decomp.items():
                self._param_vals[param_id] = param_val

        for i in range(n):
            decomp = decomp_inter(np.eye(n),
                                  'postFLU_{}'.format(chr(ord('@') + i + 1)))
            for param_id, param_val in decomp.items():
                self._param_vals[param_id] = param_val

    def _fusions(self, prog):
        """
        Add fusion MZIs between specific modes for postselected fusion.

        Args:
            prog (sf.Program): Strawberry Fields program.

        """
        fusion_B0_C0 = prog.params('fusion_B0_C0')
        fusion_B1_C1 = prog.params('fusion_B1_C1')
        fusion_A0_D0 = prog.params('fusion_A0_D0')
        fusion_A2_D2 = prog.params('fusion_A2_D2')

        prog.append(sf.ops.MZgate(phi_in=fusion_B0_C0, phi_ex=0),
                    (4, 8))
        prog.append(sf.ops.MZgate(phi_in=fusion_B1_C1, phi_ex=0),
                    (5, 9))
        prog.append(sf.ops.MZgate(phi_in=fusion_A0_D0, phi_ex=0),
                    (0, 12))
        prog.append(sf.ops.MZgate(phi_in=fusion_A2_D2, phi_ex=0),
                    (2, 14))
