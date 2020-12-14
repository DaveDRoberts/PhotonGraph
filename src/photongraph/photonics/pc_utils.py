import strawberryfields as sf
import numpy as np
import scipy as sp
import itertools as it
import math
from collections import defaultdict

from ..utils import check_integer, basis_matrix, controlled_qubit_gates, tensor


def param_inter(prog, q, label=""):
    """
    This function needs to generate an interferometer according to the Clements
    scheme decomposition and adds it to a SF program.

    Args:
        prog (sf.Program): Strawberry fields program.
        q (list): Quantum modes.
        label (str): Parameter label pre-fix.

    """

    q = sorted(q)

    N = len(q)

    diag_num = [i for i in range(1, N, 2)] + [i for i in
                                              range(N - 2 + N % 2, 0, -2)]
    diag_start_modes = [i for i in range(0, N - (N % 2), 2)] + [N - 2] * (
                (N - 2 + N % 2) // 2)

    u = 0
    for i in range(N - 1):
        for j in range(0, diag_num[i], 1):
            d = diag_start_modes[i] - j
            theta = prog.params(label + '_theta_{}'.format(u))
            phi = prog.params(label + '_phi_{}'.format(u))
            prog.append(sf.ops.MZgate(np.mod(theta, 2 * np.pi),
                                   np.mod(phi, 2 * np.pi)),
                        (q[0] + d, q[0] + d + 1))
            u += 1

    for i in range(N):
        gamma = prog.params(label + '_gamma_{}'.format(i))
        prog.append(sf.ops.Rgate(gamma), (i + q[0],))


def decomp_inter(U, label=""):
    """
    Decomposes U into a set of phases according top the rectangular-symmetric
    decomposition.

    Args:
        U (np.ndarray): Unitary matrix
        label (str): Parameter label pre-fix.

    Returns:
        dict: Parameter labels and values.

    """

    decomp = sf.decompositions.rectangular_symmetric(U)
    params = {}

    for i, uc_phases in enumerate(decomp[0]):
        params[label + "_theta_{}".format(i)] = uc_phases[2]
        params[label + "_phi_{}".format(i)] = uc_phases[3]

    for i, end_phase in enumerate(decomp[1]):
        params[label + "_gamma_{}".format(i)] = np.angle(end_phase)

    return params


def logical_fock_states(qudit_dim, qudit_num, photon_cutoff=1):
    """
    Function to generate all the Fock states which correspond to each
    qudit state - one photon per qudit.

    The task is broken down into generating the logical Fock states for a
    single qudit and then taking the cartesian product between the logical
    Fock states for n number of qudits. This gives all the n-qubit logical Fock
    states.

    Since non-photon number resolving detectors are used in experiments Fock
    states which have more than one photon in a single mode are indiscernable
    from Fock states with only a single photon in a single mode. These
    multi-photon logical Fock states must be taken into account when
    considering photon counting statistics and graph state fidelity.

    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits
        photon_cutoff (int): Max number of photons per mode

    Returns:
        dict: Key is a qudit state and its value is a list of logical Fock
                states which correspond to that qudit state.

    Examples:
    >>>logical_fock_states(2,2,2)
    {(0, 0): [(1, 0, 1, 0), (1, 0, 2, 0), (2, 0, 1, 0), (2, 0, 2, 0)],
     (0, 1): [(1, 0, 0, 1), (1, 0, 0, 2), (2, 0, 0, 1), (2, 0, 0, 2)],
     (1, 0): [(0, 1, 1, 0), (0, 1, 2, 0), (0, 2, 1, 0), (0, 2, 2, 0)],
     (1, 1): [(0, 1, 0, 1), (0, 1, 0, 2), (0, 2, 0, 1), (0, 2, 0, 2)]}

    """

    check_integer(qudit_dim, 0)
    check_integer(qudit_num, 0)
    check_integer(photon_cutoff, 0)

    single_qudit_fock_map = {}

    for i in range(qudit_dim):
        fock_states = []
        for j in range(1, photon_cutoff + 1):
            s = list(np.zeros(qudit_dim, dtype=int))
            s[i] = j
            fock_states.append(tuple(s))

        single_qudit_fock_map[i] = tuple(fock_states)

    fock_single_qudit_map = dict(
        (tuple(vl), k) for k, v in single_qudit_fock_map.items() for vl in v)

    ind_fock_state_combos = it.product(
        *[[k for j in single_qudit_fock_map.values() for k in j] for i in
          range(qudit_num)])

    qudit_fock_map = defaultdict(list)
    for ind_fock_state_combo in ind_fock_state_combos:
        qudit_state = tuple(
            fock_single_qudit_map[tuple(ind_fock_state)] for ind_fock_state in
            ind_fock_state_combo)
        qudit_fock_map[qudit_state].append(
            tuple(np.ndarray.flatten(np.array(ind_fock_state_combo))))

    return dict(qudit_fock_map)


def qudit_qubit_encoding(qudit_dim, qudit_num):
    """
    Generates the mapping between qudit and qubit states.

    In order to map qudit states to qubit states qudit dimension
    must be a power of 2.

    Args:
        qudit_dim (int): Qudit dimension
        qudit_num (int): Number of qudits


    Returns:
        (dict): Key is a qudit state and value is its corresponding qubit state

    Examples:
    >>>qudit_qubit_encoding(4,2)
    {(0, 0): (0, 0, 0, 0),
     (0, 1): (0, 0, 0, 1),
     (0, 2): (0, 0, 1, 0),
     (0, 3): (0, 0, 1, 1),
     (1, 0): (0, 1, 0, 0),
     (1, 1): (0, 1, 0, 1),
     (1, 2): (0, 1, 1, 0),
     (1, 3): (0, 1, 1, 1),
     (2, 0): (1, 0, 0, 0),
     (2, 1): (1, 0, 0, 1),
     (2, 2): (1, 0, 1, 0),
     (2, 3): (1, 0, 1, 1),
     (3, 0): (1, 1, 0, 0),
     (3, 1): (1, 1, 0, 1),
     (3, 2): (1, 1, 1, 0),
     (3, 3): (1, 1, 1, 1)}
    """

    assert (qudit_dim & (qudit_dim - 1) == 0) and qudit_dim != 0, \
        'Qudit dimension must be a power of 2.'

    check_integer(qudit_num, 0)

    basis_state_num = qudit_dim ** qudit_num
    qubit_num = int(math.log2(qudit_dim ** qudit_num))
    qudit_lb = basis_matrix(qudit_dim, qudit_num)
    qubit_lb = basis_matrix(2, qubit_num)

    qudit_to_qubit_map = {tuple(qudit_lb[i]): tuple(qubit_lb[i])
                          for i in range(basis_state_num)}

    return qudit_to_qubit_map


def intra_qubit_gate_set(qudit_dim):
    """
    Generate the logical qubit operations available for qubits encoded
    in a particular qudit dimension.

    Args:
        qudit_dim (int): Qudit dimension

    Returns:
        dict: key=gate label, value=matrix (numpy.ndarray())

    """
    assert (qudit_dim & (qudit_dim - 1) == 0) and qudit_dim != 0, \
        'Qudit dimension must be a power of 2.'

    qubit_num = int(np.log2(qudit_dim))
    qubits = range(qubit_num)

    # Single-qubit unitary operations
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Y = 1j * X @ Z
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    HS = H @ S
    SH = S @ H
    HSH = H @ S @ H
    si_Z = np.sqrt(1j) * S
    si_X = np.sqrt(-1j) * HSH
    T = np.array([[1, 0], [0, np.exp(0.25 * np.pi * 1j)]])
    T_inv = T.conj().T
    X_fr = H @ T @ H
    X_fr_inv = X_fr.conj().T

    qubit_gate_set = {"H": H, "I": I, "X": X, "Z": Z, "Y": Y, "S": S, "HS": HS,
                      "HSH": HSH, "SH": SH, "si_Z": si_Z, "si_X": si_X, "T": T,
                      "T_inv": T_inv, "X_fr_inv": X_fr_inv, "X_fr": X_fr}

    # add control gates
    CX_gates = controlled_qubit_gates(qubit_num, X, 'X')
    CZ_gates = controlled_qubit_gates(qubit_num, Z, 'Z')
    CS_gates = controlled_qubit_gates(qubit_num, S, 'S')

    CU_gates_list = [CX_gates, CZ_gates, CS_gates]

    for CU_gates in CU_gates_list:
        for gate_label, gate_U in CU_gates.items():
            qubit_gate_set[gate_label] = gate_U

    return qubit_gate_set


def compile_qudit_LU(qudit_dim, *qubit_gate_columns):
    """
    Function compiles the unitary operation which must be applied to the modes
    of a qudit such that the specified quantum gates are applied to its
    constituent qubits.

    Args:
        qudit_dim (int): Qudit dimension, this should be a power of two so that qudit
        states can be simply mapped to qubit states.
        qubit_gates (list): This is just a list of strings, each specifies the gates
        to be applied. Each list may contain between 1 and number of qubits of elements.

        e.g. if qudit_dim=4 then this maps to 2 qubits, each set of qubit gates can either
        be two single-qubit gates or one two-qubit gate.
        e.g. if qudit_dim=8 then this maps to 3 qubits and we can have 3 single-qubit gates,
        1 single-qubit gate and a two-qubit gate or 1 three-qubit gate.

    Returns:
        numpy.ndarray: Returns the compiled unitary matrix

    """

    # check that the qudit dimension is a power of 2

    # generate qubit gate set
    qgs = intra_qubit_gate_set(qudit_dim)

    qudit_LU = np.eye(qudit_dim)

    for qubit_gates in qubit_gate_columns:
        qubit_gates_matrices = [qgs[qg] for qg in qubit_gates]
        qudit_LU = np.matmul(tensor(*qubit_gates_matrices), qudit_LU)

    return qudit_LU


def efficiency_scale_factor(photon_occ, eta):
    """
    This function calculates the scale factor which scales the coincidence
    detection probability for a particular photon occupation of modes.

    This code is based off the formula on p81 of Generating Optical
    Graph States by JC Adcock.

    The motivation for this scale factor is that multi-photon Fock states
    are indiscernible from Fock states with a single photon per mode
    (due to the use of non-photon-number resolving detectors).
    We don't require the entire Fock state, only the modes which when
    occupied with at least one photon correspond to a logical state.

    Args:
        photon_occ (iterable): Contains the number of photons in each mode
                               of interest.
        eta (float): Detection efficiency for each mode

    Returns:
        float: Real, scalar value

    Examples:

    >>>efficiency_scale_factor([1,1,1,1],0.96)
    0.8493465599999999
    >>>efficiency_scale_factor([2,2,1,1],0.96)
    0.918653239296
    >>>efficiency_scale_factor([2,2,2,2],0.96)
    0.9936153436225534
    """

    gamma = 0
    for ls in it.product(*[range(k) for k in photon_occ]):
        binomials = 1
        for i, k in enumerate(photon_occ):
            binomials *= int(sp.special.binom(k, ls[i]))
            if not binomials:
                print(binomials)
        one_min_eta_pow = sum(ls)
        eta_pow = sum(photon_occ) - one_min_eta_pow
        gamma += binomials * (eta ** eta_pow) * ((1 - eta) ** one_min_eta_pow)
    return gamma


def loss_dB_to_eff(loss_dB):
    """
    Converts loss in dB to a positive efficiency value

    Args:
        loss_dB(float): loss in dB

    Returns:
        float: efficiency 0<=eff<=1

    Examples:
        >>>loss_dB_to_eff(3)
        0.5011872336272722
        >>>loss_dB_to_eff(6)
        0.25118864315095796
        >>>loss_dB_to_eff(0)
        1
    """
    return 10 ** (-0.1 * loss_dB)


def efficiency_calc(loss_params):
    """
    Calculates the efficiency for a photon to be detected from generation,
    through propagation to detection.

    Args:
        loss_params (dict): Each item of the dict has key:value pairs of the
                            form str(param_name): param_name dict e.g.

    Returns:
        (float): total efficiency

    Examples:


    """

    component_losses = loss_params["component_losses"]
    component_numbers = loss_params["component_numbers"]
    propagation_losses = loss_params["propagation_losses"]
    propagation_lengths = loss_params["propagation_lengths"]
    det_efficiency = loss_params["det_efficiency"]

    # calculate total component losses
    tot_comp_loss = sum(
        component_losses[comp] * component_numbers[comp] for comp in
        component_losses.keys())
    # calculate total propagation losses
    tot_prop_loss = sum(
        propagation_losses[prop] * propagation_lengths[prop] for prop in
        propagation_losses.keys())
    # calculate total efficiency
    eta = det_efficiency * loss_dB_to_eff(tot_comp_loss + tot_prop_loss)

    return eta