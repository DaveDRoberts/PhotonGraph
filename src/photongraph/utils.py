import numpy as np
import itertools
import math
import scipy
import thewalrus.quantum as twq
from collections import defaultdict


def cartesian_product(*iterable):
    """
    Performs a Cartesian product between all the elements of an arbitrary
    number of iterables

    Args:
        iterable (any interable object):

    Returns:
        (list): Result of Cartesian product between iterables

    Examples:
    >>>cartesian_product([[1,0],[1,0]], [[1,0],[1,0]])
    [[[1, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, 1], [0, 1]]]

    >>>cartesian_product(['a','b'],['c','d'])
    [['a', 'c'], ['a', 'd'], ['b', 'c'], ['b', 'd']]

    """

    if not iterable:
        return [[]]
    else:
        return [[x] + p for x in iterable[0] for p in
                cartesian_product(*iterable[1:])]


def tensor(*matrices):
    """
    Function to calculate tensor product of multiple matrices.

    Args:
        matrices (numpy.ndarray): arbitary number of matrices

    Returns:
        (numpy.ndarray):

    Examples:
    >>>H = (1/np.sqrt(2)) * np.array([[1, 1],[1, -1]])
    >>>tensor(H,H)
    np.array([[ 0.5,  0.5,  0.5,  0.5],
              [ 0.5, -0.5,  0.5, -0.5],
              [ 0.5,  0.5, -0.5, -0.5],
              [ 0.5, -0.5, -0.5,  0.5]])

    """

    res = np.array([[1]])
    for i in matrices:
        res = np.kron(res, i)
    return res


def logical_basis(qudit_dim, qudit_num, rev=False):
    """
        Generates all of the logical basis states for a specific qudit
        dimension and number of qudits.

        Args:
            qudit_dim (int): Qudit dimension
            qudit_num (int): Number of qudits
            rev (bool): reverses the order of the binary bit strings

        Returns:
            (list): A list of tuples where each tuple is a basis state.
            Elements are in the canonical order.

        Examples:
        >>>logical_basis_states(2,2)
        [(0, 0), (0, 1), (1, 0), (1, 1)]

        >>>logical_basis_states(2,2, True)
        [(0, 0), (1, 0), (0, 1), (1, 1)]

        >>>logical_basis_states(4,1)
        [(0,), (1,), (2,), (3,)]
    """
    if rev:
        return [tuple(reversed(tuple(np.array(list(''.join(i)), dtype=int))))
                for i in itertools.product(''.join(str(i)
                        for i in np.arange(qudit_dim)), repeat=qudit_num)]
    else:
        return [tuple(np.array(list(''.join(i)), dtype=int))
                for i in itertools.product(''.join(str(i)
                        for i in np.arange(qudit_dim)), repeat=qudit_num)]


def logical_fock_states(qudit_dim, qudit_num, photon_cutoff=1):
    """
    Function to generate all the Fock states which correspond to each
    qudit state. The task is broken down into generating the logical Fock states
    for a single qudit and then taking the cartesian product between the logical
    Fock states for n number of qudits. This gives all the n-qubit logical Fock
    states.

    Since non-photon number resolving detectors are used in experiments Fock states
    which have more than one photon in a single mode are indiscernable from Fock
    states with only a single photon in a single mode. This higher-photon logical
    Fock states must be taken into account when considering photon counting statistics
    and graph state fidelity.


    Todo: check that all input args are integers greater than 0

    Todo: two functions which use most of the code in this function that will
          return logical fock states in different forms: dict and np.array.



    Args:
        qudit_num (int): Number of qudits
        qudit_dim (int): Qudit dimension
        photon_cutoff (int): Max number of photons per mode

    Returns:
        (dict): Key is a qudit state and its value is a list of logical Fock states
                which correspond to that qudit state.

    Examples:
    >>>logical_fock_states(2,2,2)
    {(0, 0): [(1, 0, 1, 0), (1, 0, 2, 0), (2, 0, 1, 0), (2, 0, 2, 0)],
     (0, 1): [(1, 0, 0, 1), (1, 0, 0, 2), (2, 0, 0, 1), (2, 0, 0, 2)],
     (1, 0): [(0, 1, 1, 0), (0, 1, 2, 0), (0, 2, 1, 0), (0, 2, 2, 0)],
     (1, 1): [(0, 1, 0, 1), (0, 1, 0, 2), (0, 2, 0, 1), (0, 2, 0, 2)]}

    """

    def check_input(param):
        """Checks if param is a non-zero integer."""
        assert isinstance(param, int)
        assert param > 0

    check_input(qudit_dim)
    check_input(qudit_num)
    check_input(photon_cutoff)

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

    ind_fock_state_combos = cartesian_product(
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


def logical_fock_states_lists(qudit_dim, qudit_num):
    """

    Args:
        qudit_dim:
        qudit_num:

    Returns:
        tuple(np.array, np.array):
    """
    lfs = logical_fock_states(qudit_dim, qudit_num)
    qudit_basis_states = []
    fock_states = []
    for qds, fs in lfs.items():
        qudit_basis_states.append(qds)
        fock_states.append(fs[0])

    return qudit_basis_states, fock_states


def qudit_qubit_encoding(qudit_dim, qudit_num):
    """
    Generates the mapping between qudit and qubit states.
    In order to map qudit states to qubit states qudit dimension
    must be a power of 2.

    Args:
        qudit_num (int): Number of qudits
        qudit_dim (int): Qudit dimension

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

    try:
        assert (qudit_dim & (qudit_dim - 1) == 0) and qudit_dim != 0
    except:
        raise AssertionError('Qudit dimension must be a power of 2.')

    basis_state_num = qudit_dim ** qudit_num
    qubit_num = int(math.log2(qudit_dim ** qudit_num))
    qudit_logical_basis = logical_basis(qudit_dim, qudit_num)
    qubit_logical_basis = logical_basis(2, qubit_num)

    qudit_to_qubit_map = {qudit_logical_basis[i]: qubit_logical_basis[i] for i
                          in range(basis_state_num)}

    return qudit_to_qubit_map


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
    for ls in itertools.product(*[range(k) for k in photon_occ]):
        binomials = 1
        for i, k in enumerate(photon_occ):
            binomials *= int(scipy.special.binom(k, ls[i]))
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


def calc_coin_rate(cov_matrix, qudit_state, qudit_fock_map, loss_params,
                   units="n"):
    """
    Calculates the estimated coincidence rate for a particular qudit Fock state.

    TODO: Format outputs with appropriate units

    Args:
        cov_matrix (numpy.ndarray):
        qudit_state (tuple):
        qudit_fock_map (dict):
        loss_params (dict):
        units (str):

    Returns:
        float: coincidence rate in chosen units
    Examples:


    """

    # determine efficiency for each mode
    eta = efficiency_calc(loss_params)
    UOC_pulse_rate = 0.5 * 10 ** 9

    coin_prob = 0
    for fock_state in qudit_fock_map[qudit_state]:
        num_of_modes = len(fock_state)
        prob_amp = twq.pure_state_amplitude(
            np.zeros(2 * num_of_modes), cov_matrix, fock_state)
        prob = (np.abs(prob_amp)) ** 2
        photon_occ = [fock_state[i] for i in np.array(fock_state).nonzero()[0]]
        scaled_prob = prob * efficiency_scale_factor(photon_occ, eta)
        coin_prob += scaled_prob

    coin_rate = UOC_pulse_rate * coin_prob

    if units == "s":
        return round(coin_rate, 3)
    elif units == "m":
        return round(coin_rate * 60, 2)
    elif units == "h":
        return round(coin_rate * 3600, 2)
    elif units == "d":
        return round(coin_rate * 3600 * 24, 2)
    else:
        return coin_rate


def intra_qubit_gate_set(qudit_dim):
    """
    Generate the logical qubit operations available for qubits encoded
    in a particular qudit dimension.

    Args:
        qudit_dim (int): Qudit dimension

    Returns:
        dict: key=gate label, value=matrix (numpy.ndarray())

    TODO: Include CNOT, CCNOT etc.

    Examples:

    """
    try:
        assert (qudit_dim & (qudit_dim - 1) == 0) and qudit_dim != 0
    except:
        raise AssertionError('Qudit dimension must be a power of 2.')


    qubit_num = int(np.log2(qudit_dim))
    qubits = range(qubit_num)

    # Single-qubit unitary operations
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    Y = 1j * X @ Z
    S = np.array([[1, 0], [0, 1j]])
    HS = H @ S
    SH = S @ H
    HSH = H @ S @ H
    si_Z = np.sqrt(1j) * S
    si_X = np.sqrt(-1j) * HSH
    T = np.array([[1, 0], [0, np.exp(0.25*np.pi*1j)]])
    T_inv = T.conj().T
    X_fr = H @ T @ H
    X_fr_inv = X_fr.conj().T
    qubit_gate_set = {"H": H, "I": I, "X": X, "Z": Z, "Y": Y, "S": S, "HS": HS,
                      "HSH": HSH, "SH": SH, "si_Z": si_Z, "si_X": si_X, "T": T,
                      "T_inv": T_inv, "X_fr_inv": X_fr_inv, "X_fr": X_fr}

    binary_bit_strings = logical_basis(2, qubit_num)

    for i in range(2, qubit_num + 1):
        gate_qubit_combos = list(itertools.combinations(qubits, i))
        for gqc in gate_qubit_combos:
            gate = np.eye(qudit_dim)

            for j, bit_str in enumerate(binary_bit_strings):
                if sum([bit_str[qb] for qb in gqc]) == i:
                    gate[j][j] = -1

            gate_name = "C" * (i - 1) + "Z_" + "".join(np.array(gqc, dtype=str))
            qubit_gate_set[gate_name] = gate

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

    Examples:


    """

    # check that the qudit dimension is a power of 2

    # generate qubit gate set
    qgs = intra_qubit_gate_set(qudit_dim)

    qudit_LU = np.eye(qudit_dim)

    for qubit_gates in qubit_gate_columns:
        qubit_gates_matrices = [qgs[qg] for qg in qubit_gates]
        qudit_LU = np.matmul(tensor(*qubit_gates_matrices), qudit_LU)

    return qudit_LU


def sort_tuples_by_ele(groups, n):
    """
    Takes in a list of tuples and sorts them by the value of the nth element
    in each tuple.

    Args:
        groups (list):

    Returns:
        list:

    Examples:

    """
    # get length of list of groups (tuples)
    lst = len(groups)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if (groups[j][n] > groups[j + 1][n]):
                temp = groups[j]
                groups[j] = groups[j + 1]
                groups[j + 1] = temp
    return groups


def common_member(a, b):
    """
    Checks if two iterables contain at least one common element, returns
    true if they do.

    Args:
        a:
        b:

    Returns:

    Examples:

    """
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False


