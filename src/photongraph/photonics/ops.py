import abc
import numpy as np
from strawberryfields.ops import BSgate, MZgate, Interferometer, Rgate, S2gate


class Op(abc.ABC):
    """
    An abstract base class which serves as a template for photonic operators.
    Utilises the Strawberry Fields (SF) photonic operators.

    """
    def __init__(self, modes):
        self._modes = tuple(sorted(modes))

    @abc.abstractmethod
    def update(self, **op_params):
        """Only implemented in subclasses."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sf_op(self):
        """Only implemented in subclasses."""
        raise NotImplementedError()

    @property
    def modes(self):
        """tuple: contains modes that operator acts on in ascending order."""
        return self._modes


class BS(Op):
    """
    Beamsplitter operator

    """

    def __init__(self, modes, theta, phi):
        """

        Args:
            modes:
            theta:
            phi:
        """

        self._sf_params = {'theta': theta, 'phi':phi}
        super().__init__(modes)

    def sf_op(self):
        """

        Returns:

        """
        theta = self._sf_params['theta']
        phi = self._sf_params['phi']
        return BSgate(theta=theta, phi=phi)

    @property
    def sf_params(self):
        """dict: Contains parameters for SF operator."""
        return self._sf_params

    def update(self, **op_params):
        """

        Args:
            **op_params:

        Returns:

        """
        if "modes" in op_params.keys():
            # performs some checks here
            self._modes = op_params["modes"]
        if "sf_params" in op_params.keys():
            # ensure that parameters are appropriate
            self._sf_params = op_params["sf_params"]


class PS(Op):
    """


    """

    def __init__(self, modes, theta):
        """

        Args:
            modes:
            theta:

        """

        self._sf_params = {'theta': theta}
        super().__init__(modes)

    def sf_op(self):
        """

        Returns:

        """
        theta = self._sf_params['theta']
        return Rgate(theta=theta)

    @property
    def sf_params(self):
        return self._sf_params

    def update(self, **op_params):
        """

        Args:
            **op_params:

        Returns:

        """
        if "modes" in op_params.keys():
            # performs some checks here
            self._modes = op_params["modes"]
        if "sf_params" in op_params.keys():
            # ensure that parameters are appropriate
            self._sf_params = op_params["sf_params"]


class MZI(Op):
    """
    params for this object should be reflectivity phase and BS convention
    Want to use the SF gate ops

    """

    def __init__(self, modes, phi_in, phi_ex):
        """

        Args:
            modes:
            theta:
            phi:
        """

        self._sf_params = {'phi_in': phi_in, 'phi_ex': phi_ex}
        super().__init__(modes)

    def sf_op(self):
        """

        Returns:

        """
        phi_in = self._sf_params['phi_in']
        phi_ex = self._sf_params['phi_ex']
        return MZgate(phi_in=phi_in, phi_ex=phi_ex)

    @property
    def sf_params(self):
        return self._sf_params

    def update(self, **op_params):
        """

        Args:
            **op_params:

        Returns:

        """
        if "modes" in op_params.keys():
            # performs some checks here
            self._modes = op_params["modes"]
        if "sf_params" in op_params.keys():
            # ensure that parameters are appropriate
            self._sf_params = op_params["sf_params"]


class PPS(Op):
    """
    params for this object should be reflectivity phase and BS convention
    Want to use the SF gate ops

    """

    def __init__(self, modes, tmsv_param=(0.2, 0.0*np.pi)):
        """

        Args:
            modes:
            theta:
            phi:
        """

        self._sf_params = {'tmsv_param': tmsv_param}
        super().__init__(modes)

    def sf_op(self):
        """

        Returns:

        """
        tmsv_param = self._sf_params['tmsv_param']

        return S2gate(r=tmsv_param[0], phi=tmsv_param[1])

    @property
    def sf_params(self):
        return self._sf_params

    def update(self, **op_params):
        """

        Args:
            **op_params:

        Returns:

        """
        if "modes" in op_params.keys():
            # performs some checks here
            self._modes = op_params["modes"]
        if "sf_params" in op_params.keys():
            # ensure that parameters are appropriate
            self._sf_params = op_params["sf_params"]


class Inter(Op):
    """
    params for this object should be reflectivity phase and BS convention
    Want to use the SF gate ops

    """

    def __init__(self, modes, U):
        """

        Args:
            modes:
            U (numpy.ndarray):
        """

        self._sf_params = {'U': U}
        super().__init__(modes)

    def sf_op(self):
        """

        Returns:

        """
        U = self._sf_params['U']
        return Interferometer(U=U, mesh='rectangular_symmetric')

    @property
    def sf_params(self):
        return self._sf_params

    def update(self, **op_params):
        """

        Args:
            **op_params:

        Returns:

        """
        if "modes" in op_params.keys():
            # performs some checks here
            self._modes = op_params["modes"]
        if "sf_params" in op_params.keys():
            # ensure that parameters are appropriate
            self._sf_params = op_params["sf_params"]


class Fusion(MZI):
    """
    A MZI set to swap optical modes.

    Swaps modes to "push" particular logical Fock states out of
    postselection. In certain situations this has the effective performing a
    fusion operation under postselection.

    """
    def __init__(self, modes, status):
        """

        Args:
            modes (iterable): Contains numbers of modes on which to act the op.
            status (bool):
        """
        phase = int(not status)*np.pi
        super().__init__(modes, phi_in=phase, phi_ex=0.0)



