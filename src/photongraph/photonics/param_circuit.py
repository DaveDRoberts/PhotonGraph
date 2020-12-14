import abc
import strawberryfields as sf


class ParamCircuit(abc.ABC):
    """
    Parametrized Photonic Circuit.

    Convenience wrapper for parametrizing SF programs.

    """
    def __init__(self, mode_num):
        """

        Args:
            mode_num (int): Number of circuit modes.

        """
        self._mode_num = mode_num
        self._param_vals = {}

    @abc.abstractmethod
    def _program(self):
        """
        Generates a parametrised SF program.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _init_param_vals(self):
        """
        Initialises circuit parameters.

        """
        raise NotImplementedError()

    def _run(self, prog, backend, backend_options={}):
        """

        """
        assert backend in ["fock", 'gaussian', 'tf']

        eng = sf.Engine(backend=backend, backend_options=backend_options)

        return eng.run(prog, args=self._param_vals)

    def update_params(self, params):
        """
        Updates program parameter values.

        Args:
            params (dict):

        """
        # Todo: This assert fails for some, unknown reason
        #print(set(self._param_vals.keys()))
        #assert set(params.keys()) in set(self._param_vals.keys()), \
        #    "Invalid parameter(s)."

        for param_id, param_val in params.items():
            self._param_vals[param_id] = param_val

    @abc.abstractmethod
    def run(self):
        """"""
        raise NotImplementedError()

    @property
    def params(self):
        """"""
        return self._param_vals

    @property
    def mode_num(self):
        """"""
        return self._mode_num



