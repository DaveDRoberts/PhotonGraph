# -*- coding: utf-8 -*-
from .photonics.post_gsg import ParamCircuit, PostGSG, PostGSG4P4D
from .graphs import GraphState
from .graphs.gs_utils import gs_from_sv
from .states.statevector import StateVector
from . import utils
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

