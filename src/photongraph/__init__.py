# -*- coding: utf-8 -*-
from .photonics.circuit import Circuit, PostGSG
from .photonics.ops import BS, PS, Inter, Fusion, MZI, PPS
from .graphs import GraphState, StateVector
from .graphs.gs_utils import gs_from_sv
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

