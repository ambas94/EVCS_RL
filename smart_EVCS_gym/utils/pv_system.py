from dataclasses import dataclass
from numpy import ndarray


@dataclass
class PVSystem:
    length: float
    width: float
    depth: float
    total_dimensions: float
    efficiency: float
    # solar_irradiance: ndarray = None
