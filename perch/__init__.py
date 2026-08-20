"""Two-stage attack-region and attack-position selection for brain.

Perch first uses local doctrine and an LLM to rank polygon regions, then runs
FREA inside those regions while considering terrain masking, weapon range,
ingress exposure, obstacles, and designation constraints.
"""

from .region_recommender import AttackRegionRecommender
from .region_selector import AttackRegionSelector
from .selector import FREAPositionSelector
from .situation_understanding import SituationUnderstanding

__all__ = [
    "AttackRegionRecommender",
    "AttackRegionSelector",
    "FREAPositionSelector",
    "SituationUnderstanding",
]
