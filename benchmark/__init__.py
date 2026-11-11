"""
Cyrus Humanoid Quantitative Benchmarking & Evaluation Suite.
Grounded in empirical evaluation principles from van Marum et al.
"""

from .metrics import (
    compute_mechanical_work,
    compute_mechanical_effort,
    compute_step_symmetry,
    compute_foot_slipping,
    compute_joint_jerk
)

__all__ = [
    "compute_mechanical_work",
    "compute_mechanical_effort",
    "compute_step_symmetry",
    "compute_foot_slipping",
    "compute_joint_jerk"
]
