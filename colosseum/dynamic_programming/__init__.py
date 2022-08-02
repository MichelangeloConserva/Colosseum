DP_MAX_ITERATION = int(1e6)

from colosseum.dynamic_programming.finite_horizon import (
    episodic_policy_evaluation,
    episodic_value_iteration,
)
from colosseum.dynamic_programming.infinite_horizon import (
    discounted_policy_iteration,
    discounted_value_iteration,
)
