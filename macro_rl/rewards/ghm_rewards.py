"""
Reward function for the GHM equity model.

Objective: Maximize present value of net payouts to shareholders
    = Dividends - Equity dilution cost
"""

import torch
from torch import Tensor

from macro_rl.rewards.base import RewardFunction


class GHMRewardFunction(RewardFunction):
    """
    Reward function for GHM equity model.

    The firm maximizes:
        V(c) = E[ ∫_0^τ e^(-ρt) dL_t - (1+λ) dE_t ]

    where:
        - dL_t: Dividend payouts (continuous control a_L)
        - dE_t: Equity issuances (singular control a_E)
        - λ: Fixed cost of equity issuance
        - ρ: Discount rate = r - μ
        - τ: Liquidation time (c reaches 0)

    In discrete time:
        Per-step reward: r_t = a_L·dt - (1+λ)·a_E
        Terminal reward: r_T = ω·α/(r-μ) (liquidation value)

    where ω is the liquidation recovery rate.

    Parameters:
        discount_rate: ρ = r - μ
        issuance_cost: λ (fixed cost parameter)
        liquidation_rate: ω (recovery rate at bankruptcy)
        liquidation_flow: α (expected flow at bankruptcy)

    Example:
        >>> reward_fn = GHMRewardFunction(
        ...     discount_rate=0.03,
        ...     issuance_cost=0.1,
        ...     liquidation_rate=0.8,
        ...     liquidation_flow=0.5,
        ... )
        >>>
        >>> # Compute step reward
        >>> state = torch.tensor([[2.0]])  # c = 2.0
        >>> action = torch.tensor([[0.5, 0.0]])  # a_L=0.5, a_E=0
        >>> reward = reward_fn.step_reward(state, action, state, dt=0.01)
        >>> # reward = 0.5 * 0.01 - (1+0.1) * 0 = 0.005
    """

    def __init__(
        self,
        discount_rate: float,
        issuance_cost: float = 0.0,
        liquidation_rate: float = 1.0,
        liquidation_flow: float = 0.0,
    ):
        """
        Initialize GHM reward function.

        Args:
            discount_rate: ρ = r - μ
            issuance_cost: λ (multiplicative cost)
            liquidation_rate: ω (fraction recovered)
            liquidation_flow: α (expected flow)

        TODO: Store parameters
        """
        self.discount_rate_value = discount_rate
        self.issuance_cost = issuance_cost
        self.liquidation_rate = liquidation_rate
        self.liquidation_flow = liquidation_flow

        # Compute liquidation value
        if discount_rate > 0:
            self.liquidation_value = liquidation_rate * liquidation_flow / discount_rate
        else:
            self.liquidation_value = 0.0

    def step_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Compute per-step reward.

        Formula:
            r_t = a_L·dt - (1+λ)·a_E

        where:
            - a_L = action[:, 0] (dividend rate)
            - a_E = action[:, 1] (equity issuance)

        Args:
            state: Current states (batch, state_dim)
            action: Actions (batch, 2) where [:, 0]=a_L, [:, 1]=a_E
            next_state: Next states (batch, state_dim)
            dt: Time step size

        Returns:
            Rewards (batch,)
        """
        a_L = action[:, 0]
        a_E = action[:, 1]
        # Net payout to existing shareholders: dividends minus equity dilution cost
        # If a_E is GROSS equity issued, then:
        # - Firm receives: a_E/p in cash
        # - New shareholders get: a_E in equity
        # - Net cost to existing shareholders: a_E - a_E/p = a_E(p-1)/p
        # Where issuance_cost = (p-1), so cost = issuance_cost * a_E / p
        # Note: For p=1.06, this is 0.06/1.06 ≈ 0.0566, not 0.06
        # But we use the approximation (p-1)*a_E ≈ (p-1)/p * a_E for p ≈ 1
        return a_L * dt - self.issuance_cost * a_E

    def terminal_reward(
        self,
        state: Tensor,
        terminated: Tensor,
        value_function=None,
        recapitalization_target: float = 0.5,
    ) -> Tensor:
        """
        Compute terminal reward with optimal liquidation/recapitalization choice.

        Implements boundary condition (equation 5 from GHM):
            F(0) = max{ max_c (F(c) - p(c+φ)), ωα/(r-μ) }

        If value_function is provided, computes:
            terminal_value = max{ liquidation_value,
                                  V(c*,τ) - recapitalization_cost }

        Otherwise, uses simple liquidation value.

        Args:
            state: Terminal states (batch, state_dim)
            terminated: Boolean mask (batch,)
            value_function: Optional value network for continuation value
            recapitalization_target: Target cash level c* for recapitalization

        Returns:
            Terminal rewards (batch,)
        """
        terminated_f = terminated.to(dtype=torch.float32, device=state.device)
        liquidation_value = torch.tensor(self.liquidation_value, dtype=torch.float32, device=state.device)

        if value_function is None or not terminated.any():
            # Simple case: just use liquidation value
            return terminated_f * liquidation_value

        # Implement boundary condition: choose between liquidation and recapitalization
        batch_size = state.shape[0]
        terminal_values = torch.zeros(batch_size, dtype=torch.float32, device=state.device)

        # Get indices of terminated trajectories
        terminated_idx = terminated.nonzero(as_tuple=True)[0]

        if len(terminated_idx) > 0:
            # For terminated trajectories, compute continuation value at recapitalization target
            # State is (c, τ) for time-augmented, or just (c) for standard
            recapitalized_states = state[terminated_idx].clone()
            recapitalized_states[:, 0] = recapitalization_target  # Set c to target level

            with torch.no_grad():
                # Get continuation value V(c*, τ)
                continuation_values = value_function(recapitalized_states).squeeze()

                # Compute recapitalization cost: p(c* + φ)
                # Assuming we're starting from c ≈ 0, need to reach c*
                from macro_rl.dynamics.ghm_equity import GHMEquityParams
                params = GHMEquityParams()
                recapitalization_cost = params.p * (recapitalization_target + params.phi)

                # Recapitalization value = continuation value - cost
                recapitalization_values = continuation_values - recapitalization_cost

                # Choose max between liquidation and recapitalization
                optimal_values = torch.max(
                    liquidation_value.expand_as(recapitalization_values),
                    recapitalization_values
                )

                terminal_values[terminated_idx] = optimal_values

        return terminal_values

    def net_payout(
        self,
        action: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Compute net payout to shareholders per step.

        Same as step_reward but as a standalone utility.

        Args:
            action: Actions (batch, 2)
            dt: Time step

        Returns:
            Net payouts (batch,)
        """
        # Create dummy state (not used in step_reward)
        dummy_state = torch.zeros(action.shape[0], 1, device=action.device)
        return self.step_reward(dummy_state, action, dummy_state, dt)

    def total_issuance_cost(
        self,
        action: Tensor,
    ) -> Tensor:
        """
        Compute total cost of equity issuance.

        Formula:
            cost = (1 + λ)·a_E

        Args:
            action: Actions (batch, 2)

        Returns:
            Total costs (batch,)
        """
        a_E = action[:, 1]
        return (1.0 + self.issuance_cost) * a_E


class GHMRewardWithPenalty(GHMRewardFunction):
    """
    GHM reward with additional penalty terms.

    Extensions:
        - Bankruptcy penalty (beyond liquidation value)
        - Variance penalty (for risk aversion)
        - Constraint violations (soft penalties)

    TODO: Implement extensions (future)
    """

    def __init__(
        self,
        discount_rate: float,
        issuance_cost: float = 0.0,
        liquidation_rate: float = 1.0,
        liquidation_flow: float = 0.0,
        bankruptcy_penalty: float = 0.0,
        variance_penalty: float = 0.0,
    ):
        """
        Initialize with penalty terms.

        Args:
            discount_rate: ρ = r - μ
            issuance_cost: λ
            liquidation_rate: ω
            liquidation_flow: α
            bankruptcy_penalty: Additional cost of bankruptcy
            variance_penalty: Risk aversion parameter

        TODO: Implement initialization
        """
        super().__init__(discount_rate, issuance_cost, liquidation_rate, liquidation_flow)
        self.bankruptcy_penalty = bankruptcy_penalty
        self.variance_penalty = variance_penalty

    def terminal_reward(
        self,
        state: Tensor,
        terminated: Tensor,
    ) -> Tensor:
        """
        Terminal reward with bankruptcy penalty.

        Formula:
            r_T = ω·α/(r-μ) - penalty if terminated

        TODO: Implement with penalty
        """
        raise NotImplementedError
