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
        fixed_cost: float = 0.0,
        proportional_cost: float = 1.06,  # p parameter from model
    ):
        """
        Initialize GHM reward function.

        Args:
            discount_rate: ρ = r - μ
            issuance_cost: (p-1) or (p-1)/p depending on formulation
            liquidation_rate: ω (fraction recovered)
            liquidation_flow: α (expected flow)
            fixed_cost: φ (fixed cost per issuance)
            proportional_cost: p (gross-to-net conversion factor)

        IMPORTANT: The correct cost to existing shareholders when issuing
        gross equity a_E is: a_E * (p-1)/p, not a_E * (p-1).

        For backward compatibility, if issuance_cost is provided as (p-1),
        it will be automatically converted to (p-1)/p using proportional_cost.
        """
        self.discount_rate_value = discount_rate
        self.proportional_cost = proportional_cost
        self.fixed_cost = fixed_cost
        self.liquidation_rate = liquidation_rate
        self.liquidation_flow = liquidation_flow

        # Convert issuance_cost to correct formulation: (p-1)/p
        # If issuance_cost ≈ (p-1), convert it; otherwise use as-is
        if abs(issuance_cost - (proportional_cost - 1.0)) < 0.001:
            # User passed (p-1), convert to (p-1)/p
            self.issuance_cost = (proportional_cost - 1.0) / proportional_cost
        else:
            # User passed correct value or custom value
            self.issuance_cost = issuance_cost

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

        # Net payout to existing shareholders: dividends minus equity dilution cost minus fixed cost
        #
        # CORRECT FORMULATION (Décamps et al 2017):
        # If a_E is GROSS equity issued, then:
        # - Firm receives: a_E/p in cash (dynamics)
        # - New shareholders get: a_E in equity
        # - Net cost to existing shareholders: a_E - a_E/p = a_E(p-1)/p
        # - Fixed cost φ paid when issuing equity (𝟙(a_E > 0) · φ)
        #
        # Where p = proportional_cost (e.g., 1.06), so:
        # - issuance_cost should be (p-1)/p = 0.06/1.06 ≈ 0.0566
        # - NOT (p-1) = 0.06 (this overestimates cost by ~6%)

        # Fixed cost: only paid when issuing equity (𝟙(a_E > threshold) · φ)
        # Use threshold 1e-6 to match dynamics implementation
        is_issuing = (a_E > 1e-6).to(dtype=action.dtype)
        fixed_cost_penalty = self.fixed_cost * is_issuing

        # Now using correct cost: (p-1)/p * a_E
        return a_L * dt - self.issuance_cost * a_E - fixed_cost_penalty

    def terminal_reward(
        self,
        state: Tensor,
        terminated: Tensor,
        value_function=None,
        recapitalization_target: float = 0.5,
    ) -> Tensor:
        """
        Compute terminal reward with proper boundary conditions.

        Boundary conditions:
            1. Bankruptcy (c=0): V(0, τ) = liquidation_value (usually 0)
            2. End of horizon (τ=0): V(c, 0) = V(c) from value function (bootstrap)

        For time-augmented dynamics:
            - Terminated trajectories (c≤0): terminal_reward = liquidation_value
            - Non-terminated at horizon end: terminal_reward = V(c, 0) for bootstrapping

        Args:
            state: Terminal states (batch, state_dim)
            terminated: Boolean mask (batch,) - True if bankrupt
            value_function: Optional value network for continuation value
            recapitalization_target: Target cash level c* for recapitalization

        Returns:
            Terminal rewards (batch,)
        """
        batch_size = state.shape[0]
        device = state.device
        terminal_rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)

        terminated_f = terminated.to(dtype=torch.float32, device=device)
        liquidation_value = torch.tensor(self.liquidation_value, dtype=torch.float32, device=device)

        # For bankrupt trajectories: use liquidation value (typically 0)
        terminal_rewards = terminal_rewards + terminated_f * liquidation_value

        # For non-bankrupt trajectories at horizon end: bootstrap from value function
        if value_function is not None:
            non_terminated = ~terminated
            if non_terminated.any():
                with torch.no_grad():
                    # Bootstrap from value function V(c, τ) at final state
                    # This provides the boundary condition V(c, 0) = V(c)
                    continuation_values = value_function(state).squeeze()
                    terminal_rewards = terminal_rewards + non_terminated.float() * continuation_values

        return terminal_rewards

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
