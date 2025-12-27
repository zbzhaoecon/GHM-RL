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
        V(c) = E[ ∫_0^τ e^(-ρt) (dL_t - dE_t - φ·𝟙(dE>0)) ]

    where:
        - dL_t: Dividend payouts (continuous control a_L)
        - dE_t: Gross equity issuances (what new shareholders pay)
        - φ: Fixed cost of equity issuance
        - ρ: Discount rate = r - μ
        - τ: Liquidation time (c reaches 0)

    IMPORTANT: The cost of issuing gross equity a_E is the FULL amount a_E,
    not (p-1)/p * a_E. This is because:
        - New shareholders pay a_E (gross)
        - Firm receives a_E/p in cash (after proportional cost)
        - New shareholders get claims worth a_E
        - Dilution to existing shareholders = a_E

    The dynamics add a_E/p to cash, so the VALUE FUNCTION captures the
    benefit of the cash inflow. The REWARD captures the immediate dilution.

    HJB first-order condition for optimal equity issuance:
        -1 + V'(c)/p = 0  →  V' = p
    So equity should only be issued when marginal value of cash > p = 1.06.

    In discrete time:
        Per-step reward: r_t = (a_L - a_E - φ·𝟙(a_E>0))·dt
        Terminal reward: r_T = ω·α/(r-μ) (liquidation value)

    Note: All terms are scaled by dt since a_L and a_E are RATES (per unit time)
    in the dynamics.

    Parameters:
        discount_rate: ρ = r - μ
        liquidation_rate: ω (recovery rate at bankruptcy)
        liquidation_flow: α (expected flow at bankruptcy)

    Example:
        >>> reward_fn = GHMRewardFunction(
        ...     discount_rate=0.03,
        ...     liquidation_rate=0.8,
        ...     liquidation_flow=0.5,
        ... )
        >>>
        >>> # Compute step reward
        >>> state = torch.tensor([[2.0]])  # c = 2.0
        >>> action = torch.tensor([[0.5, 1.0]])  # a_L=0.5, a_E=1.0
        >>> reward = reward_fn.step_reward(state, action, state, dt=0.01)
        >>> # reward = (0.5 - 1.0) * 0.01 = -0.005
    """

    def __init__(
        self,
        discount_rate: float,
        issuance_cost: float = 0.0,  # Deprecated, kept for backward compatibility
        liquidation_rate: float = 1.0,
        liquidation_flow: float = 0.0,
        fixed_cost: float = 0.0,
        proportional_cost: float = 1.06,  # p parameter from model
    ):
        """
        Initialize GHM reward function.

        Args:
            discount_rate: ρ = r - μ
            issuance_cost: DEPRECATED - ignored, cost is always 1.0 (full dilution)
            liquidation_rate: ω (fraction recovered)
            liquidation_flow: α (expected flow)
            fixed_cost: φ (fixed cost per issuance)
            proportional_cost: p (gross-to-net conversion factor, used in dynamics)

        IMPORTANT: The dilution cost is the FULL gross equity a_E, not a fraction.
        This ensures the HJB first-order condition is: V'(c) = p (not V'(c) = p-1).
        Equity issuance is only optimal when marginal value of cash exceeds p.
        """
        self.discount_rate_value = discount_rate
        self.proportional_cost = proportional_cost
        self.fixed_cost = fixed_cost
        self.liquidation_rate = liquidation_rate
        self.liquidation_flow = liquidation_flow

        # Cost coefficient is 1.0 (full gross equity = full dilution)
        # The issuance_cost parameter is deprecated and ignored
        self.issuance_cost = 1.0

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
            r_t = (a_L - a_E - φ·𝟙(a_E>0))·dt

        where:
            - a_L = action[:, 0] (dividend rate, per unit time)
            - a_E = action[:, 1] (gross equity issuance rate, per unit time)
            - φ = fixed_cost

        The cost of equity issuance is the FULL gross amount a_E (not a fraction).
        This is the correct formulation because:
            - New shareholders pay a_E (gross)
            - Firm receives a_E/p in cash (captured by dynamics)
            - Dilution to existing shareholders = a_E (captured by reward)

        HJB first-order condition: -1 + V'(c)/p = 0 → V' = p
        Equity is optimal only when marginal value of cash > p = 1.06.

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

        # Fixed cost: only paid when issuing equity (𝟙(a_E > threshold) · φ)
        is_issuing = (a_E > 1e-6).to(dtype=action.dtype)
        fixed_cost_penalty = self.fixed_cost * is_issuing

        # Reward = (dividends - full dilution - fixed cost) * dt
        # issuance_cost = 1.0 (full gross equity as dilution cost)
        return (a_L - self.issuance_cost * a_E - fixed_cost_penalty) * dt

    def terminal_reward(
        self,
        state: Tensor,
        terminated: Tensor,
        value_function=None,
        recapitalization_target: float = 0.5,
    ) -> Tensor:
        """
        Compute terminal reward with proper boundary conditions.

        Boundary conditions for time-augmented finite-horizon problem:
            1. Bankruptcy (c≤0): V(0, τ) = liquidation_value (typically 0)
            2. Horizon end (τ=0): V(c, 0) = 0 (firm liquidates, cash is worthless)

        The optimal policy should pay out all remaining cash as dividends
        in the final period before τ=0, since:
            - Dividend constraint: a_L ≤ c/dt allows full extraction
            - Any cash remaining at τ=0 is lost (terminal value = 0)

        Args:
            state: Terminal states (batch, state_dim)
            terminated: Boolean mask (batch,) - True if bankrupt
            value_function: Optional value network (not used, for compatibility)
            recapitalization_target: Not used in finite-horizon setting

        Returns:
            Terminal rewards (batch,) - always 0 or liquidation_value
        """
        batch_size = state.shape[0]
        device = state.device

        # Terminal value is liquidation_value for all trajectories
        # (typically 0 for both bankrupt and horizon end)
        terminal_rewards = torch.full(
            (batch_size,),
            self.liquidation_value,
            dtype=torch.float32,
            device=device
        )

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
