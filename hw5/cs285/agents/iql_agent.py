from typing import Optional
import torch
from torch import nn
from cs285.agents.awac_agent import AWACAgent

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(AWACAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_value_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_value_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        expectile: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        self.value_critic = make_value_critic(observation_shape)

        # Note: According to the reference implementations
        # https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/iql_trainer.py
        # https://github.com/ikostrikov/implicit_q_learning
        # V function does not need a moving target version; V itself is a target for Q
        # self.target_value_critic = make_value_critic(observation_shape)
        # self.target_value_critic.load_state_dict(self.value_critic.state_dict())

        self.value_critic_optimizer = make_value_critic_optimizer(
            self.value_critic.parameters()
        )
        self.expectile = expectile

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # DONE(student): Compute advantage with IQL
        qa_values = self.critic(observations)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        q_values = qa_values.gather(1, actions).squeeze(1)
        values = self.value_critic(observations)
        advantages = q_values - values
        return advantages

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # DONE(student): Update Q(s, a) to match targets (based on V)
        target_values = rewards + torch.logical_not(dones) * self.discount * self.value_critic(next_observations)
        target_values.detach_()

        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        q_values = self.critic(observations).gather(1, actions).squeeze(1)

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            "q_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "q_mse_target_values": target_values.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }

        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # DONE(student): Compute the expectile loss
        u = target_qs - vs
        l2_tau = torch.abs(expectile - (u < 0).float()) * (u**2)
        return l2_tau.mean()

    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)
        """
        # DONE(student): Compute target values for V(s)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        target_qs = self.target_critic(observations).gather(1, actions).squeeze(1).detach()
        vs = self.value_critic(observations)

        # DONE(student): Update V(s) using the loss from the IQL paper
        loss = self.iql_expectile_loss(self.expectile, vs, target_qs)

        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            "vs_adv": (vs - target_qs).mean().item(),
            "vs": vs.mean().item(),
            "v_expectile_target_values": target_qs.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """

        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(observations, actions, rewards, next_observations, dones)
        metrics["actor_loss"] = self.update_actor(observations, actions)

        if step % self.target_update_period == 0:
            self.update_target_critic()
            # self.update_target_value_critic()

        return metrics

    # def update_target_value_critic(self):
    #     self.target_value_critic.load_state_dict(self.value_critic.state_dict())
