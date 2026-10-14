# SPDX-License-Identifier: Apache-2.0

"""Require successful placements to settle before terminating an episode."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch


def _target_names(params: dict[str, Any]) -> list[str]:
    """Return target object names referenced by a success term."""
    names: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, dict):
            add(value.get("object"))
            add(value.get("objects"))
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    add(params.get("object"))
    add(params.get("objects"))
    add(params.get("groups"))
    return list(dict.fromkeys(names))


class SuccessConfirmer:
    """Confirm success once every target has remained at rest for a short interval."""

    def __init__(
        self,
        success_predicate: Callable[..., torch.Tensor],
        rest_s: float,
        max_speed: float,
        targets: list[str],
    ) -> None:
        self.success_predicate = success_predicate
        self.rest_s = float(rest_s)
        self.max_speed = float(max_speed)
        self.targets = targets
        self.rest_count: torch.Tensor | None = None
        self.first_hold_step: torch.Tensor | None = None
        self.confirmed_step: torch.Tensor | None = None

    def _initialize(self, env: Any) -> None:
        self.rest_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.first_hold_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.confirmed_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def _required_rest_steps(self, env: Any) -> int:
        step_dt = float(env.step_dt)
        if step_dt <= 0:
            raise ValueError(f"Expected env.step_dt to be positive, received {step_dt}.")
        return max(1, round(self.rest_s / step_dt))

    def _targets_at_rest(self, env: Any) -> torch.Tensor:
        if not self.targets:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        from robolab.core.world.world_state import get_world

        world = get_world(env)
        at_rest = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for target_name in self.targets:
            velocity = world.get_velocity(target_name, env_id=None)
            at_rest &= torch.linalg.vector_norm(velocity[:, :3], dim=-1) < self.max_speed
        return at_rest

    def __call__(self, env: Any, **params: Any) -> torch.Tensor:
        raw_success = self.success_predicate(env, **params)
        if not isinstance(raw_success, torch.Tensor):
            raise TypeError(
                "Expected the success predicate to return torch.Tensor, "
                f"received {type(raw_success).__name__}."
            )
        if raw_success.shape != (env.num_envs,):
            raise ValueError(
                f"Expected success predicate shape ({env.num_envs},), received {tuple(raw_success.shape)}."
            )

        raw_success = raw_success.bool()
        if self.rest_count is None or self.rest_count.shape[0] != env.num_envs:
            self._initialize(env)

        assert self.rest_count is not None
        assert self.first_hold_step is not None
        assert self.confirmed_step is not None

        episode_step = env.episode_length_buf
        just_reset = episode_step <= 1
        self.rest_count = torch.where(just_reset, 0, self.rest_count)
        self.first_hold_step = torch.where(just_reset, -1, self.first_hold_step)
        self.confirmed_step = torch.where(just_reset, -1, self.confirmed_step)

        targets_at_rest = self._targets_at_rest(env)
        self.rest_count = torch.where(targets_at_rest, self.rest_count + 1, 0)

        first_hold = raw_success & (self.first_hold_step < 0)
        self.first_hold_step = torch.where(first_hold, episode_step, self.first_hold_step)

        confirmed = raw_success & (self.rest_count >= self._required_rest_steps(env))
        first_confirmation = confirmed & (self.confirmed_step < 0)
        self.confirmed_step = torch.where(first_confirmation, episode_step, self.confirmed_step)
        return confirmed


def confirmed_success_term(term: Any, rest_s: float, max_speed: float) -> Any:
    """Wrap an Isaac Lab success term without changing its validated signature."""
    if rest_s <= 0:
        return term

    confirmer = SuccessConfirmer(
        term.func,
        rest_s=rest_s,
        max_speed=max_speed,
        targets=_target_names(dict(term.params or {})),
    )

    @functools.wraps(term.func)
    def wrapped(env: Any, **params: Any) -> torch.Tensor:
        return confirmer(env, **params)

    wrapped.__signature__ = inspect.signature(term.func)
    wrapped.confirmer = confirmer
    term.func = wrapped
    return term
