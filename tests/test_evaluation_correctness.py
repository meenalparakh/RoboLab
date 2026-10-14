# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for evaluation correctness fixes."""

import types

import torch

from robolab.core.environments.env import (
    MAX_EARLY_RESETS,
    should_retry_early_termination,
    success_from_termination_manager,
)
from robolab.core.task.confirm import SuccessConfirmer, _target_names
from robolab.core.task.off_table import group_lost, required_groups, task_lost


def test_early_termination_retries_are_bounded():
    assert should_retry_early_termination(2, MAX_EARLY_RESETS - 1)
    assert not should_retry_early_termination(2, MAX_EARLY_RESETS)
    assert not should_retry_early_termination(3, 0)


class _World:
    def __init__(self, speeds: dict[str, torch.Tensor]) -> None:
        self.speeds = speeds

    def get_velocity(self, name: str, env_id=None) -> torch.Tensor:
        velocity = torch.zeros(self.speeds[name].shape[0], 6)
        velocity[:, 0] = self.speeds[name]
        return velocity


class _Env:
    def __init__(self, num_envs: int, step_dt: float = 0.1) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = step_dt
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)


def test_success_waits_for_moving_targets(monkeypatch):
    env = _Env(2)
    speeds = {"banana": torch.tensor([0.0, 0.5])}
    world_state = types.SimpleNamespace(get_world=lambda _env: _World(speeds))
    monkeypatch.setitem(__import__("sys").modules, "robolab.core.world.world_state", world_state)

    raw_success = torch.tensor([False, False])
    confirmer = SuccessConfirmer(
        lambda _env, **_params: raw_success,
        rest_s=0.2,
        max_speed=0.02,
        targets=["banana"],
    )

    for step in range(1, 4):
        env.episode_length_buf[:] = step
        assert confirmer(env).tolist() == [False, False]

    raw_success[:] = True
    env.episode_length_buf[:] = 4
    assert confirmer(env).tolist() == [True, False]

    speeds["banana"][1] = 0.0
    env.episode_length_buf[:] = 5
    assert confirmer(env).tolist() == [True, False]
    env.episode_length_buf[:] = 6
    assert confirmer(env).tolist() == [True, True]


def test_success_target_names_support_grouped_predicates():
    assert _target_names(
        {"groups": [{"object": ["apple", "banana"]}, {"objects": "orange"}]}
    ) == ["apple", "banana", "orange"]


def test_success_comes_only_from_the_named_success_term():
    class _TerminationManager:
        terminated = torch.tensor([True, True])

        @staticmethod
        def get_term(name: str) -> torch.Tensor:
            assert name == "success"
            return torch.tensor([False, True])

    manager = _TerminationManager()
    assert not success_from_termination_manager(manager, 0)
    assert success_from_termination_manager(manager, 1)


def test_required_groups_include_single_group_destinations():
    assert required_groups({"object": "banana", "container": "bowl"}) == [
        (["banana"], "all", 1),
        (["bowl"], "all", 1),
    ]
    assert required_groups(
        {"object": ["a", "b", "c"], "logical": "choose", "K": 2}
    ) == [(["a", "b", "c"], "choose", 2)]


def test_off_table_loss_respects_all_any_and_choose_semantics():
    fallen = {
        "a": torch.tensor([True, False, True]),
        "b": torch.tensor([False, False, True]),
        "c": torch.tensor([False, False, False]),
    }
    assert group_lost(fallen, ["a", "b", "c"], "all", 1, 3, "cpu").tolist() == [True, False, True]
    assert group_lost(fallen, ["a", "b", "c"], "any", 1, 3, "cpu").tolist() == [False, False, False]
    assert group_lost(fallen, ["a", "b", "c"], "choose", 2, 3, "cpu").tolist() == [False, False, True]


def test_lost_destination_ends_single_group_task():
    groups = required_groups({"object": ["mug", "ceramic_mug"], "container": "rack"})
    fallen = {
        "mug": torch.tensor([False, False]),
        "ceramic_mug": torch.tensor([False, False]),
        "rack": torch.tensor([True, False]),
    }
    assert task_lost(fallen, groups, 2, "cpu").tolist() == [True, False]
