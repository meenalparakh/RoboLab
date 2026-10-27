# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Absolute camera resets preserve a common orientation across camera order."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from robolab.core.events import reset_camera
from robolab.core.utils import isaaclab_compat


class _Camera:
    device = "cpu"

    def __init__(self, index):
        self.data = SimpleNamespace(
            pos_w=torch.arange(9, dtype=torch.float32).reshape(3, 3) + index,
            quat_w_ros=torch.eye(4, dtype=torch.float32)[index:index + 1].repeat(3, 1),
        )
        self.set_world_poses = Mock()


@pytest.fixture
def camera_env(monkeypatch):
    cameras = {name: _Camera(index) for index, name in enumerate(("front", "left", "right"))}
    monkeypatch.setattr(reset_camera, "is_camera", lambda sensor: isinstance(sensor, _Camera))
    monkeypatch.setattr(reset_camera.robolab.constants, "VERBOSE", False)
    return SimpleNamespace(scene=SimpleNamespace(sensors=cameras))


@pytest.mark.parametrize("uses_xyzw", [False, True])
@pytest.mark.parametrize("orientation", [
    (1.0, 0.0, 0.0, 0.0),
    tuple(value / 30**0.5 for value in (1.0, 2.0, 3.0, 4.0)),
])
@pytest.mark.parametrize("camera_names", [
    ["front", "left", "right"],
    ["right", "left", "front"],
    "front",
])
def test_absolute_reset_uses_same_orientation_for_every_camera(
    monkeypatch, camera_env, uses_xyzw, orientation, camera_names
):
    monkeypatch.setattr(isaaclab_compat, "ISAACLAB_USES_XYZW", uses_xyzw)
    env_ids = torch.tensor([2, 0])
    original_orientation = orientation

    reset_camera.reset_camera_pose_absolute(
        camera_env, env_ids, camera_names, orientation=orientation
    )

    w, x, y, z = orientation
    expected = torch.tensor((x, y, z, w) if uses_xyzw else (w, x, y, z)).repeat(2, 1)
    selected = [camera_names] if isinstance(camera_names, str) else camera_names
    for name, camera in camera_env.scene.sensors.items():
        if name not in selected:
            camera.set_world_poses.assert_not_called()
            continue
        camera.set_world_poses.assert_called_once()
        actual = camera.set_world_poses.call_args.kwargs
        torch.testing.assert_close(actual["orientations"], expected)
        torch.testing.assert_close(actual["positions"], camera.data.pos_w[env_ids])
        torch.testing.assert_close(actual["env_ids"], env_ids)
    assert orientation == original_orientation


@pytest.mark.parametrize("uses_xyzw", [False, True])
def test_position_only_reset_preserves_each_camera_orientation(monkeypatch, camera_env, uses_xyzw):
    monkeypatch.setattr(isaaclab_compat, "ISAACLAB_USES_XYZW", uses_xyzw)
    env_ids = torch.tensor([2, 0])
    reset_camera.reset_camera_pose_absolute(
        camera_env, env_ids, list(camera_env.scene.sensors), position=(0.5, 1.0, 2.0)
    )

    for camera in camera_env.scene.sensors.values():
        camera.set_world_poses.assert_called_once()
        actual = camera.set_world_poses.call_args.kwargs
        torch.testing.assert_close(actual["orientations"], camera.data.quat_w_ros[env_ids])
        torch.testing.assert_close(actual["positions"], torch.tensor([[0.5, 1.0, 2.0]]).repeat(2, 1))
