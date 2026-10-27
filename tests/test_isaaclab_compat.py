# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

import robolab.core.utils.isaaclab_compat as compat


def test_quaternion_conversions_are_inverse():
    wxyz = (0.5, 0.1, 0.2, 0.3)
    xyzw = compat.quat_wxyz_to_isaaclab(wxyz, uses_xyzw=True)
    assert xyzw == (0.1, 0.2, 0.3, 0.5)
    assert compat.quat_isaaclab_to_wxyz(xyzw, uses_xyzw=True) == wxyz


def test_pose_conversion_preserves_position():
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3]])
    internal = compat.pose_wxyz_to_isaaclab(pose, uses_xyzw=True)
    torch.testing.assert_close(internal, torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5]]))
    torch.testing.assert_close(compat.pose_isaaclab_to_wxyz(internal, uses_xyzw=True), pose)


def test_scene_state_round_trip(monkeypatch):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", True)
    state = {
        "articulation": {
            "robot": {
                "root_pose": torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]),
                "joint_position": torch.tensor([[0.25]]),
            }
        },
        "rigid_object": {
            "cube": {"root_pose": torch.tensor([[4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]])}
        },
    }
    internal = compat.scene_state_to_isaaclab(state)
    assert internal["articulation"]["robot"]["root_pose"][0].tolist() == [1, 2, 3, 0, 0, 0, 1]
    restored = compat.scene_state_from_isaaclab(internal)
    torch.testing.assert_close(restored["articulation"]["robot"]["root_pose"], state["articulation"]["robot"]["root_pose"])
    torch.testing.assert_close(restored["rigid_object"]["cube"]["root_pose"], state["rigid_object"]["cube"]["root_pose"])


@pytest.mark.parametrize("uses_xyzw", [False, True])
def test_config_rotations_are_prepared_once_and_recorded_as_wxyz(monkeypatch, uses_xyzw):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", uses_xyzw)

    class Config:
        def __init__(self):
            self.rot = (1.0, 0.0, 0.0, 0.0)
            self.child = {"rot": [0.5, 0.1, 0.2, 0.3]}
            self.half_turn = {"rot": (0.0, 0.0, 0.0, 1.0)}

        def to_dict(self):
            return deepcopy(vars(self))

    cfg = Config()
    compat.prepare_env_cfg(cfg)
    assert cfg.rot == ((0.0, 0.0, 0.0, 1.0) if uses_xyzw else (1.0, 0.0, 0.0, 0.0))
    assert cfg.child["rot"] == ([0.1, 0.2, 0.3, 0.5] if uses_xyzw else [0.5, 0.1, 0.2, 0.3])
    assert cfg.half_turn["rot"] == ((0.0, 0.0, 1.0, 0.0) if uses_xyzw else (0.0, 0.0, 0.0, 1.0))
    compat.prepare_env_cfg(cfg)
    assert cfg.rot == ((0.0, 0.0, 0.0, 1.0) if uses_xyzw else (1.0, 0.0, 0.0, 0.0))
    recorded = compat.env_cfg_to_recording_dict(cfg)
    assert recorded["rot"] == (1.0, 0.0, 0.0, 0.0)
    assert recorded["child"]["rot"] == [0.5, 0.1, 0.2, 0.3]
    assert recorded["half_turn"]["rot"] == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("uses_xyzw", [False, True])
@pytest.mark.parametrize("module_name,class_name", [
    ("droid", "DroidIKActionCfg"),
    ("droid", "DroidRelIKActionCfg"),
    ("franka_definitions", "FrankaIKActionCfg"),
    ("franka_definitions", "FrankaRelIKActionCfg"),
])
def test_ik_configs_author_wxyz_identity(monkeypatch, uses_xyzw, module_name, class_name):
    """Exercise real action configs, whose upstream defaults vary by stack."""
    from importlib import import_module

    cfg = getattr(import_module(f"robolab.robots.{module_name}"), class_name)()
    assert cfg.arm_action.body_offset.rot == (1.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", uses_xyzw)
    expected = (0.0, 0.0, 0.0, 1.0) if uses_xyzw else (1.0, 0.0, 0.0, 0.0)
    compat.prepare_env_cfg(cfg)
    assert cfg.arm_action.body_offset.rot == expected
    compat.prepare_env_cfg(cfg)
    assert cfg.arm_action.body_offset.rot == expected
    assert compat.env_cfg_to_recording_dict(cfg)["arm_action"]["body_offset"]["rot"] == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("uses_xyzw", [False, True])
@pytest.mark.parametrize("module_name,class_name", [
    ("droid", "DroidCfg"),
    ("franka", "FrankaCfg"),
    ("franka_high_pd", "FrankaCfg"),
    ("kinova_gen3", "KinovaGen3Cfg"),
])
def test_robot_root_and_frame_rotations_round_trip(monkeypatch, uses_xyzw, module_name, class_name):
    from importlib import import_module

    from isaaclab.utils import configclass

    from robolab.core.environments.config import generate_scene_env_cfg

    @configclass
    class EmptyTaskScene:
        pass

    class TestTask:
        scene = EmptyTaskScene

    robot_cfg = getattr(import_module(f"robolab.robots.{module_name}"), class_name)
    # Resolve robot metadata just as environment construction does. A raw robot
    # config still contains the immutable TableFixtureCfg authoring label.
    cfg = generate_scene_env_cfg(TestTask, robot_cfg)(num_envs=1, env_spacing=2.0)
    before = deepcopy(cfg.to_dict())
    assert cfg.robot.init_state.rot == (1.0, 0.0, 0.0, 0.0)
    frame_sensors = [cfg.frames]
    if hasattr(cfg, "ee_frame"):
        frame_sensors.append(cfg.ee_frame)
    for sensor in frame_sensors:
        assert sensor.source_frame_offset.rot == (1.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", uses_xyzw)
    expected_identity = (0.0, 0.0, 0.0, 1.0) if uses_xyzw else (1.0, 0.0, 0.0, 0.0)
    compat.prepare_env_cfg(cfg)
    assert cfg.robot.init_state.rot == expected_identity
    for sensor in frame_sensors:
        assert sensor.source_frame_offset.rot == expected_identity
    compat.prepare_env_cfg(cfg)
    recorded = compat.env_cfg_to_recording_dict(cfg)
    assert recorded["robot"]["init_state"]["rot"] == before["robot"]["init_state"]["rot"]
    for name in ("frames", "ee_frame"):
        if name in before:
            assert recorded[name] == before[name]
