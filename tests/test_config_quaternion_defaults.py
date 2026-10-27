# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject simulator-native rotation defaults in RoboLab-authored configs.

This audit runs without Isaac Sim. A runtime test on Lab 2 alone cannot catch
omitted rotations, since its defaults already happen to match WXYZ.
"""

import ast
from pathlib import Path


# These constructors own rotations directly or introduce nested configs with
# rotation defaults. Require the boundary to be explicit, including offsets
# that would otherwise be created implicitly by the upstream config class.
REQUIRED_FIELDS = {
    "InitialStateCfg": "rot",
    "OffsetCfg": "rot",
    "AssetBaseCfg": "init_state",
    "ArticulationCfg": "init_state",
    "RigidObjectCfg": "init_state",
    "FrameTransformerCfg": "source_frame_offset",
    "FrameCfg": "offset",
    "CameraCfg": "offset",
    "TiledCameraCfg": "offset",
}


def test_config_rotations_do_not_inherit_simulator_defaults():
    root = Path(__file__).resolve().parents[1]
    missing = []
    for path in sorted((root / "robolab").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            field = REQUIRED_FIELDS.get(name)
            if field and field not in {keyword.arg for keyword in node.keywords}:
                missing.append(f"{path.relative_to(root)}:{node.lineno}: {name} requires explicit {field}")
    assert not missing, "Simulator-dependent quaternion defaults:\n" + "\n".join(missing)
