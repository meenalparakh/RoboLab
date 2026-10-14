# SPDX-License-Identifier: Apache-2.0

"""Terminate episodes whose success condition can no longer be satisfied."""

from __future__ import annotations

from typing import Any

import torch

from robolab.constants import OFF_TABLE_DROP_M


RequiredGroup = tuple[list[str], str, int]


def _names(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        names: list[str] = []
        for item in value:
            names.extend(_names(item, field_name))
        return names
    raise TypeError(
        f"Expected success parameter '{field_name}' to be a string or sequence of strings, "
        f"received {type(value).__name__}."
    )


def _destinations(params: dict[str, Any]) -> list[str]:
    destinations: list[str] = []
    for field_name in ("container", "surface", "containers", "reference_object"):
        destinations.extend(_names(params.get(field_name), field_name))
    return list(dict.fromkeys(destinations))


def _required_object_group(params: dict[str, Any]) -> RequiredGroup | None:
    objects = _names(params.get("object"), "object") + _names(params.get("objects"), "objects")
    if not objects:
        return None

    logical = str(params.get("logical", "all"))
    if logical not in {"all", "any", "choose"}:
        raise ValueError(f"Expected logical mode 'all', 'any', or 'choose', received {logical!r}.")

    required_count = int(params.get("K", params.get("k", 1)) or 1)
    if logical == "choose" and not 1 <= required_count <= len(objects):
        raise ValueError(
            f"Expected choose count between 1 and {len(objects)}, received {required_count}."
        )
    return (objects, logical, required_count)


def required_groups(params: dict[str, Any]) -> list[RequiredGroup]:
    """Extract assets whose loss makes a success predicate impossible."""
    raw_groups = params.get("groups")
    if raw_groups is not None:
        if not isinstance(raw_groups, (list, tuple)):
            raise TypeError(
                "Expected success parameter 'groups' to be a sequence of mappings, "
                f"received {type(raw_groups).__name__}."
            )

        groups: list[RequiredGroup] = []
        destinations: list[str] = []
        for group in raw_groups:
            if not isinstance(group, dict):
                raise TypeError(
                    "Expected every success group to be a mapping, "
                    f"received {type(group).__name__}."
                )
            required_group = _required_object_group(group)
            if required_group is not None:
                groups.append(required_group)
            destinations.extend(_destinations(group))

        # Multi-group predicates can combine destinations with higher-level
        # any/choose semantics that are not represented by the term parameters.
        if len(groups) == 1:
            groups.extend(([destination], "all", 1) for destination in destinations)
        return groups

    required_group = _required_object_group(params)
    if required_group is None:
        return []
    groups = [required_group]
    groups.extend(([destination], "all", 1) for destination in _destinations(params))
    return groups


def group_lost(
    fallen: dict[str, torch.Tensor],
    objects: list[str],
    logical: str,
    required_count: int,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Return the environments where an object group is no longer achievable."""
    missing = [name for name in objects if name not in fallen]
    if missing:
        raise KeyError(f"Missing off-table state for required objects: {missing}.")

    fallen_by_object = torch.stack([fallen[name] for name in objects], dim=0)
    if fallen_by_object.shape != (len(objects), num_envs):
        raise ValueError(
            f"Expected fallen state shape ({len(objects)}, {num_envs}), "
            f"received {tuple(fallen_by_object.shape)}."
        )

    if logical == "all":
        return fallen_by_object.any(dim=0)
    if logical == "any":
        return fallen_by_object.all(dim=0)
    if logical == "choose":
        available_count = len(objects) - fallen_by_object.sum(dim=0)
        return available_count < required_count
    raise ValueError(f"Expected logical mode 'all', 'any', or 'choose', received {logical!r}.")


def task_lost(
    fallen: dict[str, torch.Tensor],
    groups: list[RequiredGroup],
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Return environments where any required group is no longer achievable."""
    lost = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for objects, logical, required_count in groups:
        lost |= group_lost(fallen, objects, logical, required_count, num_envs, device)
    return lost


class OffTableMonitor:
    """Track per-environment reset heights for required task assets."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.initial_z: dict[str, torch.Tensor] = {}

    def fallen_masks(self, objects: list[str]) -> dict[str, torch.Tensor]:
        from robolab.core.world.world_state import get_world

        world = get_world(self.env)
        just_reset = self.env.episode_length_buf <= 1
        fallen: dict[str, torch.Tensor] = {}
        for object_name in objects:
            position, _ = world.get_pose(object_name, env_id=None)
            current_z = position[:, 2]
            initial_z = self.initial_z.get(object_name)
            if initial_z is None:
                initial_z = current_z.clone()
            else:
                initial_z = torch.where(just_reset, current_z, initial_z)
            self.initial_z[object_name] = initial_z
            fallen[object_name] = current_z < initial_z - OFF_TABLE_DROP_M
        return fallen


_MONITORS: dict[int, OffTableMonitor] = {}


def _get_monitor(env: Any) -> OffTableMonitor:
    monitor = _MONITORS.get(id(env))
    if monitor is None or monitor.env is not env:
        monitor = OffTableMonitor(env)
        _MONITORS[id(env)] = monitor
    return monitor


def targets_lost(env: Any, groups: list[RequiredGroup]) -> torch.Tensor:
    """Isaac Lab termination term for an irrecoverably lost target or destination."""
    objects = sorted({object_name for group in groups for object_name in group[0]})
    fallen = _get_monitor(env).fallen_masks(objects)
    return task_lost(fallen, groups, env.num_envs, env.device)
