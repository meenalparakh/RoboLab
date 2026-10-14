# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for evaluation correctness fixes."""

from robolab.core.environments.env import MAX_EARLY_RESETS, should_retry_early_termination


def test_early_termination_retries_are_bounded():
    assert should_retry_early_termination(2, MAX_EARLY_RESETS - 1)
    assert not should_retry_early_termination(2, MAX_EARLY_RESETS)
    assert not should_retry_early_termination(3, 0)
