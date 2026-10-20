# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the imported dashboard fixes through real HTTP requests."""

from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from dashboard.app import _video_response, create_app


@pytest.fixture
def video_client(tmp_path: Path) -> TestClient:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"0123456789")
    app = FastAPI()

    @app.get("/video")
    def video_route(request: Request):
        return _video_response(video, request)

    return TestClient(app)


@pytest.mark.parametrize(
    ("byte_range", "expected", "content_range"),
    [
        ("bytes=2-5", b"2345", "bytes 2-5/10"),
        ("bytes=7-", b"789", "bytes 7-9/10"),
        ("bytes=-3", b"789", "bytes 7-9/10"),
    ],
)
def test_video_can_seek(video_client: TestClient, byte_range: str, expected: bytes, content_range: str) -> None:
    response = video_client.get("/video", headers={"Range": byte_range})
    assert response.status_code == 206
    assert response.content == expected
    assert response.headers["content-range"] == content_range
    assert response.headers["accept-ranges"] == "bytes"


def test_full_video_advertises_seeking(video_client: TestClient) -> None:
    response = video_client.get("/video")
    assert response.status_code == 200
    assert response.content == b"0123456789"
    assert response.headers["accept-ranges"] == "bytes"


def test_unsatisfiable_video_range(video_client: TestClient) -> None:
    response = video_client.get("/video", headers={"Range": "bytes=10-"})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */10"


def test_dashboard_homepage_renders_with_current_starlette(tmp_path: Path) -> None:
    with TestClient(create_app(initial_dir=tmp_path)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
