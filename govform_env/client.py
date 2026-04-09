"""HTTP client for GovFormEnv with from_docker_image() support."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from govform_env.models import GovFormAction, Observation


@dataclass
class StepResult:
    """Result returned by env.step()."""
    observation: dict
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_action_error(self) -> Optional[str]:
        return self.info.get("error")


@dataclass
class ResetResult:
    """Result returned by env.reset()."""
    observation: dict
    done: bool = False

    @property
    def task_id(self) -> str:
        return self.observation.get("task_id", "")


class GovFormEnv:
    """
    Async HTTP client that wraps the GovForm FastAPI server.

    Supports two modes:
    1. from_docker_image(image_name) — spins up a Docker container
    2. from_server_url(url) — connects to an already-running server
    """

    def __init__(self, base_url: str, container_id: Optional[str] = None):
        self._base_url = base_url.rstrip("/")
        self._container_id = container_id
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60)

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: int = 7860,
        timeout: int = 60,
    ) -> "GovFormEnv":
        """Spin up a Docker container from the given image and return a client."""
        base_url = f"http://localhost:{port}"

        # ── Step 0: Check if a server is ALREADY running on this port ──
        # This is common in evaluators where the server might already be up.
        try:
            async with httpx.AsyncClient() as probe:
                resp = await probe.get(f"{base_url}/health", timeout=2)
                if resp.status_code == 200:
                    # Server is already running. Reuse it.
                    return cls(base_url=base_url)
        except Exception:
            pass

        # ── Step 1: Try to start the container ──
        container_id = None
        try:
            result = subprocess.run(
                [
                    "docker", "run", "-d", "--rm",
                    "-p", f"{port}:{port}",
                    image_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            container_id = result.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError(
                "Docker is not installed or not in PATH. "
                "Install Docker: https://docs.docker.com/get-docker/"
            )
        except subprocess.CalledProcessError as e:
            # If we still fail with port conflict or other issue, try one last check
            # in case the server came up in the meantime.
            try:
                async with httpx.AsyncClient() as probe:
                    resp = await probe.get(f"{base_url}/health", timeout=5)
                    if resp.status_code == 200:
                        return cls(base_url=base_url)
            except Exception:
                pass
            raise RuntimeError(f"Failed to start Docker container (exit {e.returncode}): {e.stderr or e.stdout}")

        base_url = f"http://localhost:{port}"
        client = cls(base_url=base_url, container_id=container_id)

        # Wait for server readiness
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient() as probe:
                    resp = await probe.get(f"{base_url}/health", timeout=2)
                    if resp.status_code == 200:
                        return client
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(1)

        # Timed out
        await client.close()
        raise RuntimeError(
            f"Docker container started but server did not become ready within {timeout}s"
        )

    @classmethod
    async def from_server_url(cls, url: str) -> "GovFormEnv":
        """Connect to an already-running server."""
        client = cls(base_url=url)
        # Verify connectivity
        try:
            async with httpx.AsyncClient() as probe:
                resp = await probe.get(f"{url}/health", timeout=5)
                resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to server at {url}: {e}")
        return client

    async def reset(self, task_id: Optional[str] = None) -> ResetResult:
        """Reset the environment for a given task."""
        body = {"task_id": task_id} if task_id else {}
        resp = await self._client.post("/reset", json=body)
        resp.raise_for_status()
        data = resp.json()
        return ResetResult(observation=data)

    async def step(self, GovFormAction: GovFormAction) -> StepResult:
        """Send an GovFormAction to the environment."""
        resp = await self._client.post(
            "/step",
            json={
                "field_name": GovFormAction.field_name,
                "value": GovFormAction.value,
                "reasoning": GovFormAction.reasoning,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=data["observation"],
            reward=float(data["reward"]),
            done=bool(data["done"]),
            info=data.get("info", {}),
        )

    async def get_state(self) -> dict:
        """Get full environment state."""
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Clean up: close HTTP client and stop Docker container."""
        await self._client.aclose()
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True,
                    timeout=15,
                )
            except Exception as e:
                print(f"[DEBUG] Failed to stop container {self._container_id}: {e}")
            self._container_id = None
