"""
client.py
=========
MLOpsEnv HTTP client — OpenEnv-compatible client for the MLOps environment.

Usage:
    from client import MLOpsEnvClient
    import asyncio

    async def main():
        # Connect to local server
        client = MLOpsEnvClient(base_url="http://localhost:8000")
        # Or from Docker image
        # client = await MLOpsEnvClient.from_docker_image(image_name)

        result = await client.reset("data_quality_triage")
        obs = result["observation"]
        result = await client.step({"action_type": "accept_record", "target_id": "rec_000"})

    asyncio.run(main())
"""

from __future__ import annotations

import json
import asyncio
import urllib.request
from typing import Any


class MLOpsEnvClient:
    """
    Async HTTP client for MLOpsEnv.
    Compatible with the OpenEnv client interface.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._closed  = False

    # ─── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 8000) -> "MLOpsEnvClient":
        """
        Start environment from a Docker image and return a connected client.
        Used when IMAGE_NAME env var is set by the evaluator.
        """
        import subprocess, time

        container_name = "mlops-env-eval"
        # Stop existing container if running
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True
        )
        # Start container
        subprocess.Popen([
            "docker", "run", "--rm",
            "--name", container_name,
            "-p", f"{port}:8000",
            image_name
        ])
        # Wait for server to start
        base_url = f"http://localhost:{port}"
        for _ in range(30):
            try:
                req = urllib.request.Request(f"{base_url}/health")
                urllib.request.urlopen(req, timeout=2)
                break
            except Exception:
                await asyncio.sleep(1)

        return cls(base_url=base_url)

    # ─── Core interface ───────────────────────────────────────────────────────

    async def reset(
        self,
        task_id: str = "data_quality_triage",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Reset environment and return initial observation."""
        body: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            body["seed"] = seed
        return await self._post("/reset", body)

    async def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one action and return (observation, reward, done, info)."""
        return await self._post("/step", {"action": action})

    async def state(self) -> dict[str, Any]:
        """Return current state without advancing the episode."""
        return await self._get("/state")

    async def close(self) -> None:
        """Clean up resources."""
        self._closed = True

    # ─── Convenience helpers ──────────────────────────────────────────────────

    async def triage(
        self,
        record_id: str,
        action_type: str,
        **params: Any,
    ) -> dict[str, Any]:
        """Helper for data_quality_triage actions."""
        return await self.step({
            "action_type": action_type,
            "target_id":   record_id,
            "parameters":  params,
            "reasoning":   "client helper",
        })

    async def investigate(self, component: str) -> dict[str, Any]:
        """Helper for incident_cascade investigation."""
        return await self.step({
            "action_type": "investigate",
            "target_id":   None,
            "parameters":  {"component": component},
            "reasoning":   "investigating root cause",
        })

    async def restart_service(self, component: str) -> dict[str, Any]:
        """Helper for incident_cascade service restart."""
        return await self.step({
            "action_type": "restart_service",
            "target_id":   None,
            "parameters":  {"component": component},
            "reasoning":   "restarting component",
        })

    async def deploy_canary(
        self,
        canary_pct: int = 5,
        rollback_threshold_pct: float = 0.4,
    ) -> dict[str, Any]:
        """Helper for deployment canary rollout."""
        return await self.step({
            "action_type": "deploy_canary",
            "target_id":   None,
            "parameters":  {
                "canary_pct":             canary_pct,
                "rollback_threshold_pct": rollback_threshold_pct,
            },
            "reasoning": "canary deployment to limit blast radius",
        })

    # ─── HTTP helpers ─────────────────────────────────────────────────────────

    async def _post(self, path: str, body: dict) -> dict[str, Any]:
        url  = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_request, req)

    async def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_request, req)

    @staticmethod
    def _do_request(req: urllib.request.Request) -> dict[str, Any]:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))