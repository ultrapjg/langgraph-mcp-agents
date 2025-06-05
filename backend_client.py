import os
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def backend_initialize(selected_model: str, mcp_config=None) -> dict:
    """Initialize backend session via HTTP."""
    payload = {"selected_model": selected_model}
    if mcp_config is not None:
        payload["mcp_config"] = mcp_config
    resp = requests.post(f"{BACKEND_URL}/initialize", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def backend_process(query: str, timeout_seconds: int = 60) -> dict:
    """Send a query to the backend and return the response."""
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"query": query, "timeout_seconds": timeout_seconds},
        timeout=timeout_seconds + 5,
    )
    resp.raise_for_status()
    return resp.json()
