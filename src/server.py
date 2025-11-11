# src/server.py
"""
Meshy AI MCP Server — tools & resources for Meshy AI 3D generation.
Works on Render with: uvicorn src.server:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import httpx
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# MCP runtime
from fastmcp import FastMCP

# Small HTTP helpers for health routes
from starlette.responses import JSONResponse
from starlette.requests import Request

# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

load_dotenv()

mcp = FastMCP("Meshy AI MCP Server", log_level="ERROR")


def require_api_key() -> str:
    """Read API key from env and fail fast if missing (tool call time)."""
    key = os.getenv("MESHY_API_KEY")
    if not key:
        raise RuntimeError("MESHY_API_KEY environment variable is not set")
    return key


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class TextTo3DTaskRequest(BaseModel):
    mode: str = Field(..., description="Task mode: 'preview' or 'refine'")
    prompt: str = Field(..., description="Text prompt describing the 3D model")
    art_style: str = Field("realistic", description="Art style")
    should_remesh: bool = Field(True, description="Whether to remesh after generation")


class RemeshTaskRequest(BaseModel):
    input_task_id: str = Field(..., description="ID of the input task to remesh")
    target_formats: List[str] = Field(default_factory=lambda: ["glb", "fbx"])
    topology: str = Field("quad", description="Topology: 'quad' or 'triangle'")
    target_polycount: int = Field(50_000, description="Target polygon count")
    resize_height: float = Field(1.0, description="Resize height")
    origin_at: str = Field("bottom", description="Origin position")


class ImageTo3DTaskRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to convert to 3D")
    prompt: Optional[str] = Field(None, description="Optional guidance prompt")
    art_style: str = Field("realistic", description="Art style")


class TextToTextureTaskRequest(BaseModel):
    model_url: str = Field(..., description="URL of the 3D model to texture")
    object_prompt: str = Field(..., description="Object description")
    style_prompt: Optional[str] = Field(None, description="Style description")
    enable_original_uv: bool = Field(True)
    enable_pbr: bool = Field(True)
    resolution: str = Field("1024")
    negative_prompt: Optional[str] = Field(None)
    art_style: str = Field("realistic")


class ListTasksParams(BaseModel):
    page_size: int = Field(10)
    page: int = Field(1)


class TaskResponse(BaseModel):
    id: str
    result: Optional[str] = None


# ---------------------------------------------------------------------
# Tools (create/retrieve/list/stream/balance)
# ---------------------------------------------------------------------

@mcp.tool()
async def create_text_to_3d_task(request: TextTo3DTaskRequest) -> TaskResponse:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.meshy.ai/openapi/v2/text-to-3d",
            headers=headers,
            json=request.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        data = resp.json()
        # v2 returns task id in "result"
        return TaskResponse(id=data["result"], result=data.get("result"))


@mcp.tool()
async def retrieve_text_to_3d_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v2/text-to-3d/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def create_image_to_3d_task(request: ImageTo3DTaskRequest) -> TaskResponse:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.meshy.ai/openapi/v1/image-to-3d",
            headers=headers,
            json=request.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        data = resp.json()
        return TaskResponse(id=data["id"], result=data.get("id"))


@mcp.tool()
async def retrieve_image_to_3d_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def create_remesh_task(request: RemeshTaskRequest) -> TaskResponse:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.meshy.ai/openapi/v1/remesh",
            headers=headers,
            json=request.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        data = resp.json()
        return TaskResponse(id=data["id"], result=data.get("id"))


@mcp.tool()
async def retrieve_remesh_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v1/remesh/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def create_text_to_texture_task(request: TextToTextureTaskRequest) -> TaskResponse:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.meshy.ai/openapi/v1/text-to-texture",
            headers=headers,
            json=request.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        data = resp.json()
        return TaskResponse(id=data["result"], result=data.get("result"))


@mcp.tool()
async def retrieve_text_to_texture_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v1/text-to-texture/{task_id}",
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_text_to_3d_tasks(params: Optional[ListTasksParams] = None) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    q = params.model_dump(exclude_none=True) if params else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.meshy.ai/openapi/v2/text-to-3d", headers=headers, params=q
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_image_to_3d_tasks(params: Optional[ListTasksParams] = None) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    q = params.model_dump(exclude_none=True) if params else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.meshy.ai/openapi/v1/image-to-3d", headers=headers, params=q
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_remesh_tasks(params: Optional[ListTasksParams] = None) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    q = params.model_dump(exclude_none=True) if params else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.meshy.ai/openapi/v1/remesh", headers=headers, params=q
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_text_to_texture_tasks(params: Optional[ListTasksParams] = None) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    q = params.model_dump(exclude_none=True) if params else {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.meshy.ai/openapi/v1/text-to-texture", headers=headers, params=q
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def stream_text_to_3d_task(task_id: str, timeout: int = 300) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}", "Accept": "text/event-stream"}
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"https://api.meshy.ai/openapi/v2/text-to-3d/{task_id}/stream",
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            final_data = None
            async for line in resp.aiter_lines():
                if line and line.startswith("data:"):
                    data = json.loads(line[5:])
                    final_data = data
                    if data.get("status") in {"SUCCEEDED", "FAILED", "CANCELED"}:
                        break
            return final_data or {"error": "No data received from stream"}


@mcp.tool()
async def stream_image_to_3d_task(task_id: str, timeout: int = 300) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}", "Accept": "text/event-stream"}
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}/stream",
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            final_data = None
            async for line in resp.aiter_lines():
                if line and line.startswith("data:"):
                    data = json.loads(line[5:])
                    final_data = data
                    if data.get("status") in {"SUCCEEDED", "FAILED", "CANCELED"}:
                        break
            return final_data or {"error": "No data received from stream"}


@mcp.tool()
async def stream_remesh_task(task_id: str, timeout: int = 300) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}", "Accept": "text/event-stream"}
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"https://api.meshy.ai/openapi/v1/remesh/{task_id}/stream",
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            final_data = None
            async for line in resp.aiter_lines():
                if line and line.startswith("data:"):
                    data = json.loads(line[5:])
                    final_data = data
                    if data.get("status") in {"SUCCEEDED", "FAILED", "CANCELED"}:
                        break
            return final_data or {"error": "No data received from stream"}


@mcp.tool()
async def stream_text_to_texture_task(task_id: str, timeout: int = 300) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}", "Accept": "text/event-stream"}
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"https://api.meshy.ai/openapi/v1/text-to-texture/{task_id}/stream",
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            final_data = None
            async for line in resp.aiter_lines():
                if line and line.startswith("data:"):
                    data = json.loads(line[5:])
                    final_data = data
                    if data.get("status") in {"SUCCEEDED", "FAILED", "CANCELED"}:
                        break
            return final_data or {"error": "No data received from stream"}


@mcp.tool()
async def get_balance() -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.meshy.ai/openapi/v1/balance", headers=headers)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------
# MCP resources
# ---------------------------------------------------------------------

@mcp.resource("health://status")
def health_check() -> Dict[str, Any]:
    return {"status": "ok", "api_key_configured": bool(os.getenv("MESHY_API_KEY"))}


@mcp.resource("task://text-to-3d/{task_id}")
async def get_text_to_3d_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v2/text-to-3d/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


@mcp.resource("task://image-to-3d/{task_id}")
async def get_image_to_3d_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


@mcp.resource("task://remesh/{task_id}")
async def get_remesh_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v1/remesh/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


@mcp.resource("task://text-to-texture/{task_id}")
async def get_text_to_texture_task(task_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.meshy.ai/openapi/v1/text-to-texture/{task_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------
# Health routes for Render + ASGI app exposure
# ---------------------------------------------------------------------

@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_: Request):
    return JSONResponse({"ok": True})


@mcp.custom_route("/", methods=["GET"])
async def root(_: Request):
    return JSONResponse({"service": "meshy-mcp-server", "status": "up"})


# This is what uvicorn will import: `src.server:app`
app = mcp.http_app()

