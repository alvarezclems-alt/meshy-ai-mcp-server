# src/server.py
import os
import json
import httpx
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP        # <-- garder CE seul import
from starlette.responses import JSONResponse

# Ne PAS lever d'erreur au démarrage : on lira la clé dans les outils.
MESHY_API_KEY = os.getenv("MESHY_API_KEY")

mcp = FastMCP("meshy-mcp-server", log_level="INFO")

# --- Models (si tu en as besoin plus tard)
class TextTo3DTaskRequest(BaseModel):
    mode: str = Field(..., description="preview|refine")
    prompt: str
    art_style: str = "realistic"
    should_remesh: bool = True

# ... (garde tes autres modèles/outils si tu veux, mais ce qui suit suffit pour booter)

# --- Health & root (pour Render)
@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_):
    return JSONResponse({"ok": True})

@mcp.custom_route("/", methods=["GET"])
async def root(_):
    return JSONResponse({"service": "meshy-mcp-server", "status": "up"})

# IMPORTANT : exposer l’app ASGI pour Uvicorn
app = mcp.http_app()
