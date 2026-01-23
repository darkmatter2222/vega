#!/usr/bin/env python3
"""
Vega Ingress - API Gateway / Reverse Proxy
Routes requests to appropriate backend services (TTS, LLM)
"""

import os
import logging
import argparse
from typing import Optional
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

# Backend service URLs (internal Docker network or localhost)
TTS_BACKEND = os.getenv("VEGA_TTS_BACKEND", "http://host.docker.internal:8000")
LLM_BACKEND = os.getenv("VEGA_LLM_BACKEND", "http://host.docker.internal:8001")

# Timeout for backend requests (seconds)
BACKEND_TIMEOUT = float(os.getenv("VEGA_BACKEND_TIMEOUT", "120"))

# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Vega Ingress",
    description="API Gateway for Vega services (TTS, LLM)",
    version="1.0.0"
)

# CORS - allow all for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client for backend requests
http_client = httpx.AsyncClient(timeout=BACKEND_TIMEOUT)

# ==============================================================================
# Health & Info
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check - also checks backend connectivity."""
    backends = {}
    
    # Check TTS
    try:
        resp = await http_client.get(f"{TTS_BACKEND}/health", timeout=5)
        backends["tts"] = resp.json() if resp.status_code == 200 else {"status": "error", "code": resp.status_code}
    except Exception as e:
        backends["tts"] = {"status": "unreachable", "error": str(e)}
    
    # Check LLM
    try:
        resp = await http_client.get(f"{LLM_BACKEND}/health", timeout=5)
        backends["llm"] = resp.json() if resp.status_code == 200 else {"status": "error", "code": resp.status_code}
    except Exception as e:
        backends["llm"] = {"status": "unreachable", "error": str(e)}
    
    all_healthy = all(
        b.get("status") in ["healthy", "ok"] 
        for b in backends.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "backends": backends
    }

@app.get("/info")
async def get_info():
    """Get information about available services."""
    return {
        "service": "vega-ingress",
        "version": "1.0.0",
        "routes": {
            "/tts/*": "Text-to-Speech service",
            "/llm/*": "Language Model service",
            "/api/tts/*": "Alias for TTS",
            "/api/llm/*": "Alias for LLM",
        },
        "backends": {
            "tts": TTS_BACKEND,
            "llm": LLM_BACKEND
        }
    }

# ==============================================================================
# Proxy Helper
# ==============================================================================

async def proxy_request(
    request: Request,
    backend_url: str,
    path: str
) -> Response:
    """Proxy a request to a backend service."""
    
    # Build target URL
    target_url = f"{backend_url}/{path}"
    if request.query_params:
        target_url += f"?{request.query_params}"
    
    # Get request body
    body = await request.body()
    
    # Forward headers (excluding hop-by-hop headers)
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ["host", "connection", "keep-alive", "transfer-encoding"]:
            headers[key] = value
    
    try:
        # Make request to backend
        response = await http_client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=headers,
        )
        
        # Build response headers
        response_headers = {}
        for key, value in response.headers.items():
            if key.lower() not in ["transfer-encoding", "connection", "content-encoding"]:
                response_headers[key] = value
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type")
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Backend timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Backend unavailable")
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# TTS Routes
# ==============================================================================

@app.api_route("/tts/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_tts(request: Request, path: str):
    """Proxy requests to TTS service."""
    return await proxy_request(request, TTS_BACKEND, path)

@app.api_route("/api/tts/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_api_tts(request: Request, path: str):
    """Alias: Proxy requests to TTS service."""
    return await proxy_request(request, TTS_BACKEND, path)

# ==============================================================================
# LLM Routes
# ==============================================================================

@app.api_route("/llm/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_llm(request: Request, path: str):
    """Proxy requests to LLM service."""
    return await proxy_request(request, LLM_BACKEND, path)

@app.api_route("/api/llm/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_api_llm(request: Request, path: str):
    """Alias: Proxy requests to LLM service."""
    return await proxy_request(request, LLM_BACKEND, path)

# ==============================================================================
# Convenience Routes (direct access to common endpoints)
# ==============================================================================

@app.post("/synthesize")
async def synthesize(request: Request):
    """Direct route to TTS synthesize."""
    return await proxy_request(request, TTS_BACKEND, "synthesize")

@app.post("/synthesize/b64")
async def synthesize_b64(request: Request):
    """Direct route to TTS synthesize base64."""
    return await proxy_request(request, TTS_BACKEND, "synthesize/b64")

@app.post("/chat")
async def chat(request: Request):
    """Direct route to LLM chat."""
    return await proxy_request(request, LLM_BACKEND, "chat")

@app.post("/generate")
async def generate(request: Request):
    """Direct route to LLM generate."""
    return await proxy_request(request, LLM_BACKEND, "generate")

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vega Ingress API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
