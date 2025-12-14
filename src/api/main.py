"""
FastAPI Application with WebSocket support for multi-client RAG demo.

This demonstrates async/non-blocking behavior with 4 concurrent clients.
"""

from core.rag_engine import RAGEngine
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import asyncio
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


app = FastAPI(title="RAG Multi-Client Demo")

# Mount static files
static_path = Path(__file__).parent.parent / "ui" / "static"
templates_path = Path(__file__).parent.parent / "ui" / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# Initialize RAG engine
rag_engine = RAGEngine()

# Track active connections
active_connections: dict[str, WebSocket] = {}
metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "total_time": 0.0,
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    avg_time = (
        round(metrics["total_time"] / metrics["total_queries"], 2)
        if metrics["total_queries"] > 0
        else 0
    )
    cache_hit_rate = (
        round((metrics["cache_hits"] / metrics["total_queries"]) * 100, 1)
        if metrics["total_queries"] > 0
        else 0
    )

    return {
        "total_queries": metrics["total_queries"],
        "cache_hits": metrics["cache_hits"],
        "cache_hit_rate": cache_hit_rate,
        "avg_response_time": avg_time,
        "active_clients": len(active_connections),
    }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for client connections.

    Each client gets its own WebSocket connection for real-time communication.
    """
    await websocket.accept()
    active_connections[client_id] = websocket

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "message": f"Client {client_id} connected",
                "client_id": client_id,
            }
        )

        while True:
            # Receive query from client
            data = await websocket.receive_json()

            if data["type"] == "query":
                query = data["query"]

                # Send processing status
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "processing",
                        "message": "Searching vector database...",
                    }
                )

                try:
                    # Process query (async, non-blocking!)
                    result = await rag_engine.process_query(query, client_id)

                    # Update metrics
                    metrics["total_queries"] += 1
                    metrics["total_time"] += result["processing_time"]
                    if result["cache_hit"]:
                        metrics["cache_hits"] += 1

                    # Send response in chunks (simulate streaming)
                    response_text = result["response"]
                    words = response_text.split()

                    # Stream response word by word
                    for i, word in enumerate(words):
                        await websocket.send_json(
                            {"type": "chunk", "content": word +
                                " ", "progress": i + 1}
                        )
                        # Small delay for visual effect
                        await asyncio.sleep(0.01)

                    # Send completion
                    await websocket.send_json(
                        {
                            "type": "complete",
                            "cache_hit": result["cache_hit"],
                            "processing_time": result["processing_time"],
                            "total_words": len(words),
                        }
                    )

                except Exception as e:
                    await websocket.send_json(
                        {"type": "error",
                            "message": f"Error processing query: {str(e)}"}
                    )

    except WebSocketDisconnect:
        if client_id in active_connections:
            del active_connections[client_id]
        print(f"Client {client_id} disconnected")

    except Exception as e:
        print(f"Error with client {client_id}: {e}")
        if client_id in active_connections:
            del active_connections[client_id]


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 Starting RAG Multi-Client Demo Server")
    print("=" * 60)
    print("📍 Open your browser to: http://localhost:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
