# RAG System with Single-Node & Distributed Architectures

A production-ready Retrieval-Augmented Generation (RAG) system with support for both single-node (development) and distributed (production) deployments.

## 🌟 Features

### Core RAG Capabilities
- **Retrieval-Augmented Generation**: Context-aware answers using PDF content indexed in Qdrant
- **LangGraph Orchestration**: State machine-based conversational flow
- **Cross-Encoder Reranking**: Improved retrieval relevance
- **Redis Caching**: 5-minute TTL for query results
- **Answer Validation Loop**: Automatic relevance checking with retry logic

### Web Interface
- **4-Client Demo**: Concurrent WebSocket-based query processing
- **Real-time Monitoring**: Live metrics and query logging
- **Cache Indicators**: Visual cache hit/miss status
- **Streaming Responses**: Word-by-word response delivery

### Architecture Modes
- **Single-Node Mode**: Lightweight setup for local development
- **Distributed Mode**: Multi-node clusters for production scaling
  - 3-node Qdrant cluster with sharding
  - 3-node MongoDB replica set
  - 6-node Redis cluster
  - 3-node RabbitMQ cluster

## 🚀 Quick Start

### Single-Node Mode (Development)
```bash
# Start infrastructure
docker-compose up -d

# Install dependencies
pip install -e .

# Index PDF
python embed.py

# Start server
RAG_MODE=single uvicorn src.api.main:app --port 8000
```

### Distributed Mode (Production-like)
```bash
# Start distributed cluster
docker-compose -f docker-compose.distributed.yml up -d

# Index to cluster
python embed_distributed.py

# Start server
RAG_MODE=distributed uvicorn src.api.main:app --port 8000
```

**📖 For detailed instructions, see [DEMO_GUIDE.md](DEMO_GUIDE.md)**

## 📋 Requirements

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key in `.env`:
  ```env
  OPENAI_API_KEY=your-key-here
  ```

## 🌐 Access Points

| Service | Single Mode | Distributed Mode |
|---------|-------------|------------------|
| Demo UI | http://localhost:8000 | http://localhost:8000 |
| Qdrant | :6333 | :6333, :6336, :6338 |
| Monitoring | :3000 (Grafana) | :3000 (Grafana) |

## 📁 Project Structure

```
Project-RAG/
├── src/
│   ├── api/main.py              # FastAPI server
│   ├── core/
│   │   ├── config.py            # Mode configuration
│   │   ├── engine_factory.py    # Engine factory
│   │   ├── rag_engine.py        # Single-node engine
│   │   └── rag_engine_distributed.py
│   └── ui/                      # Web interface
├── cluster_clients.py           # Distributed clients
├── qdrant_sharding.py          # Sharding strategies
├── embed.py                    # Single-node embedding
├── embed_distributed.py        # Distributed embedding
├── docker-compose.yml          # Single-node setup
├── docker-compose.distributed.yml
└── DEMO_GUIDE.md               # Complete documentation
```

## 🎯 Usage

### Web Demo
1. Start the server (see Quick Start)
2. Visit http://localhost:8000
3. Use the 4-client interface to test concurrent queries
4. Monitor metrics at http://localhost:8000/monitor

### Command Line
```bash
python main.py
```

### Check Current Mode
```bash
curl http://localhost:8000/mode
```

## 🔧 Configuration

Switch modes via `RAG_MODE` environment variable:
- `RAG_MODE=single` - Single-node mode (default)
- `RAG_MODE=distributed` - Distributed cluster mode

## 📚 Documentation

- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Complete setup and usage guide
- **[pyproject.toml](pyproject.toml)** - Project dependencies

## 🤝 Development

1. Use **single-node mode** for rapid development
2. Test features locally with the demo UI
3. Switch to **distributed mode** for scaling tests
4. Use helper scripts in `scripts/` for environment management

---

**Ready to start?** See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed instructions.
