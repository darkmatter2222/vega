# Vega

A suite of AI services designed for local GPU deployment.

## Services

| Service | Description | Port | Model |
|---------|-------------|------|-------|
| [vega-tts](./vega-tts/) | Text-to-Speech with voice cloning | 8000 | Chatterbox TTS |
| [vega-llm](./vega-llm/) | Chat & text generation | 8001 | Qwen2.5-0.5B-Instruct |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                   │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│   vega-tts    │           │   vega-llm    │
│   Port 8000   │           │   Port 8001   │
├───────────────┤           ├───────────────┤
│ Chatterbox    │           │ Qwen3-0.6B    │
│ Voice Clone   │           │ Chat/Generate │
└───────────────┘           └───────────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
               ┌──────┴──────┐
               │  RTX 3090   │
               │   24GB      │
               └─────────────┘
```

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (RTX 3090 recommended)
- HuggingFace token for model downloads

### Deploy TTS Service

```powershell
cd vega-tts
Copy-Item .env.example .env
# Edit .env with your settings
.\deploy.ps1
```

### Deploy LLM Service

```powershell
cd vega-llm
Copy-Item .env.example .env
# Edit .env with your settings
.\deploy.ps1
```

## API Usage

### Text-to-Speech (vega-tts)

```python
import requests

response = requests.post("http://your-server:8000/synthesize", json={
    "text": "Hello, this is Vega speaking!"
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### LLM Chat (vega-llm)

```python
import requests

response = requests.post("http://your-server:8001/chat", json={
    "message": "What is machine learning?"
})

print(response.json()["response"])
```

## Server Requirements

- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **OS**: Ubuntu 22.04 LTS recommended
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **RAM**: 32GB recommended
- **Storage**: 50GB+ for models

## Project Structure

```
vega/
├── vega-tts/           # Text-to-Speech service
│   ├── api.py          # FastAPI server
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── deploy.ps1      # Deployment script
│   ├── client_sample.py
│   └── models/         # Voice conditioning files
│
├── vega-llm/           # LLM service
│   ├── api.py          # FastAPI server
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── deploy.ps1      # Deployment script
│   └── client_sample.py
│
└── README.md           # This file
```

## License

See [LICENSE](./LICENSE) for details.
