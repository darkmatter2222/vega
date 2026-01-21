# Vega TTS API

REST API for text-to-speech synthesis using the tuned Vega voice.

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux

# Install dependencies
pip install -r requirements-api.txt

# Run the API
python api.py --port 8000
```

### Docker

```bash
# Build and run
docker compose up -d --build

# Check logs
docker logs vega-api -f
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/synthesize` | POST | Generate audio, returns WAV file |
| `/synthesize/b64` | POST | Generate audio, returns base64 JSON |
| `/health` | GET | Health check |
| `/info` | GET | Model status and config |

## Usage

### curl
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Vega."}' \
  -o output.wav
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/synthesize",
    json={"text": "Radiation levels nominal."}
)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment instructions including:
- Docker/Kubernetes setup
- GPU configuration
- Port forwarding & DNS
- HTTPS with reverse proxy

## Requirements

- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (for CUDA inference)
- Docker with nvidia-container-toolkit (for containerized deployment)

## License

See [LICENSE](LICENSE)
