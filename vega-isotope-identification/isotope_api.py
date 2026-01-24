#!/usr/bin/env python3
"""
Vega Isotope Identification REST API

FastAPI server for gamma spectrum isotope identification using the trained Vega model.

Endpoints:
    POST /identify         - Identify isotopes from spectrum data (numpy array)
    POST /identify/b64     - Identify isotopes from base64-encoded spectrum
    POST /identify/batch   - Batch identification for multiple spectra
    GET  /health           - Health check
    GET  /info             - Model info and supported isotopes
    GET  /isotopes         - List all supported isotopes

The model accepts:
    - 1D spectrum: shape (1023,) - single measurement
    - 2D spectrum: shape (N, 1023) - time series (averaged automatically)
    
Energy range: 20 keV to 3000 keV across 1023 channels
"""

import os
import sys
import base64
import io
import json
import logging
import argparse
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

MODEL_PATH = os.getenv("VEGA_ISOTOPE_MODEL", "models/vega_v2_final.pt")
DEVICE = os.getenv("VEGA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = float(os.getenv("VEGA_DEFAULT_THRESHOLD", "0.5"))
NUM_CHANNELS = 1023  # Fixed by model architecture
NUM_ISOTOPES = 82    # Fixed by model architecture

# ==============================================================================
# Model Architecture (embedded from vega_portable_inference.py)
# ==============================================================================

# Complete list of 82 isotopes supported by the model (alphabetically sorted)
ISOTOPE_NAMES = [
    "Ac-228", "Ag-110m", "Am-241", "Ba-133", "Be-7", "Bi-207", "Bi-211",
    "Bi-212", "Bi-214", "C-14", "Cd-109", "Ce-141", "Ce-144", "Co-57",
    "Co-60", "Cr-51", "Cs-134", "Cs-137", "Eu-152", "Eu-154", "F-18",
    "Fe-59", "Ga-67", "Ga-68", "H-3", "I-123", "I-125", "I-131", "In-111",
    "Ir-192", "K-40", "Lu-177", "Mn-54", "Na-22", "Nb-95", "Pa-231",
    "Pa-234m", "Pb-210", "Pb-211", "Pb-212", "Pb-214", "Po-210", "Ra-223",
    "Ra-224", "Ra-226", "Rn-219", "Rn-222", "Ru-103", "Ru-106", "Sb-124",
    "Sb-125", "Se-75", "Sn-113", "Sr-85", "Sr-90", "Tc-99m", "Th-227",
    "Th-228", "Th-230", "Th-232", "Th-234", "Tl-201", "Tl-208", "U-234",
    "U-235", "U-238", "Y-90", "Zn-65", "Zr-95",
    # Additional isotopes to reach 82
    "Ba-140", "Br-82", "Ca-45", "Ca-47", "Cf-252", "Cl-36", "Cm-244",
    "Cu-64", "Gd-153", "Hg-203", "Np-237", "P-32", "Pu-239"
]

# Gamma emission lines (keV) for reference
GAMMA_LINES = {
    "Am-241": [(59.54, 0.359)],
    "Ba-133": [(81.0, 0.329), (276.4, 0.071), (302.9, 0.183), (356.0, 0.620), (383.8, 0.089)],
    "Cs-137": [(661.7, 0.851)],
    "Co-57": [(122.1, 0.856), (136.5, 0.107)],
    "Co-60": [(1173.2, 0.999), (1332.5, 0.999)],
    "Eu-152": [(121.8, 0.284), (344.3, 0.265), (778.9, 0.129), (964.1, 0.146), (1112.1, 0.136), (1408.0, 0.210)],
    "Na-22": [(511.0, 1.798), (1274.5, 0.999)],
    "Mn-54": [(834.8, 0.9998)],
    "K-40": [(1460.8, 0.107)],
    "Ra-226": [(186.2, 0.036)],
    "Pb-214": [(295.2, 0.192), (351.9, 0.371)],
    "Bi-214": [(609.3, 0.461), (1120.3, 0.150), (1764.5, 0.154)],
    "Pb-212": [(238.6, 0.436)],
    "Tl-208": [(583.2, 0.845), (2614.5, 0.99)],
    "Ac-228": [(338.3, 0.113), (911.2, 0.258), (969.0, 0.158)],
    "I-131": [(364.5, 0.817), (637.0, 0.072)],
    "Tc-99m": [(140.5, 0.890)],
    "F-18": [(511.0, 1.934)],
    "Ir-192": [(296.0, 0.287), (308.5, 0.300), (316.5, 0.828), (468.1, 0.478)],
    "Th-232": [(63.8, 0.0026)],
    "U-238": [(49.6, 0.064), (113.5, 0.017)],
}


class IsotopeIndex:
    """Maps isotope names to model output indices and vice versa."""
    
    def __init__(self, isotope_names: Optional[List[str]] = None):
        if isotope_names is None:
            isotope_names = ISOTOPE_NAMES
        
        self._isotope_names = sorted(isotope_names)
        self._name_to_idx = {name: idx for idx, name in enumerate(self._isotope_names)}
        self._idx_to_name = {idx: name for idx, name in enumerate(self._isotope_names)}
    
    @property
    def num_isotopes(self) -> int:
        return len(self._isotope_names)
    
    @property
    def isotope_names(self) -> List[str]:
        return self._isotope_names.copy()
    
    def name_to_index(self, name: str) -> int:
        if name not in self._name_to_idx:
            raise KeyError(f"Isotope '{name}' not in index")
        return self._name_to_idx[name]
    
    def index_to_name(self, idx: int) -> str:
        if idx not in self._idx_to_name:
            raise KeyError(f"Index {idx} out of range [0, {self.num_isotopes-1}]")
        return self._idx_to_name[idx]


from dataclasses import dataclass, field
from typing import List


@dataclass
class VegaConfig:
    """Configuration for the Vega model architecture."""
    num_channels: int = 1023
    num_isotopes: int = 82
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    conv_kernel_size: int = 7
    pool_size: int = 2
    fc_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout_rate: float = 0.3
    spatial_dropout_rate: float = 0.1
    leaky_relu_slope: float = 0.1
    classification_weight: float = 1.0
    regression_weight: float = 0.1
    max_activity_bq: float = 1000.0


class ConvBlock(torch.nn.Module):
    """CNN block: Conv → BN → LeakyReLU → Conv → BN → LeakyReLU → MaxPool → Dropout"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout_rate: float = 0.1,
        leaky_slope: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.act1 = torch.nn.LeakyReLU(leaky_slope)
        
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.act2 = torch.nn.LeakyReLU(leaky_slope)
        
        self.pool = torch.nn.MaxPool1d(pool_size)
        self.dropout = torch.nn.Dropout1d(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.dropout(self.pool(x))
        return x


class VegaModel(torch.nn.Module):
    """Vega: CNN-FCNN for Multi-Label Isotope Classification + Activity Regression"""
    
    def __init__(self, config: VegaConfig):
        super().__init__()
        self.config = config
        self.backbone = self._build_backbone()
        self._flat_size = self._calculate_flat_size()
        self.classifier = self._build_classifier()
        self.regressor = self._build_regressor()
    
    def _build_backbone(self) -> torch.nn.Sequential:
        layers = []
        in_ch = 1
        for out_ch in self.config.conv_channels:
            layers.append(ConvBlock(
                in_ch, out_ch,
                kernel_size=self.config.conv_kernel_size,
                pool_size=self.config.pool_size,
                dropout_rate=self.config.spatial_dropout_rate,
                leaky_slope=self.config.leaky_relu_slope
            ))
            in_ch = out_ch
        return torch.nn.Sequential(*layers)
    
    def _calculate_flat_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.config.num_channels)
            x = self.backbone(x)
            return x.view(1, -1).size(1)
    
    def _build_classifier(self) -> torch.nn.Sequential:
        layers = []
        in_dim = self._flat_size
        for hidden_dim in self.config.fc_hidden_dims:
            layers.extend([
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(self.config.leaky_relu_slope),
                torch.nn.Dropout(self.config.dropout_rate)
            ])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, self.config.num_isotopes))
        return torch.nn.Sequential(*layers)
    
    def _build_regressor(self) -> torch.nn.Sequential:
        layers = []
        in_dim = self._flat_size
        for hidden_dim in self.config.fc_hidden_dims:
            layers.extend([
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(self.config.leaky_relu_slope),
                torch.nn.Dropout(self.config.dropout_rate)
            ])
            in_dim = hidden_dim
        layers.extend([
            torch.nn.Linear(in_dim, self.config.num_isotopes),
            torch.nn.ReLU()  # Activities must be positive
        ])
        return torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        activities = self.regressor(features)
        return logits, activities


# ==============================================================================
# API Models (Pydantic)
# ==============================================================================

class IsotopePrediction(BaseModel):
    """Single isotope prediction result."""
    name: str = Field(..., description="Isotope name (e.g., 'Cs-137')")
    probability: float = Field(..., ge=0, le=1, description="Detection confidence [0, 1]")
    activity_bq: float = Field(..., ge=0, description="Estimated activity in Becquerels")
    present: bool = Field(..., description="True if probability >= threshold")


class IdentifyRequest(BaseModel):
    """Request to identify isotopes from spectrum data."""
    spectrum: List[float] = Field(
        ..., 
        min_length=NUM_CHANNELS, 
        max_length=NUM_CHANNELS,
        description=f"Gamma spectrum as list of {NUM_CHANNELS} float values (counts per channel)"
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD, 
        ge=0, 
        le=1,
        description="Detection threshold (0-1). Lower = more sensitive, higher = more specific"
    )
    return_all: bool = Field(
        default=False,
        description="If true, return all 82 isotopes. If false, only detected ones."
    )


class IdentifyB64Request(BaseModel):
    """Request with base64-encoded numpy array."""
    spectrum_b64: str = Field(
        ...,
        description="Base64-encoded numpy array (.npy format bytes)"
    )
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0, le=1)
    return_all: bool = Field(default=False)


class IdentifyBatchRequest(BaseModel):
    """Batch identification request."""
    spectra: List[List[float]] = Field(
        ...,
        description=f"List of spectra, each with {NUM_CHANNELS} channels"
    )
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0, le=1)
    return_all: bool = Field(default=False)


class IdentifyResponse(BaseModel):
    """Response from isotope identification."""
    isotopes: List[IsotopePrediction] = Field(..., description="List of isotope predictions")
    num_detected: int = Field(..., description="Number of isotopes detected above threshold")
    confidence: float = Field(..., description="Overall prediction confidence")
    threshold_used: float = Field(..., description="Detection threshold that was applied")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class IdentifyBatchResponse(BaseModel):
    """Response from batch identification."""
    results: List[IdentifyResponse]
    total_spectra: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


class InfoResponse(BaseModel):
    """Model information response."""
    service: str = "vega-isotope-identification"
    version: str = "1.0.0"
    model_path: str
    device: str
    num_isotopes: int
    num_channels: int
    default_threshold: float
    model_loaded: bool
    architecture: str


class IsotopeListResponse(BaseModel):
    """List of supported isotopes."""
    isotopes: List[str]
    total: int
    gamma_lines: Dict[str, List[List[float]]]


# ==============================================================================
# Inference Engine Singleton
# ==============================================================================

class VegaInferenceEngine:
    """Singleton inference engine for isotope identification."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, model_path: str, device: str = None):
        """Load the model."""
        if self._initialized:
            return
        
        self.model_path = model_path
        
        # Device selection
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device='cuda') + 1
                self.device = torch.device('cuda')
            except RuntimeError:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load model config
        if 'model_config' in self.checkpoint:
            config_dict = self.checkpoint['model_config']
            self.model_config = VegaConfig(**config_dict)
        elif 'params' in self.checkpoint:
            params = self.checkpoint['params']
            self.model_config = VegaConfig(
                conv_channels=params.get('conv_channels', [64, 128, 256]),
                conv_kernel_size=params.get('conv_kernel_size', 7),
                pool_size=params.get('pool_size', 2),
                fc_hidden_dims=params.get('fc_hidden_dims', [512, 256]),
                dropout_rate=params.get('dropout_rate', 0.3),
                spatial_dropout_rate=params.get('spatial_dropout_rate', 0.1),
                leaky_relu_slope=params.get('leaky_relu_slope', 0.1)
            )
        else:
            self.model_config = VegaConfig()
        
        # Create and load model
        self.model = VegaModel(self.model_config)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Isotope index
        self.isotope_index = IsotopeIndex()
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Isotopes: {self.isotope_index.num_isotopes}")
        logger.info(f"  Architecture: CNN{self.model_config.conv_channels} → FC{self.model_config.fc_hidden_dims}")
        
        self._initialized = True
    
    @property
    def is_loaded(self) -> bool:
        return self._initialized
    
    def preprocess(self, spectrum: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Preprocess spectrum for model input."""
        # Average time series if 2D
        if spectrum.ndim == 2:
            spectrum = spectrum.mean(axis=0)
        
        # Normalize
        if normalize and spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        # To tensor with batch dimension
        tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        spectrum: np.ndarray,
        threshold: float = 0.5,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """Run inference on a gamma spectrum."""
        import time
        start_time = time.perf_counter()
        
        # Preprocess
        tensor = self.preprocess(spectrum)
        
        # Run model
        logits, activities = self.model(tensor)
        
        # Convert to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        activities = activities.cpu().numpy()[0] * self.model_config.max_activity_bq
        
        # Build predictions
        isotopes = []
        for i in range(len(probs)):
            prob = float(probs[i])
            activity = float(activities[i])
            present = prob >= threshold
            
            if return_all or present:
                isotopes.append({
                    "name": self.isotope_index.index_to_name(i),
                    "probability": round(prob, 4),
                    "activity_bq": round(activity, 2) if present else 0.0,
                    "present": present
                })
        
        # Calculate confidence
        present_isotopes = [iso for iso in isotopes if iso["present"]]
        if present_isotopes:
            confidence = np.mean([iso["probability"] for iso in present_isotopes])
        else:
            confidence = 1.0 - float(probs.max())
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "isotopes": isotopes,
            "num_detected": len(present_isotopes),
            "confidence": round(float(confidence), 4),
            "threshold_used": threshold,
            "processing_time_ms": round(processing_time, 2)
        }


# Global inference engine
engine = VegaInferenceEngine()


# ==============================================================================
# FastAPI Application
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup: load model
    logger.info("Starting Vega Isotope Identification API...")
    try:
        engine.initialize(MODEL_PATH, DEVICE)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't crash - allow health check to report unhealthy
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Vega Isotope Identification API",
    description="Gamma spectrum isotope identification using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if engine.is_loaded else "unhealthy",
        "model_loaded": engine.is_loaded,
        "device": str(engine.device) if engine.is_loaded else "N/A"
    }


@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get model and service information."""
    return {
        "service": "vega-isotope-identification",
        "version": "1.0.0",
        "model_path": MODEL_PATH,
        "device": str(engine.device) if engine.is_loaded else "N/A",
        "num_isotopes": NUM_ISOTOPES,
        "num_channels": NUM_CHANNELS,
        "default_threshold": DEFAULT_THRESHOLD,
        "model_loaded": engine.is_loaded,
        "architecture": f"CNN{engine.model_config.conv_channels} → FC{engine.model_config.fc_hidden_dims}" if engine.is_loaded else "N/A"
    }


@app.get("/isotopes", response_model=IsotopeListResponse)
async def list_isotopes():
    """List all supported isotopes with their gamma lines."""
    return {
        "isotopes": ISOTOPE_NAMES,
        "total": len(ISOTOPE_NAMES),
        "gamma_lines": {k: [list(line) for line in v] for k, v in GAMMA_LINES.items()}
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify_isotopes(request: IdentifyRequest):
    """
    Identify isotopes from a gamma spectrum.
    
    The spectrum should be 1023 float values representing counts per channel.
    Energy range: 20 keV to 3000 keV.
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        spectrum = np.array(request.spectrum, dtype=np.float32)
        result = engine.predict(spectrum, request.threshold, request.return_all)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify/b64", response_model=IdentifyResponse)
async def identify_isotopes_b64(request: IdentifyB64Request):
    """
    Identify isotopes from a base64-encoded numpy array.
    
    The spectrum should be a .npy file encoded as base64.
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64
        npy_bytes = base64.b64decode(request.spectrum_b64)
        spectrum = np.load(io.BytesIO(npy_bytes))
        
        # Validate shape
        if spectrum.ndim == 1 and len(spectrum) != NUM_CHANNELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {NUM_CHANNELS} channels, got {len(spectrum)}"
            )
        if spectrum.ndim == 2 and spectrum.shape[1] != NUM_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {NUM_CHANNELS} channels, got {spectrum.shape[1]}"
            )
        
        result = engine.predict(spectrum, request.threshold, request.return_all)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify/batch", response_model=IdentifyBatchResponse)
async def identify_isotopes_batch(request: IdentifyBatchRequest):
    """
    Batch identification for multiple spectra.
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.perf_counter()
    
    results = []
    for spectrum_data in request.spectra:
        if len(spectrum_data) != NUM_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"Each spectrum must have {NUM_CHANNELS} channels"
            )
        spectrum = np.array(spectrum_data, dtype=np.float32)
        result = engine.predict(spectrum, request.threshold, request.return_all)
        results.append(result)
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    return {
        "results": results,
        "total_spectra": len(results),
        "total_processing_time_ms": round(total_time, 2)
    }


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vega Isotope Identification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8020, help="Port to bind to")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model checkpoint")
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    uvicorn.run(app, host=args.host, port=args.port)
