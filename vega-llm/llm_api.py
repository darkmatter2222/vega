#!/usr/bin/env python3
"""
Vega LLM API - Qwen2.5 Language Model Service
FastAPI REST API for text generation using Qwen2.5-0.5B-Instruct
"""

import os
import sys
import logging
import argparse
from typing import Optional, List
from contextlib import asynccontextmanager

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

MODEL_ID = os.getenv("VEGA_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.getenv("VEGA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = int(os.getenv("VEGA_MAX_TOKENS", "2048"))

DEFAULT_SYSTEM_PROMPT = """You are Vega, a helpful AI assistant. You are:
- Concise and direct in your responses
- Knowledgeable and accurate
- Friendly but professional
- Focused on being genuinely helpful

Respond naturally and helpfully to the user's request."""

# ==============================================================================
# Request/Response Models
# ==============================================================================

class Message(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation messages")
    system_prompt: Optional[str] = Field(None, description="Override default system prompt")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate", ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling", ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_generated: int
    finish_reason: str

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to complete")
    max_tokens: Optional[int] = Field(512, description="Max tokens to generate", ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling", ge=0.0, le=1.0)

class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_id: str

# ==============================================================================
# Model Manager
# ==============================================================================

class VegaLLM:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        self.model_id = MODEL_ID
        self._initialized = True
    
    def load_model(self):
        """Load Qwen3 model and tokenizer."""
        if self.model is not None:
            return
        
        logger.info(f"Loading {self.model_id} on {self.device}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings for RTX 3090
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[str, int, str]:
        """Generate chat response."""
        if self.model is None:
            self.load_model()
        
        # Build conversation with system prompt
        system = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        # Format messages for Qwen
        formatted_messages = [{"role": "system", "content": system}]
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Determine finish reason
        finish_reason = "stop"
        if len(generated_tokens) >= max_tokens:
            finish_reason = "length"
        
        return response.strip(), len(generated_tokens), finish_reason
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[str, int]:
        """Generate text completion."""
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return text.strip(), len(generated_tokens)
    
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

# ==============================================================================
# FastAPI Application
# ==============================================================================

llm = VegaLLM()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting Vega LLM API...")
    logger.info(f"  Model: {MODEL_ID}")
    logger.info(f"  Device: {DEVICE}")
    
    # Try to preload model
    try:
        llm.load_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")
        logger.warning("Model will be loaded on first request.")
    
    yield
    
    logger.info("Shutting down Vega LLM API...")

app = FastAPI(
    title="Vega LLM API",
    description="Language model API powered by Qwen3",
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
# API Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=llm.is_loaded,
        device=llm.device,
        model_id=llm.model_id
    )

@app.get("/info")
async def get_info():
    """Get model information."""
    return {
        "model_id": llm.model_id,
        "device": llm.device,
        "model_loaded": llm.is_loaded,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "max_tokens": MAX_NEW_TOKENS,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat completion endpoint."""
    try:
        response, tokens, finish_reason = llm.chat(
            messages=request.messages,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens or MAX_NEW_TOKENS,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return ChatResponse(
            response=response,
            model=llm.model_id,
            tokens_generated=tokens,
            finish_reason=finish_reason
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Text completion endpoint."""
    try:
        text, tokens = llm.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return GenerateResponse(
            text=text,
            model=llm.model_id,
            tokens_generated=tokens
        )
    
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vega LLM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
