#!/usr/bin/env python3
"""
Vega Ingress API Test Suite

Tests all endpoints via the ingress API gateway.

Usage:
    python test_ingress_api.py [--host HOST] [--port PORT] [--save-audio]
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install with: pip install requests")
    sys.exit(1)

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_HOST = "192.168.86.48"
DEFAULT_PORT = 8080

# Test data
TEST_TEXT_SHORT = "Hello, I am Vega."
TEST_TEXT_MEDIUM = "The quick brown fox jumps over the lazy dog. This is a test of the voice synthesis system."
TEST_CHAT_MESSAGE = "What is the capital of France?"
TEST_GENERATE_PROMPT = "Once upon a time, in a land far away,"

# ==============================================================================
# Test Results
# ==============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str
    response_data: Optional[dict] = None


class TestRunner:
    def __init__(self, base_url: str, save_audio: bool = False):
        self.base_url = base_url
        self.save_audio = save_audio
        self.results: list[TestResult] = []
        self.output_dir = Path(__file__).parent / "output"
        
        if save_audio:
            self.output_dir.mkdir(exist_ok=True)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        return requests.request(method, url, **kwargs)
    
    def _test(self, name: str, method: str, endpoint: str, 
              expected_status: int = 200, **kwargs) -> TestResult:
        """Run a single test."""
        start = time.perf_counter()
        try:
            response = self._make_request(method, endpoint, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            
            passed = response.status_code == expected_status
            try:
                data = response.json()
            except:
                data = {"raw": response.text[:500] if response.text else None}
            
            if passed:
                msg = f"✓ {response.status_code} OK ({duration:.0f}ms)"
            else:
                msg = f"✗ Expected {expected_status}, got {response.status_code}"
            
            result = TestResult(name, passed, duration, msg, data)
            
        except requests.exceptions.ConnectionError as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name, False, duration, f"✗ Connection error: {e}", None)
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name, False, duration, f"✗ Error: {e}", None)
        
        self.results.append(result)
        return result

    # ==========================================================================
    # Health & Info Tests
    # ==========================================================================
    
    def test_health(self) -> TestResult:
        """Test /health endpoint."""
        result = self._test("Health Check", "GET", "/health")
        if result.passed and result.response_data:
            backends = result.response_data.get("backends", {})
            tts_status = backends.get("tts", {}).get("status", "unknown")
            llm_status = backends.get("llm", {}).get("status", "unknown")
            result.message += f" [TTS: {tts_status}, LLM: {llm_status}]"
        return result
    
    def test_info(self) -> TestResult:
        """Test /info endpoint."""
        return self._test("Info", "GET", "/info")
    
    # ==========================================================================
    # TTS Health & Info (via proxy)
    # ==========================================================================
    
    def test_tts_health(self) -> TestResult:
        """Test /tts/health endpoint."""
        result = self._test("TTS Health (via ingress)", "GET", "/tts/health")
        if result.passed and result.response_data:
            model_loaded = result.response_data.get("model_loaded", False)
            device = result.response_data.get("device", "unknown")
            result.message += f" [model_loaded: {model_loaded}, device: {device}]"
        return result
    
    def test_tts_info(self) -> TestResult:
        """Test /tts/info endpoint."""
        return self._test("TTS Info (via ingress)", "GET", "/tts/info")
    
    # ==========================================================================
    # LLM Health & Info (via proxy)
    # ==========================================================================
    
    def test_llm_health(self) -> TestResult:
        """Test /llm/health endpoint."""
        result = self._test("LLM Health (via ingress)", "GET", "/llm/health")
        if result.passed and result.response_data:
            model_id = result.response_data.get("model_id", "unknown")
            result.message += f" [model: {model_id}]"
        return result
    
    def test_llm_info(self) -> TestResult:
        """Test /llm/info endpoint."""
        return self._test("LLM Info (via ingress)", "GET", "/llm/info")
    
    # ==========================================================================
    # TTS Synthesis Tests
    # ==========================================================================
    
    def test_synthesize_root(self) -> TestResult:
        """Test /synthesize endpoint (root shortcut)."""
        result = self._test(
            "Synthesize (root)",
            "POST",
            "/synthesize",
            json={"text": TEST_TEXT_SHORT},
            timeout=120
        )
        # Note: May return 500 if HuggingFace token not configured
        if not result.passed and result.response_data:
            detail = result.response_data.get("detail", "")
            if "Token is required" in str(detail) or "token" in str(detail).lower():
                result.message += " [HF token required - config issue]"
        elif result.passed and result.response_data:
            result.message += " [WAV audio returned]"
        return result
    
    def test_synthesize_via_tts(self) -> TestResult:
        """Test /tts/synthesize endpoint."""
        result = self._test(
            "Synthesize (via /tts)",
            "POST",
            "/tts/synthesize",
            json={"text": TEST_TEXT_SHORT},
            timeout=120
        )
        return result
    
    def test_synthesize_b64_root(self) -> TestResult:
        """Test /synthesize/b64 endpoint (base64 response)."""
        result = self._test(
            "Synthesize Base64 (root)",
            "POST",
            "/synthesize/b64",
            json={"text": TEST_TEXT_SHORT},
            timeout=120
        )
        if result.passed and result.response_data:
            audio_b64 = result.response_data.get("audio_base64", "")
            duration = result.response_data.get("duration_seconds", 0)
            gen_time = result.response_data.get("generation_time_seconds", 0)
            result.message += f" [duration: {duration:.2f}s, gen_time: {gen_time:.2f}s]"
            
            # Save audio if requested
            if self.save_audio and audio_b64:
                audio_path = self.output_dir / "test_synthesize_b64.wav"
                audio_path.write_bytes(base64.b64decode(audio_b64))
                result.message += f" [saved: {audio_path.name}]"
        return result
    
    def test_synthesize_b64_via_tts(self) -> TestResult:
        """Test /tts/synthesize/b64 endpoint."""
        result = self._test(
            "Synthesize Base64 (via /tts)",
            "POST",
            "/tts/synthesize/b64",
            json={"text": TEST_TEXT_MEDIUM},
            timeout=120
        )
        if result.passed and result.response_data:
            duration = result.response_data.get("duration_seconds", 0)
            result.message += f" [duration: {duration:.2f}s]"
            
            if self.save_audio:
                audio_b64 = result.response_data.get("audio_base64", "")
                if audio_b64:
                    audio_path = self.output_dir / "test_synthesize_medium.wav"
                    audio_path.write_bytes(base64.b64decode(audio_b64))
        return result
    
    # ==========================================================================
    # LLM Tests
    # ==========================================================================
    
    def test_chat_root(self) -> TestResult:
        """Test /chat endpoint (root shortcut)."""
        result = self._test(
            "Chat (root)",
            "POST",
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": TEST_CHAT_MESSAGE}
                ]
            },
            timeout=60
        )
        if result.passed and result.response_data:
            response_text = result.response_data.get("response", "")[:100]
            tokens = result.response_data.get("tokens_generated", 0)
            result.message += f" [tokens: {tokens}, response: '{response_text}...']"
        return result
    
    def test_chat_via_llm(self) -> TestResult:
        """Test /llm/chat endpoint."""
        result = self._test(
            "Chat (via /llm)",
            "POST",
            "/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Be brief."},
                    {"role": "user", "content": "What is 2 + 2?"}
                ],
                "max_tokens": 50,
                "temperature": 0.3
            },
            timeout=60
        )
        if result.passed and result.response_data:
            response_text = result.response_data.get("response", "")[:80]
            result.message += f" [response: '{response_text}']"
        return result
    
    def test_generate_root(self) -> TestResult:
        """Test /generate endpoint (root shortcut)."""
        result = self._test(
            "Generate (root)",
            "POST",
            "/generate",
            json={
                "prompt": TEST_GENERATE_PROMPT,
                "max_tokens": 100
            },
            timeout=60
        )
        if result.passed and result.response_data:
            text = result.response_data.get("text", "")[:100]
            tokens = result.response_data.get("tokens_generated", 0)
            result.message += f" [tokens: {tokens}]"
        return result
    
    def test_generate_via_llm(self) -> TestResult:
        """Test /llm/generate endpoint."""
        result = self._test(
            "Generate (via /llm)",
            "POST",
            "/llm/generate",
            json={
                "prompt": "The meaning of life is",
                "max_tokens": 50,
                "temperature": 0.5
            },
            timeout=60
        )
        return result
    
    # ==========================================================================
    # Spectrogram Analysis Tests (VEGA Radiological Assistant)
    # ==========================================================================
    
    def test_spectrogram_text(self) -> TestResult:
        """Test /spectrogram endpoint with text format."""
        result = self._test(
            "Spectrogram (text format)",
            "POST",
            "/spectrogram",
            json={
                "query": "What is the current radiation status?",
                "response_format": "text",
                "dose_rate": 0.142,
                "cps": 12.5
            },
            timeout=60
        )
        if result.passed and result.response_data:
            response = result.response_data.get("response", "")[:80]
            result.message += f" [VEGA: '{response}...']"
        return result
    
    def test_spectrogram_speech(self) -> TestResult:
        """Test /spectrogram endpoint with speech format."""
        result = self._test(
            "Spectrogram (speech format)",
            "POST",
            "/spectrogram",
            json={
                "query": "Describe the current dose rate",
                "response_format": "speech",
                "dose_rate": 0.25
            },
            timeout=60
        )
        if result.passed and result.response_data:
            fmt = result.response_data.get("response_format", "")
            result.message += f" [format: {fmt}]"
        return result
    
    def test_spectrogram_via_llm(self) -> TestResult:
        """Test /llm/spectrogram endpoint."""
        result = self._test(
            "Spectrogram (via /llm)",
            "POST",
            "/llm/spectrogram",
            json={
                "query": "Is this radiation level safe?",
                "response_format": "text",
                "dose_rate": 0.08,
                "cps": 5.2
            },
            timeout=60
        )
        return result
    
    # ==========================================================================
    # API Alias Tests
    # ==========================================================================
    
    def test_api_tts_alias(self) -> TestResult:
        """Test /api/tts/* alias route."""
        return self._test(
            "API TTS Alias (/api/tts/health)",
            "GET",
            "/api/tts/health"
        )
    
    def test_api_llm_alias(self) -> TestResult:
        """Test /api/llm/* alias route."""
        return self._test(
            "API LLM Alias (/api/llm/health)",
            "GET",
            "/api/llm/health"
        )
    
    # ==========================================================================
    # Error Handling Tests
    # ==========================================================================
    
    def test_404_endpoint(self) -> TestResult:
        """Test 404 for non-existent endpoint."""
        return self._test(
            "404 Not Found",
            "GET",
            "/nonexistent",
            expected_status=404
        )
    
    def test_synthesize_empty_text(self) -> TestResult:
        """Test synthesize with empty text (should fail)."""
        return self._test(
            "Synthesize Empty Text (should fail)",
            "POST",
            "/synthesize",
            json={"text": ""},
            expected_status=422  # Validation error
        )
    
    def test_chat_empty_messages(self) -> TestResult:
        """Test chat with empty messages (behavior varies by implementation)."""
        result = self._test(
            "Chat Empty Messages",
            "POST",
            "/chat",
            json={"messages": []},
            expected_status=200  # LLM may accept empty messages with default system prompt
        )
        return result
    
    # ==========================================================================
    # Run All Tests
    # ==========================================================================
    
    def run_all(self) -> bool:
        """Run all tests and return True if all passed."""
        print("=" * 70)
        print(f"VEGA INGRESS API TEST SUITE")
        print(f"Target: {self.base_url}")
        print("=" * 70)
        print()
        
        # Group 1: Health & Info
        print("─" * 50)
        print("HEALTH & INFO ENDPOINTS")
        print("─" * 50)
        self._run_and_print(self.test_health)
        self._run_and_print(self.test_info)
        print()
        
        # Group 2: TTS Backend
        print("─" * 50)
        print("TTS BACKEND (via proxy)")
        print("─" * 50)
        self._run_and_print(self.test_tts_health)
        self._run_and_print(self.test_tts_info)
        print()
        
        # Group 3: LLM Backend
        print("─" * 50)
        print("LLM BACKEND (via proxy)")
        print("─" * 50)
        self._run_and_print(self.test_llm_health)
        self._run_and_print(self.test_llm_info)
        print()
        
        # Group 4: TTS Synthesis
        print("─" * 50)
        print("TTS SYNTHESIS")
        print("─" * 50)
        self._run_and_print(self.test_synthesize_root)
        self._run_and_print(self.test_synthesize_via_tts)
        self._run_and_print(self.test_synthesize_b64_root)
        self._run_and_print(self.test_synthesize_b64_via_tts)
        print()
        
        # Group 5: LLM Generation
        print("─" * 50)
        print("LLM GENERATION")
        print("─" * 50)
        self._run_and_print(self.test_chat_root)
        self._run_and_print(self.test_chat_via_llm)
        self._run_and_print(self.test_generate_root)
        self._run_and_print(self.test_generate_via_llm)
        print()
        
        # Group 6: Spectrogram Analysis (VEGA Radiological Assistant)
        print("─" * 50)
        print("SPECTROGRAM ANALYSIS (VEGA Radiological)")
        print("─" * 50)
        self._run_and_print(self.test_spectrogram_text)
        self._run_and_print(self.test_spectrogram_speech)
        self._run_and_print(self.test_spectrogram_via_llm)
        print()
        
        # Group 7: API Aliases
        print("─" * 50)
        print("API ALIASES")
        print("─" * 50)
        self._run_and_print(self.test_api_tts_alias)
        self._run_and_print(self.test_api_llm_alias)
        print()
        
        # Group 8: Error Handling
        print("─" * 50)
        print("ERROR HANDLING")
        print("─" * 50)
        self._run_and_print(self.test_404_endpoint)
        self._run_and_print(self.test_synthesize_empty_text)
        self._run_and_print(self.test_chat_empty_messages)
        print()
        
        # Summary
        return self._print_summary()
    
    def _run_and_print(self, test_func):
        """Run a test and print result."""
        result = test_func()
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}")
        print(f"         {result.message}")
    
    def _print_summary(self) -> bool:
        """Print test summary and return True if all passed."""
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        total_time = sum(r.duration_ms for r in self.results)
        
        print(f"  Total:  {total} tests")
        print(f"  Passed: {passed} ✓")
        print(f"  Failed: {failed} ✗")
        print(f"  Time:   {total_time:.0f}ms ({total_time/1000:.2f}s)")
        print()
        
        if failed > 0:
            print("FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
            print()
        
        all_passed = failed == 0
        status = "ALL TESTS PASSED ✓" if all_passed else f"SOME TESTS FAILED ({failed}/{total}) ✗"
        print(f"Result: {status}")
        print("=" * 70)
        
        return all_passed


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Vega Ingress API endpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host", 
        default=DEFAULT_HOST,
        help="Ingress host"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=DEFAULT_PORT,
        help="Ingress port"
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio files to output directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only health check tests (fast)"
    )
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    runner = TestRunner(base_url, save_audio=args.save_audio)
    
    if args.quick:
        # Quick mode - just health checks
        print(f"Quick test mode - checking {base_url}")
        print("-" * 40)
        runner.test_health()
        runner.test_tts_health()
        runner.test_llm_health()
        runner._print_summary()
    else:
        # Full test suite
        all_passed = runner.run_all()
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
