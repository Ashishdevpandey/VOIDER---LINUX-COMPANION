"""
STT Service for VOIDER
Handles Speech-to-Text using faster-whisper (local) and cloud APIs (Groq/OpenAI)
"""

import logging
import os
import threading
from typing import Optional, Union
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class STTService:
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        provider: str = "local",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._model_ready = threading.Event()
        self._model_load_error: Optional[str] = None
        
        logger.info(f"STT Service initialized with provider: {provider}")

    def preload_model_async(self):
        """Start loading the Whisper model in a background thread"""
        if self.provider != "local":
            return
            
        def _load():
            try:
                self._load_local_model(sync=True)
            except Exception as e:
                self._model_load_error = str(e)
                logger.error(f"Background STT load failed: {e}")
            finally:
                self._model_ready.set()
                
        thread = threading.Thread(target=_load, daemon=True, name="stt-model-loader")
        thread.start()
        return thread

    def _load_local_model(self, sync: bool = False):
        """Lazy load the faster-whisper model"""
        if self._model is None:
            if not sync and not self._model_ready.is_set():
                logger.info("Waiting for STT model to finish loading...")
                self._model_ready.wait(timeout=300) # Wait up to 5 min
                
            if self._model_load_error:
                raise RuntimeError(f"STT model failed to load: {self._model_load_error}")
                
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                    logger.info(f"Loading local Whisper model: {self.model_size} on {self.device}")
                    self._model = WhisperModel(
                        self.model_size, 
                        device=self.device, 
                        compute_type=self.compute_type
                    )
                    self._model_ready.set()
                except Exception as e:
                    logger.error(f"Failed to load local Whisper model: {e}")
                    raise

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Transcribe an audio file using the configured provider"""
        if self.provider == "groq":
            return self._transcribe_groq(audio_path)
        elif self.provider == "openai":
            return self._transcribe_openai(audio_path)
        else:
            return self._transcribe_local(audio_path)

    def _transcribe_local(self, audio_path: Union[str, Path]) -> str:
        """Transcribe using local faster-whisper"""
        self._load_local_model()
        
        segments, info = self._model.transcribe(str(audio_path), beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text
            
        return text.strip()

    def _transcribe_groq(self, audio_path: Union[str, Path]) -> str:
        """Transcribe using Groq API"""
        if not self.api_key:
            logger.warning("Groq API key not provided for STT, falling back to local")
            return self._transcribe_local(audio_path)
            
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            with open(audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3",
                    response_format="text",
                )
            return transcription
        except Exception as e:
            logger.error(f"Groq STT error: {e}")
            return self._transcribe_local(audio_path)

    def _transcribe_openai(self, audio_path: Union[str, Path]) -> str:
        """Transcribe using OpenAI API"""
        if not self.api_key:
            logger.warning("OpenAI API key not provided for STT, falling back to local")
            return self._transcribe_local(audio_path)
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            with open(audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=file,
                    response_format="text"
                )
            return transcription
        except Exception as e:
            logger.error(f"OpenAI STT error: {e}")
            return self._transcribe_local(audio_path)

# Singleton factory
_stt_service: Optional[STTService] = None

def get_stt_service(
    provider: str = "local",
    api_key: Optional[str] = None,
    model_size: str = "tiny.en"
) -> STTService:
    global _stt_service
    if _stt_service is None or _stt_service.provider != provider or _stt_service.model_size != model_size:
        _stt_service = STTService(
            provider=provider, 
            api_key=api_key, 
            model_size=model_size
        )
    return _stt_service
