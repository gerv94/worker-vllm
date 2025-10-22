import os
import logging
import threading
import time
from typing import Dict, Optional, Tuple
from vllm import AsyncEngineArgs

from model_config import get_model_configs, get_engine_args_for_model, validate_model_exists, get_default_model_name
from engine import vLLMEngine, OpenAIvLLMEngine


class ModelManager:
    """
    Manages multiple model engines with lazy loading and thread safety.
    Supports kvcached for elastic GPU memory sharing between models.
    """
    
    def __init__(self):
        self._engines: Dict[str, Tuple[vLLMEngine, OpenAIvLLMEngine]] = {}
        self._model_configs = get_model_configs()
        self._load_lock = threading.RLock()
        self._engine_stats: Dict[str, Dict[str, any]] = {}
        
        # Log configuration
        model_names = list(self._model_configs.keys())
        logging.info(f"ModelManager initialized with {len(model_names)} models: {model_names}")
        
        # Set up stats tracking
        for model_name in model_names:
            self._engine_stats[model_name] = {
                "loaded": False,
                "load_time": None,
                "last_accessed": None,
                "total_requests": 0
            }

    def get_available_models(self) -> Dict[str, Dict]:
        """Get all configured models and their metadata."""
        return self._model_configs.copy()
    
    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model names."""
        with self._load_lock:
            return list(self._engines.keys())
    
    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics for all models."""
        return self._engine_stats.copy()

    def _create_engine_pair(self, model_name: str) -> Tuple[vLLMEngine, OpenAIvLLMEngine]:
        """Create a vLLM engine and OpenAI wrapper for the specified model."""
        if model_name not in self._model_configs:
            raise ValueError(f"Model '{model_name}' not found in configuration")
            
        model_config = self._model_configs[model_name]
        
        logging.info(f"Creating engine for model '{model_name}'...")
        start_time = time.time()
        
        # Create engine args specific to this model
        engine_args = get_engine_args_for_model(model_config)
        
        # Create vLLM engine with model-specific args
        # Temporarily override the global engine args generation
        import engine_args as engine_args_module
        original_get_engine_args = engine_args_module.get_engine_args
        
        def get_model_specific_args():
            return AsyncEngineArgs(**engine_args)
        
        engine_args_module.get_engine_args = get_model_specific_args
        
        try:
            vllm_engine = vLLMEngine()
        finally:
            # Restore original function
            engine_args_module.get_engine_args = original_get_engine_args
        
        # Initialize tokenizer if needed
        if vllm_engine.engine_args.tokenizer_mode != 'mistral':
            from tokenizer import TokenizerWrapper
            vllm_engine.tokenizer = TokenizerWrapper(
                vllm_engine.engine_args.tokenizer or vllm_engine.engine_args.model,
                vllm_engine.engine_args.tokenizer_revision,
                vllm_engine.engine_args.trust_remote_code
            )
        
        # Create OpenAI wrapper
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        
        load_time = time.time() - start_time
        
        # Update stats
        self._engine_stats[model_name].update({
            "loaded": True,
            "load_time": load_time,
            "last_accessed": time.time()
        })
        
        logging.info(f"Model '{model_name}' loaded successfully in {load_time:.2f}s")
        
        return vllm_engine, openai_engine

    def get_or_create_engine(self, model_name: Optional[str] = None) -> Tuple[vLLMEngine, OpenAIvLLMEngine]:
        """
        Get or create engine pair for the specified model.
        
        Args:
            model_name: Name of the model to load. If None, uses default model.
            
        Returns:
            Tuple of (vLLMEngine, OpenAIvLLMEngine) for the model
            
        Raises:
            ValueError: If model not found in configuration
        """
        # Use default model if none specified
        if model_name is None:
            model_name = get_default_model_name()
        
        # Validate model exists
        if not validate_model_exists(model_name):
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        # Update access stats
        if model_name in self._engine_stats:
            self._engine_stats[model_name]["last_accessed"] = time.time()
            self._engine_stats[model_name]["total_requests"] += 1
        
        # Check if already loaded
        with self._load_lock:
            if model_name in self._engines:
                logging.debug(f"Using existing engine for model '{model_name}'")
                return self._engines[model_name]
            
            # Create new engine pair
            try:
                engine_pair = self._create_engine_pair(model_name)
                self._engines[model_name] = engine_pair
                return engine_pair
                
            except Exception as e:
                logging.error(f"Failed to create engine for model '{model_name}': {e}")
                raise e

    def get_vllm_engine(self, model_name: Optional[str] = None) -> vLLMEngine:
        """Get vLLM engine for the specified model."""
        vllm_engine, _ = self.get_or_create_engine(model_name)
        return vllm_engine
    
    def get_openai_engine(self, model_name: Optional[str] = None) -> OpenAIvLLMEngine:
        """Get OpenAI engine for the specified model."""
        _, openai_engine = self.get_or_create_engine(model_name)
        return openai_engine

    def resolve_model_for_openai_request(self, served_model_name: str) -> str:
        """
        Resolve served model name to internal model name for OpenAI API requests.
        
        Args:
            served_model_name: The model name from OpenAI API request
            
        Returns:
            Internal model name
            
        Raises:
            ValueError: If served model name not found
        """
        # Find model by served name
        for model_name, config in self._model_configs.items():
            if config["served_name"] == served_model_name:
                return model_name
        
        # If not found by served name, try direct match
        if validate_model_exists(served_model_name):
            return served_model_name
            
        raise ValueError(f"Model '{served_model_name}' not found in configuration")

    def get_max_concurrency(self) -> int:
        """Get maximum concurrency across all models (for RunPod compatibility)."""
        # Return the concurrency of the default model or a reasonable default
        try:
            default_engine = self.get_vllm_engine()
            return default_engine.max_concurrency
        except Exception:
            # Fallback if no engines loaded yet
            return int(os.getenv("MAX_CONCURRENCY", "300"))

    def health_check(self) -> Dict[str, any]:
        """Perform health check on all loaded models."""
        health_status = {
            "status": "healthy",
            "loaded_models": len(self._engines),
            "configured_models": len(self._model_configs),
            "models": {}
        }
        
        with self._load_lock:
            for model_name in self._model_configs.keys():
                model_status = {
                    "configured": True,
                    "loaded": model_name in self._engines,
                    "stats": self._engine_stats.get(model_name, {})
                }
                
                # Basic engine health check if loaded
                if model_name in self._engines:
                    try:
                        vllm_engine, openai_engine = self._engines[model_name]
                        model_status["engine_healthy"] = True
                        model_status["served_name"] = self._model_configs[model_name]["served_name"]
                    except Exception as e:
                        model_status["engine_healthy"] = False
                        model_status["error"] = str(e)
                        health_status["status"] = "degraded"
                
                health_status["models"][model_name] = model_status
        
        return health_status

    def cleanup(self):
        """Cleanup resources (for graceful shutdown)."""
        logging.info("Cleaning up model engines...")
        with self._load_lock:
            for model_name, (vllm_engine, openai_engine) in self._engines.items():
                try:
                    # Note: vLLM engines don't have explicit cleanup methods
                    # but we can log the cleanup
                    logging.info(f"Cleaning up engine for model '{model_name}'")
                except Exception as e:
                    logging.warning(f"Error cleaning up engine for '{model_name}': {e}")
            
            self._engines.clear()
            
        logging.info("Model engine cleanup completed")


# Global model manager instance
_model_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance (singleton pattern)."""
    global _model_manager
    
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    
    return _model_manager


def cleanup_model_manager():
    """Cleanup the global model manager."""
    global _model_manager
    
    if _model_manager is not None:
        _model_manager.cleanup()
        _model_manager = None