import os
import logging
from typing import Dict, Any, Optional, List
from engine_args import DEFAULT_ARGS, match_vllm_args, get_local_args


def _get_default_served_name(model_name: str) -> str:
    """Generate a default served name from model path."""
    # Convert model path to served name (e.g., meta-llama/Llama-2-7b -> llama-2-7b)
    if "/" in model_name:
        return model_name.split("/")[-1].lower()
    return model_name.lower()


def _parse_indexed_env_vars(prefix: str) -> Dict[str, Dict[str, str]]:
    """Parse indexed environment variables like MODEL_1_NAME, MODEL_2_NAME, etc."""
    models = {}
    index = 1
    
    while True:
        model_name_key = f"{prefix}_{index}_NAME"
        model_name = os.getenv(model_name_key)
        
        if not model_name:
            break
            
        # Base model config
        model_config = {
            "name": model_name,
            "served_name": os.getenv(f"{prefix}_{index}_SERVED_NAME") or _get_default_served_name(model_name),
            "tokenizer": os.getenv(f"{prefix}_{index}_TOKENIZER"),
            "revision": os.getenv(f"{prefix}_{index}_REVISION"),
            "tokenizer_revision": os.getenv(f"{prefix}_{index}_TOKENIZER_REVISION"),
            "trust_remote_code": os.getenv(f"{prefix}_{index}_TRUST_REMOTE_CODE", "false").lower() == "true",
            "max_model_len": int(os.getenv(f"{prefix}_{index}_MAX_MODEL_LEN", "0")) or None,
            "quantization": os.getenv(f"{prefix}_{index}_QUANTIZATION"),
            "dtype": os.getenv(f"{prefix}_{index}_DTYPE"),
            "gpu_memory_utilization": float(os.getenv(f"{prefix}_{index}_GPU_MEMORY_UTILIZATION", "0")) or None,
            "tensor_parallel_size": int(os.getenv(f"{prefix}_{index}_TENSOR_PARALLEL_SIZE", "0")) or None,
        }
        
        # Remove None values to avoid overriding defaults
        model_config = {k: v for k, v in model_config.items() if v is not None and v != ""}
        
        models[model_name] = model_config
        index += 1
    
    return models


def _parse_comma_separated_models(model_names_str: str) -> Dict[str, Dict[str, str]]:
    """Parse comma-separated model names and create basic config for each."""
    models = {}
    model_names = [name.strip() for name in model_names_str.split(",") if name.strip()]
    
    for model_name in model_names:
        models[model_name] = {
            "name": model_name,
            "served_name": _get_default_served_name(model_name)
        }
    
    return models


def _get_single_model_config() -> Dict[str, Dict[str, Any]]:
    """Get single model configuration using existing MODEL_NAME variable."""
    model_name = os.getenv("MODEL_NAME", "facebook/opt-125m")
    served_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or _get_default_served_name(model_name)
    
    return {
        model_name: {
            "name": model_name,
            "served_name": served_name,
        }
    }


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Parse model configurations from environment variables.
    
    Supports three formats (in order of precedence):
    1. Indexed format: MODEL_1_NAME, MODEL_2_NAME, etc.
    2. Comma-separated format: MODEL_NAMES=model1,model2,model3
    3. Single model format: MODEL_NAME (backward compatibility)
    
    Returns:
        Dictionary mapping model names to their configurations
    """
    # Try indexed format first
    indexed_models = _parse_indexed_env_vars("MODEL")
    if indexed_models:
        logging.info(f"Using indexed model configuration with {len(indexed_models)} models")
        return indexed_models
    
    # Try comma-separated format
    model_names_str = os.getenv("MODEL_NAMES")
    if model_names_str:
        comma_models = _parse_comma_separated_models(model_names_str)
        logging.info(f"Using comma-separated model configuration with {len(comma_models)} models")
        return comma_models
    
    # Fall back to single model format
    single_model = _get_single_model_config()
    logging.info(f"Using single model configuration: {list(single_model.keys())[0]}")
    return single_model


def get_engine_args_for_model(model_config: Dict[str, Any]) -> Any:
    """
    Create engine arguments for a specific model configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        AsyncEngineArgs instance for the model
    """
    # Start with default args
    args = DEFAULT_ARGS.copy()
    
    # Add environment variables
    args.update(os.environ)
    
    # Add local args (for baked models)
    args.update(get_local_args())
    
    # Override with model-specific configuration
    model_name = model_config["name"]
    args["MODEL_NAME"] = model_name
    
    # Apply model-specific overrides if they exist
    for key, value in model_config.items():
        if key in ["name", "served_name"]:
            continue  # Skip metadata fields
        
        # Map to environment variable format
        env_key = key.upper()
        if env_key in ["TOKENIZER", "REVISION", "TOKENIZER_REVISION", "TRUST_REMOTE_CODE", 
                      "MAX_MODEL_LEN", "QUANTIZATION", "DTYPE", "GPU_MEMORY_UTILIZATION",
                      "TENSOR_PARALLEL_SIZE"]:
            args[env_key] = value
    
    # Match and validate vLLM args
    return match_vllm_args(args)


def get_default_model_name() -> str:
    """Get the name of the default/primary model."""
    configs = get_model_configs()
    return next(iter(configs.keys()))


def get_served_model_names() -> List[str]:
    """Get list of all served model names for OpenAI API."""
    configs = get_model_configs()
    return [config["served_name"] for config in configs.values()]


def get_model_name_by_served_name(served_name: str) -> Optional[str]:
    """Find model name by served name (for OpenAI API routing)."""
    configs = get_model_configs()
    for model_name, config in configs.items():
        if config["served_name"] == served_name:
            return model_name
    return None


def validate_model_exists(model_name: str) -> bool:
    """Check if a model exists in the configuration."""
    configs = get_model_configs()
    return model_name in configs