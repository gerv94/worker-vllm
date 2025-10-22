import os
import runpod
from utils import JobInput, create_error_response
from model_manager import get_model_manager

async def handler(job):
    job_input = JobInput(job["input"])
    model_manager = get_model_manager()
    
    try:
        # Determine which model to use
        if job_input.openai_route:
            # For OpenAI routes, resolve served model name to internal model name
            if job_input.model:
                model_name = model_manager.resolve_model_for_openai_request(job_input.model)
            else:
                # Default to first model if no model specified in OpenAI request
                model_name = None
            engine = model_manager.get_openai_engine(model_name)
        else:
            # For native vLLM routes, use model directly
            model_name = job_input.model  # Can be None for default model
            engine = model_manager.get_vllm_engine(model_name)
        
        # Generate response using the selected engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
            
    except ValueError as e:
        # Model not found or configuration error
        yield {"error": create_error_response(str(e)).model_dump()}
    except Exception as e:
        # Other errors during engine creation or generation
        yield {"error": create_error_response(f"Internal error: {str(e)}").model_dump()}

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: get_model_manager().get_max_concurrency(),
        "return_aggregate_stream": True,
    }
)
