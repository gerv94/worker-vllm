import time
from typing import List
from vllm.entrypoints.openai.protocol import ModelList, ModelCard
from model_config import get_model_configs


class MultiModelServingModels:
    """
    Multi-model version of OpenAI serving models endpoint.
    Returns all configured models instead of just one.
    """
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._model_configs = get_model_configs()
        
    async def show_available_models(self) -> ModelList:
        """Show all available models in OpenAI format."""
        model_cards = []
        current_time = int(time.time())
        
        for model_name, config in self._model_configs.items():
            served_name = config["served_name"]
            
            model_card = ModelCard(
                id=served_name,
                created=current_time,
                object="model",
                owned_by="vllm"
            )
            model_cards.append(model_card)
        
        return ModelList(data=model_cards)