"""Regester the models used in the API."""

from typing import Dict, Any

from decision_mining.core.dmn.models.decisionModel import DecisionModel
from decision_mining.core.dmn.models.DMFuzzy import DMFuzzy


class DecisionModelRegestry():
    """Registry of all available models."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self.models: Dict[str, DecisionModel] = {}

    def register(self, model: DecisionModel) -> None:
        """Add a model to the registry.

        Args:
            model (DecisionModel): decision model
        """
        self.models[model.model_id] = model

    def get_model(self, model_id: str) -> DecisionModel:
        """Get a model from the registry.

        Args:
            model_id (str): model id

        Raises:
            ValueError: if the model is not in the registry

        Returns:
            DecisionModel: decision model
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not registered.")
        return self.models[model_id]

    def get_all_models(self) -> DecisionModel:
        """Get all models from the registry.

        Returns:
            DecisionModel: decision model
        """
        return list(self.models.values())

    def get_all_models_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameters from the models in the registry.

        Returns:
            Dict[str, Dict[str, Any]]: parameters
        """
        return {model_id: model.model_parameters for model_id, model in self.models.items()}

    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameters from the models in the registry.

        Returns:
            Dict[str, Dict[str, Any]]: parameters
        """
        return {model_id: model.model_info for model_id, model in self.models.items()}


registered_models = DecisionModelRegestry()
registered_models.register(DMFuzzy())
