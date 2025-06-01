from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from Models.WandbConfig import WandbConfig


class ModelBase(ABC):
    def __init__(self) -> None:
        self.wandb_config: WandbConfig | None = None
       
    @abstractmethod
    def train(
        self, 
        images: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        pass        
    
    @abstractmethod
    def generate(self, n_samples: int) -> np.ndarray:
        pass
    
    @abstractmethod
    def generate_from_latent(self, latent_vectors: np.ndarray) -> np.ndarray:
        pass
    
    def with_wandb(self, wandb_config: WandbConfig) -> Self:
        self.wandb_config = wandb_config
        return self

    @abstractmethod
    def get_model_config_for_wandb(self) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        pass
    
    @classmethod
    @abstractmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> Self:
        pass
