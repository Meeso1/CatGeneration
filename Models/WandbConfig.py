from dataclasses import dataclass
import os
import pickle
import shutil
import tempfile
from typing import Any
import wandb
import torch
import io


@dataclass
class WandbConfig:
    PROJECT_NAME = "cat-generation"
    
    experiment_name: str
    artifact_name: str | None = None
    # Set init_project to False to manually call wandb.init()/wandb.finish() to be able to call train() multiple times or something
    init_project: bool = True
    
    def init_if_needed(self, model_config: dict[str, Any]) -> None:
        if not self.init_project:
            return
        
        wandb.init(
            project=self.PROJECT_NAME,
            name=self.experiment_name,
            config=model_config,
            settings=wandb.Settings(silent=True)
        )
        
    def log(self, metrics: dict[str, float | int | str]) -> None:
        wandb.log(metrics)
   
    def save_artifact(self, name: str, data: Any) -> None:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "model.pkl")
        
        with open(temp_file_path, "wb") as f:
            pickle.dump(data, f)
        
        artifact = wandb.Artifact(name=name, type="model", description="Model state dict")
        artifact.add_file(temp_file_path)
        
        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
        
        shutil.rmtree(temp_dir)
        
    def finish_and_save_if_needed(self, data: Any) -> None:
        if not self.init_project:
            return
        
        if self.artifact_name is not None:
            self.save_artifact(self.artifact_name, data)
        
        wandb.finish()

    @staticmethod
    def get_artifact_from_wandb(artifact_name: str, version: str = "latest") -> Any:
        artifact = wandb.Api().artifact(f"{WandbConfig.PROJECT_NAME}/{artifact_name}:{version}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(root=temp_dir)
            model_path = os.path.join(artifact_dir, "model.pkl")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            with open(model_path, "rb") as f:
                return  WandbConfig.TorchDeviceUnpickler(f, device).load()

    @staticmethod
    def get_run_metrics(run_name: str) -> dict[str, list[float | int | str]]:
        """
        Retrieves metrics from a Wandb run and returns them in a format suitable for plotting.
        Finds the latest run with the given name.
        
        Args:
            run_name: The name of the run to retrieve metrics from
            
        Returns:
            Dictionary where keys are metric names and values are lists of metric values over time
            
        Raises:
            ValueError: If no run with the given name is found, with a list of available runs
        """
        api = wandb.Api()
        
        try:
            # Get all runs from the project
            runs = api.runs(WandbConfig.PROJECT_NAME)
            
            # Find runs with matching name
            matching_runs = [run for run in runs if run.name == run_name]
            
            if not matching_runs:
                # No matching runs found, show available runs
                available_runs = [run.name for run in runs if run.name]
                available_runs_str = "\n  - ".join(available_runs) if available_runs else "No runs found"
                
                raise ValueError(
                    f"No run with name '{run_name}' found in project '{WandbConfig.PROJECT_NAME}'.\n"
                    f"Available runs:\n  - {available_runs_str}"
                )
            
            # Use the latest run (first in the list, as wandb returns runs sorted by creation time desc)
            run = matching_runs[0]
            
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            raise ValueError(
                f"Error retrieving runs from project '{WandbConfig.PROJECT_NAME}': {e}"
            )
        
        # Get the history as a pandas DataFrame
        history = run.history()
        
        # Convert to dict of lists, excluding system columns
        metrics = {}
        for column in history.columns:
            # Skip system columns like _step, _runtime, etc.
            if not column.startswith('_'):
                # Convert to list
                values = history[column].tolist()
                if values:  # Only include non-empty lists
                    metrics[column] = values
        
        return metrics

    class TorchDeviceUnpickler(pickle.Unpickler):
        def __init__(self, file: io.BytesIO, device: torch.device):
            self.device = device
            super().__init__(file)
        
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
            else:
                return super().find_class(module, name)
