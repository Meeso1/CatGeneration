from dataclasses import dataclass
import os
import pickle
import shutil
import tempfile
from typing import Any
import wandb

@dataclass
class WandbConfig:
    project: str
    experiment_name: str
    config_name: str
    artifact_name: str | None = None
    # Set init_project to False to manually call wandb.init()/wandb.finish() to be able to call train() multiple times or something
    init_project: bool = True
    
    def init_if_needed(self) -> None:
        if not self.init_project:
            return
        
        wandb.init(
            project=self.project,
            name=self.experiment_name,
            config=self.config_name,
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
    def get_artifact_from_wandb(project: str, artifact_name: str, version: str = "latest") -> Any:
        artifact = wandb.Api().artifact(f"{project}/{artifact_name}:{version}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(root=temp_dir)
            model_path = os.path.join(artifact_dir, "model.pkl")
            
            with open(model_path, "rb") as f:
                return pickle.load(f)
