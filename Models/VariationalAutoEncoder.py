from dataclasses import dataclass
from typing import Any, Self
import numpy as np
import torch.optim.adamw
from FidScorer import FidScorer
from Models.ModelBase import ModelBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class VariationalAutoEncoder(ModelBase):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: list[int] = [32, 64, 128, 256, 512],
        learning_rate: float = 1e-3,
        lr_decay: float = 1.0,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        weight_decay: float = 1e-2,
        kl_weight: float = 1.0,
        print_every: int | None = None,
        fid_scorer: FidScorer | None = None,
        n_images_for_fid: int = 1000
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        
        self.print_every = print_every
        self.fid_scorer = fid_scorer
        self.n_images_for_fid = n_images_for_fid
        self.fid_metrics_for_real_images: tuple[np.ndarray, np.ndarray] | None = None
        
        self.model: VariationalAutoEncoder.VariationalAutoEncoderModule | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.ExponentialLR | None = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    def train(
        self, 
        images: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        if self.model is None:
            self._initialize_model()
            
        loader = self._validate_and_make_loader(images, batch_size)
        
        self.model.train()
        if self.wandb_config is not None:
            self.wandb_config.init_if_needed()
        
        for epoch in range(1, epochs + 1):
            metrics = self._train_epoch(loader)
            self._print_metrics_if_needed(metrics, epoch, epochs)
            
            # Step the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.wandb_config is not None:
                self.wandb_config.log(metrics.to_dict())
        
        if self.wandb_config is not None:
            self.wandb_config.finish_and_save_if_needed(self.get_state_dict())
    
    def _initialize_model(self) -> None:
        self.model = self.VariationalAutoEncoderModule(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
    def _validate_and_make_loader(self, X: np.ndarray, batch_size: int) -> DataLoader:
        self._validate_train_data(X)
        
        # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
        X_transposed = np.transpose(X, (0, 3, 1, 2))
        dataset = TensorDataset(torch.from_numpy(X_transposed).float().to(self.device))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    def _validate_train_data(self, X: np.ndarray) -> None:     
        if X.ndim != 4 or X.shape[1:] != (64, 64, 3):
            raise ValueError(f"X must be a 4D array with shape (n, 64, 64, 3) - got shape: {X.shape}")
        
        if np.max(X) > 1 or np.min(X) < 0:
            raise ValueError(f"X must be normalized to [0, 1] - got max: {np.max(X)}, min: {np.min(X)}")
    
    @dataclass
    class EpochMetrics:
        total_loss: float
        recon_loss: float
        kl_loss: float
        fid_score: float | None
        
        def to_dict(self) -> dict[str, float]:
            return {
                "total_loss": self.total_loss,
                "recon_loss": self.recon_loss,
                "kl_loss": self.kl_loss,
                "fid_score": self.fid_score
            }
    
    def _train_epoch(self, loader: DataLoader) -> EpochMetrics:
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0
        
        for batch in loader:
            x = batch[0]  # Get images from the batch
            batch_size = x.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_x, mu, log_var = self.model(x)
            
            # Calculate VAE loss
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss with KL weight
            loss = recon_loss + self.kl_weight * kl_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += batch_size
        
        return VariationalAutoEncoder.EpochMetrics(
            total_loss=total_loss / total_samples,
            recon_loss=total_recon_loss / total_samples,
            kl_loss=total_kl_loss / total_samples,
            fid_score=self._calculate_fid_score(loader)
        )
        
    def _calculate_fid_score(self, loader: DataLoader) -> float | None:
        if self.fid_scorer is None:
            return None
        
        if self.fid_metrics_for_real_images is None:
            features = self.fid_scorer.extract_features_from_dataloader(loader)
            self.fid_metrics_for_real_images = self.fid_scorer.calculate_statistics(features)
            
        generated_images = self.generate(self.n_images_for_fid)
        return self.fid_scorer.calculate_fid(self.fid_metrics_for_real_images, generated_images)
    
    def _print_metrics_if_needed(self, metrics: EpochMetrics, epoch: int, total_epochs: int) -> None:
        if self.print_every is None or epoch % self.print_every != 0:
            return
        
        current_lr = self.optimizer.param_groups[0]['lr']
        max_epochs_str_len = len(str(total_epochs))
        
        fid_score_str = f", FID: {metrics.fid_score:.4f}" if metrics.fid_score is not None else ""
        
        print(f"Epoch {epoch:{max_epochs_str_len}d}/{total_epochs}: Total Loss: {metrics.total_loss:.4f}, (Recon: {metrics.recon_loss:.4f} + KL: {metrics.kl_loss:.4f}), LR: {current_lr:.6f}{fid_score_str}")
    
    def generate(self, n_samples: int) -> np.ndarray:
        return self.generate_from_latent(np.random.randn(n_samples, self.latent_dim))
    
    def generate_from_latent(self, latent_vectors: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not initialized - call `train()` first")
        
        self.model.eval()
        
        with torch.no_grad():
            z = torch.from_numpy(latent_vectors).float().to(self.device)
            generated_images = self.model.decode(z).cpu().numpy()
            
            # Convert from PyTorch (N, C, H, W) to expected (N, H, W, C) format
            generated_images = np.transpose(generated_images, (0, 2, 3, 1))
            
        return generated_images

    def get_model_config_for_wandb(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "lr_decay": self.lr_decay,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "weight_decay": self.weight_decay,
            "kl_weight": self.kl_weight
        }
    
    def get_state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "lr_decay": self.lr_decay,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "weight_decay": self.weight_decay,
            "kl_weight": self.kl_weight,
            
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            
            "wandb_config": self.wandb_config
        }
    
    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> Self:
        loaded_model = VariationalAutoEncoder(
            learning_rate=state_dict["learning_rate"],
            latent_dim=state_dict["latent_dim"],
            hidden_dims=state_dict["hidden_dims"],
            beta_1=state_dict["beta_1"],
            beta_2=state_dict["beta_2"],
            weight_decay=state_dict["weight_decay"],
            kl_weight=state_dict["kl_weight"],
            lr_decay=state_dict["lr_decay"]
        ).with_wandb(state_dict["wandb_config"])
        
        loaded_model.model.load_state_dict(state_dict["model"])
        loaded_model.optimizer.load_state_dict(state_dict["optimizer"])
        loaded_model.scheduler.load_state_dict(state_dict["scheduler"])
        
        return loaded_model
    
    class VariationalAutoEncoderModule(nn.Module):
        def __init__(
            self, 
            latent_dim: int = 128,
            hidden_dims: list[int] = [32, 64, 128, 256, 512]
        ) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims
            
            encoder_output_img_width = 64 // 2 ** len(hidden_dims)
            encoder_output_size = hidden_dims[-1] * encoder_output_img_width ** 2
            
            self.encoder = self._build_encoder(hidden_dims)
            
            # Latent space layers
            self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
            self.fc_var = nn.Linear(encoder_output_size, latent_dim)
            
            self.decoder_input, self.decoder = self._build_decoder(latent_dim, hidden_dims, encoder_output_size)
            
        def _build_encoder(self, hidden_dims: list[int]) -> nn.Module:
            encoder_modules = []
            in_channels = 3
            for h_dim in hidden_dims:
                encoder_modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU()
                    )
                )
                in_channels = h_dim
            
            return nn.Sequential(*encoder_modules)
        
        def _build_decoder(self, latent_dim: int, hidden_dims: list[int], encoder_output_size: int) -> nn.Module:
            decoder_input = nn.Linear(latent_dim, encoder_output_size)
            
            decoder_modules = []
            reversed_hidden_dims = list(reversed(hidden_dims))
            for i in range(len(reversed_hidden_dims) - 1):
                decoder_modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            reversed_hidden_dims[i],
                            reversed_hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1
                        ),
                        nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                        nn.LeakyReLU()
                    )
                )
            
            final_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    reversed_hidden_dims[-1],
                    3,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.Sigmoid()
            )
            
            return decoder_input, nn.Sequential(*decoder_modules, final_layer)
            
        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Encode input to latent space.
            Returns: (mu, log_var)
            """
            result = self.encoder(x)
            result = torch.flatten(result, start_dim=1)
            
            mu = self.fc_mu(result)
            log_var = self.fc_var(result)
            
            return mu, log_var
        
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """
            Decode from latent space to image.
            """
            result = self.decoder_input(z)
            result = result.view(-1, self.hidden_dims[-1], 2, 2)
            result = self.decoder(result)
            return result
        
        def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """
            Reparameterization trick: z = mu + std * epsilon
            """
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass through the VAE.
            Returns: (reconstructed_x, mu, log_var)
            """
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            reconstructed_x = self.decode(z)
            return reconstructed_x, mu, log_var
