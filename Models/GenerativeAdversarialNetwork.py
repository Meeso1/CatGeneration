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


class GenerativeAdversarialNetwork(ModelBase):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims_generator: list[int] = [512, 256, 128, 64, 32],
        hidden_dims_discriminator: list[int] = [32, 64, 128, 256, 512],
        learning_rate_generator: float = 2e-4,
        learning_rate_discriminator: float = 2e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        weight_decay: float = 0.0,
        print_every: int | None = None,
        fid_scorer: FidScorer | None = None,
        n_images_for_fid: int = 1000,
        use_wgan_gp: bool = False,
        gradient_penalty_weight: float = 10.0,
        critic_iterations: int = 5
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims_generator = hidden_dims_generator
        self.hidden_dims_discriminator = hidden_dims_discriminator
        self.learning_rate_generator = learning_rate_generator
        self.learning_rate_discriminator = learning_rate_discriminator
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        
        self.print_every = print_every
        self.fid_scorer = fid_scorer
        self.n_images_for_fid = n_images_for_fid
        self.fid_metrics_for_real_images: tuple[np.ndarray, np.ndarray] | None = None
        
        # WGAN-GP specific parameters
        self.use_wgan_gp = use_wgan_gp
        self.gradient_penalty_weight = gradient_penalty_weight
        self.critic_iterations = critic_iterations
        
        self.generator: GenerativeAdversarialNetwork.GeneratorModule | None = None
        self.discriminator: GenerativeAdversarialNetwork.DiscriminatorModule | None = None
        self.optimizer_generator: torch.optim.Optimizer | None = None
        self.optimizer_discriminator: torch.optim.Optimizer | None = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self, 
        images: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        if self.generator is None or self.discriminator is None:
            self._initialize_model()
            
        loader = self._validate_and_make_loader(images, batch_size)
        
        self.generator.train()
        self.discriminator.train()
        if self.wandb_config is not None:
            self.wandb_config.init_if_needed(self.get_model_config_for_wandb())
        
        for epoch in range(1, epochs + 1):
            if self.use_wgan_gp:
                metrics = self._train_epoch_wgan_gp(loader)
                self._print_metrics_if_needed_wgan_gp(metrics, epoch, epochs)
            else:
                metrics = self._train_epoch(loader)
                self._print_metrics_if_needed(metrics, epoch, epochs)
            
            if self.wandb_config is not None:
                self.wandb_config.log(metrics.to_dict())
        
        if self.wandb_config is not None:
            self.wandb_config.finish_and_save_if_needed(self.get_state_dict())
    
    def _initialize_model(self) -> None:
        self.generator = self.GeneratorModule(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims_generator
        ).to(self.device)
        
        self.discriminator = self.DiscriminatorModule(
            hidden_dims=self.hidden_dims_discriminator,
            use_sigmoid=not self.use_wgan_gp
        ).to(self.device)
        
        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.learning_rate_generator, 
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay
        )
        
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.learning_rate_discriminator, 
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay
        )
        
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
        generator_loss: float
        discriminator_loss: float
        discriminator_real_accuracy: float
        discriminator_fake_accuracy: float
        fid_score: float | None
        
        def to_dict(self) -> dict[str, float]:
            return {
                "generator_loss": self.generator_loss,
                "discriminator_loss": self.discriminator_loss,
                "discriminator_real_accuracy": self.discriminator_real_accuracy,
                "discriminator_fake_accuracy": self.discriminator_fake_accuracy,
                "fid_score": self.fid_score
            }

    @dataclass
    class WGANGPEpochMetrics:
        generator_loss: float
        critic_loss: float
        wasserstein_distance: float
        gradient_penalty: float
        fid_score: float | None
        
        def to_dict(self) -> dict[str, float]:
            return {
                "generator_loss": self.generator_loss,
                "critic_loss": self.critic_loss,
                "wasserstein_distance": self.wasserstein_distance,
                "gradient_penalty": self.gradient_penalty,
                "fid_score": self.fid_score
            }

    def _train_epoch(self, loader: DataLoader) -> EpochMetrics:
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_disc_real_correct = 0
        total_disc_fake_correct = 0
        total_samples = 0
        
        for batch in loader:
            real_images = batch[0]
            batch_size = real_images.size(0)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Train Discriminator (multiple times if critic_iterations > 1)
            for _ in range(self.critic_iterations):
                self.optimizer_discriminator.zero_grad()
                
                # Real images
                real_outputs = self.discriminator(real_images)
                real_loss = F.binary_cross_entropy(real_outputs, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                fake_outputs = self.discriminator(fake_images.detach())
                fake_loss = F.binary_cross_entropy(fake_outputs, fake_labels)
                
                # Total discriminator loss
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                self.optimizer_discriminator.step()
            
            # Train Generator
            self.optimizer_generator.zero_grad()
            
            # Generate fake images and try to fool discriminator
            fake_outputs = self.discriminator(fake_images)
            gen_loss = F.binary_cross_entropy(fake_outputs, real_labels)  # Want discriminator to think fake is real
            
            gen_loss.backward()
            self.optimizer_generator.step()
            
            # Calculate accuracies
            real_predictions = (real_outputs > 0.5).float()
            fake_predictions = (fake_outputs > 0.5).float()
            
            disc_real_correct = (real_predictions == real_labels).sum().item()
            disc_fake_correct = (fake_predictions == fake_labels).sum().item()
            
            # Accumulate metrics
            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            total_disc_real_correct += disc_real_correct
            total_disc_fake_correct += disc_fake_correct
            total_samples += batch_size
        
        return GenerativeAdversarialNetwork.EpochMetrics(
            generator_loss=total_gen_loss / len(loader),
            discriminator_loss=total_disc_loss / len(loader),
            discriminator_real_accuracy=total_disc_real_correct / total_samples,
            discriminator_fake_accuracy=total_disc_fake_correct / total_samples,
            fid_score=self._calculate_fid_score(loader)
        )

    def _train_epoch_wgan_gp(self, loader: DataLoader) -> WGANGPEpochMetrics:
        total_gen_loss = 0.0
        total_critic_loss = 0.0
        total_wasserstein_distance = 0.0
        total_gradient_penalty = 0.0
        
        for batch in loader:
            real_images = batch[0]
            batch_size = real_images.size(0)
            
            # Train Critic (multiple times)
            for _ in range(self.critic_iterations):
                self.optimizer_discriminator.zero_grad()
                
                # Real images
                real_outputs = self.discriminator(real_images)
                
                # Fake images
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                fake_outputs = self.discriminator(fake_images.detach())
                
                # Wasserstein loss
                wasserstein_distance = torch.mean(real_outputs) - torch.mean(fake_outputs)
                
                # Gradient penalty
                gradient_penalty = self._compute_gradient_penalty(real_images, fake_images)
                
                # Total critic loss
                critic_loss = -wasserstein_distance + self.gradient_penalty_weight * gradient_penalty
                critic_loss.backward()
                self.optimizer_discriminator.step()
            
            # Train Generator
            self.optimizer_generator.zero_grad()
            
            # Generate fake images and compute generator loss
            noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            fake_outputs = self.discriminator(fake_images)
            gen_loss = -torch.mean(fake_outputs)
            
            gen_loss.backward()
            self.optimizer_generator.step()
            
            # Accumulate metrics
            total_gen_loss += gen_loss.item()
            total_critic_loss += critic_loss.item()
            total_wasserstein_distance += wasserstein_distance.item()
            total_gradient_penalty += gradient_penalty.item()
        
        return GenerativeAdversarialNetwork.WGANGPEpochMetrics(
            generator_loss=total_gen_loss / len(loader),
            critic_loss=total_critic_loss / len(loader),
            wasserstein_distance=total_wasserstein_distance / len(loader),
            gradient_penalty=total_gradient_penalty / len(loader),
            fid_score=self._calculate_fid_score(loader)
        )

    def _compute_gradient_penalty(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        batch_size = real_images.size(0)
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        
        # Interpolated images
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Critic output for interpolated images
        interpolated_outputs = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=interpolated_outputs,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_outputs),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
        
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
        
        current_lr_gen = self.optimizer_generator.param_groups[0]['lr']
        current_lr_disc = self.optimizer_discriminator.param_groups[0]['lr']
        max_epochs_str_len = len(str(total_epochs))
        
        fid_score_str = f", FID: {metrics.fid_score:.4f}" if metrics.fid_score is not None else ""
        
        print(f"Epoch {epoch:{max_epochs_str_len}d}/{total_epochs}: "
              f"Gen Loss: {metrics.generator_loss:.4f}, "
              f"Disc Loss: {metrics.discriminator_loss:.4f}, "
              f"Disc Acc (Real/Fake): {metrics.discriminator_real_accuracy:.3f}/{metrics.discriminator_fake_accuracy:.3f}, "
              f"LR (Gen/Disc): {current_lr_gen:.6f}/{current_lr_disc:.6f}{fid_score_str}")

    def _print_metrics_if_needed_wgan_gp(self, metrics: WGANGPEpochMetrics, epoch: int, total_epochs: int) -> None:
        if self.print_every is None or epoch % self.print_every != 0:
            return
        
        current_lr_gen = self.optimizer_generator.param_groups[0]['lr']
        current_lr_critic = self.optimizer_discriminator.param_groups[0]['lr']
        max_epochs_str_len = len(str(total_epochs))
        
        fid_score_str = f", FID: {metrics.fid_score:.4f}" if metrics.fid_score is not None else ""
        
        print(f"Epoch {epoch:{max_epochs_str_len}d}/{total_epochs}: "
              f"Gen Loss: {metrics.generator_loss:.4f}, "
              f"Critic Loss: {metrics.critic_loss:.4f}, "
              f"W-Distance: {metrics.wasserstein_distance:.4f}, "
              f"GP: {metrics.gradient_penalty:.4f}, "
              f"LR (Gen/Critic): {current_lr_gen:.6f}/{current_lr_critic:.6f}{fid_score_str}")
    
    def generate(self, n_samples: int) -> np.ndarray:
        return self.generate_from_latent(np.random.randn(n_samples, self.latent_dim))
    
    def generate_from_latent(self, latent_vectors: np.ndarray) -> np.ndarray:
        if self.generator is None:
            raise ValueError("Model is not initialized - call `train()` first")
        
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.from_numpy(latent_vectors).float().to(self.device)
            generated_images = self.generator(z).cpu().numpy()
            
            # Convert from PyTorch (N, C, H, W) to expected (N, H, W, C) format
            generated_images = np.transpose(generated_images, (0, 2, 3, 1))
            
        return generated_images

    def get_model_config_for_wandb(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dims_generator": self.hidden_dims_generator,
            "hidden_dims_discriminator": self.hidden_dims_discriminator,
            "learning_rate_generator": self.learning_rate_generator,
            "learning_rate_discriminator": self.learning_rate_discriminator,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "weight_decay": self.weight_decay,
            "use_wgan_gp": self.use_wgan_gp,
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "critic_iterations": self.critic_iterations
        }
    
    def get_state_dict(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dims_generator": self.hidden_dims_generator,
            "hidden_dims_discriminator": self.hidden_dims_discriminator,
            "learning_rate_generator": self.learning_rate_generator,
            "learning_rate_discriminator": self.learning_rate_discriminator,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "weight_decay": self.weight_decay,
            "use_wgan_gp": self.use_wgan_gp,
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "critic_iterations": self.critic_iterations,
            
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_generator": self.optimizer_generator.state_dict(),
            "optimizer_discriminator": self.optimizer_discriminator.state_dict(),
            
            "wandb_config": self.wandb_config
        }
    
    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> Self:
        loaded_model = GenerativeAdversarialNetwork(
            latent_dim=state_dict["latent_dim"],
            hidden_dims_generator=state_dict["hidden_dims_generator"],
            hidden_dims_discriminator=state_dict["hidden_dims_discriminator"],
            learning_rate_generator=state_dict["learning_rate_generator"],
            learning_rate_discriminator=state_dict["learning_rate_discriminator"],
            beta_1=state_dict["beta_1"],
            beta_2=state_dict["beta_2"],
            weight_decay=state_dict["weight_decay"],
            use_wgan_gp=state_dict.get("use_wgan_gp", False),
            gradient_penalty_weight=state_dict.get("gradient_penalty_weight", 10.0),
            critic_iterations=state_dict.get("critic_iterations", 1)
        ).with_wandb(state_dict["wandb_config"])
        
        loaded_model._initialize_model()
        loaded_model.generator.load_state_dict(state_dict["generator"])
        loaded_model.discriminator.load_state_dict(state_dict["discriminator"])
        loaded_model.optimizer_generator.load_state_dict(state_dict["optimizer_generator"])
        loaded_model.optimizer_discriminator.load_state_dict(state_dict["optimizer_discriminator"])
        
        return loaded_model

    class GeneratorModule(nn.Module):
        def __init__(
            self, 
            latent_dim: int = 128,
            hidden_dims: list[int] = [512, 256, 128, 64, 32]
        ) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims
            
            # Calculate the size needed for the first deconv layer
            # We'll start with 4x4 feature maps
            self.initial_size = 4
            self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.initial_size * self.initial_size)
            
            self.main = self._build_generator(hidden_dims)
            
            # Calculate final output size after all deconv layers
            final_size = self.initial_size * (2 ** len(hidden_dims))
            
            # Add adaptive layer if needed to get to 64x64
            if final_size != 64:
                self.final_resize = nn.AdaptiveAvgPool2d((64, 64))
            else:
                self.final_resize = nn.Identity()
            
        def _build_generator(self, hidden_dims: list[int]) -> nn.Module:
            layers = []
            
            for i in range(len(hidden_dims) - 1):
                layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_dims[i],
                            hidden_dims[i + 1],
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.ReLU(inplace=True)
                    )
                )
            
            # Final layer to RGB
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[-1],
                        3,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.Sigmoid()
                )
            )
            
            return nn.Sequential(*layers)
        
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            # Project and reshape
            x = self.fc(z)
            x = x.view(-1, self.hidden_dims[0], self.initial_size, self.initial_size)
            
            # Generate image
            x = self.main(x)
            
            # Ensure output is 64x64
            x = self.final_resize(x)
            return x

    class DiscriminatorModule(nn.Module):
        def __init__(
            self, 
            hidden_dims: list[int] = [32, 64, 128, 256, 512],
            use_sigmoid: bool = True
        ) -> None:
            super().__init__()
            self.hidden_dims = hidden_dims
            self.use_sigmoid = use_sigmoid
            
            self.main = self._build_discriminator(hidden_dims)
            
            # Calculate output size after convolutions
            # Starting from 64x64, after len(hidden_dims) stride-2 convolutions: 64 / 2^len(hidden_dims)
            final_size = 64 // (2 ** len(hidden_dims))
            self.fc = nn.Linear(hidden_dims[-1] * final_size * final_size, 1)
            
        def _build_discriminator(self, hidden_dims: list[int]) -> nn.Module:
            layers = []
            
            # First layer (no batch norm)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(3, hidden_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_dims[i],
                            hidden_dims[i + 1],
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            
            return nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.main(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            if self.use_sigmoid:
                return torch.sigmoid(x)
            return x
