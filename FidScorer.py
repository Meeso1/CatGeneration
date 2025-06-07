import torch
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.linalg import sqrtm
from torch.utils.data import TensorDataset, DataLoader


class FidScorer:
    """
    Frechet Inception Distance calculator using pre-trained InceptionV3.
    """ 
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception_model: nn.Module | None = None
        self.transform: transforms.Compose | None = None

    def _load_inception_model(self) -> nn.Module:
        """Load pre-trained InceptionV3 and extract features from the last pooling layer."""
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = nn.Identity()  # Remove final classification layer
        inception.eval()
        return inception.to(self.device)
    
    def _get_transform(self) -> transforms.Compose:
        """Get the preprocessing transforms for InceptionV3."""
        return transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, images: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """
        Extract features from images using InceptionV3.
        
        Args:
            images: Array of shape (N, H, W, C) with values in [0, 1]
            batch_size: Batch size for processing
            
        Returns:
            Features array of shape (N, 2048)
        """
        if len(images.shape) != 4 \
            or images.shape[1:] != (64, 64, 3) \
            or images.max() > 1.0 \
            or images.min() < 0.0:
            raise ValueError("Images must be in (N, H, W, C) format with dimensions nx64x64x3 and values in [0, 1]")
        
        # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
        images_transposed = np.transpose(images, (0, 3, 1, 2))
        
        dataset = TensorDataset(torch.from_numpy(images_transposed).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return self.extract_features_from_dataloader(dataloader)

    def extract_features_from_dataloader(self, dataloader: DataLoader) -> np.ndarray:
        """
        Extract features from images in a DataLoader using InceptionV3.
        
        Args:
            dataloader: DataLoader containing images in (N, C, H, W) format with values in [0, 1]
            
        Returns:
            Features array of shape (N, 2048)
        """
        if self.inception_model is None or self.transform is None:
            self.inception_model = self._load_inception_model()
            self.transform = self._get_transform()
        
        features_list = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0]
                
                # Validate input format
                if len(images.shape) != 4 or images.shape[1:] != (3, 64, 64):
                    raise ValueError(f"Images must be in (N, C, H, W) format with 3x64x64 shape, got shape: {images.shape}")
                
                images = (images * 2.0) - 1.0                
                images = self.transform(images).to(self.device)
                
                # Extract features
                batch_features = self.inception_model(images)
                features_list.append(batch_features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_statistics(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance matrix of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_images: np.ndarray, generated_images: np.ndarray) -> float:
        """
        Calculate FID between real and generated images.
        
        Args:
            real_images: Array of real images, shape (N, H, W, C) with values in [0, 1]
            generated_images: Array of generated images, shape (M, H, W, C) with values in [0, 1]
            
        Returns:
            FID score (lower is better)
        """
        # Extract features
        real_features = self.extract_features(real_images)
        real_mu, real_sigma = self.calculate_statistics(real_features)
        
        return self.calculate_fid(real_mu, real_sigma, generated_images)
    
    def calculate_fid(self, real_stats: tuple[np.ndarray, np.ndarray], generated_images: np.ndarray) -> float:
        real_mu, real_sigma = real_stats
        generated_features = self.extract_features(generated_images)
        generated_mu, generated_sigma = self.calculate_statistics(generated_features)
        
        return self._frechet_distance(real_mu, real_sigma, generated_mu, generated_sigma)
    
    def _frechet_distance(
        self, 
        mu1: np.ndarray, 
        sigma1: np.ndarray,
        mu2: np.ndarray, 
        sigma2: np.ndarray, 
        eps: float = 1e-6
    ) -> float:
        """Calculate the Frechet distance between two multivariate Gaussians."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Calculate sqrt((sigma1 @ sigma2)^0.5)
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding %s to diagonal of covariance estimates" % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Handle complex numbers from sqrtm
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
