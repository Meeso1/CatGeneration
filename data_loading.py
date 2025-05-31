import os
import numpy as np
from PIL import Image


def load_dataset(path: str, max_images: int = None) -> np.ndarray:
    """
    Load all images from directory given by path as numpy array.
    Returned array has shape (n_images, 64, 64, 3).
    """
    pictures_path = f"{path}/cats/Data"
    
    images = []
    for file in os.listdir(pictures_path)[:max_images]:
        if file.endswith(".png"):
            images.append(np.array(Image.open(os.path.join(pictures_path, file))))
                        
    return np.stack(images)
