import numpy as np


def preprocess_images(images: np.ndarray) -> np.ndarray:
    return (images / 255.0).astype(np.float32)

def generated_to_image(generated: np.ndarray) -> np.ndarray:
    return (generated * 255.0).astype(np.uint8)
