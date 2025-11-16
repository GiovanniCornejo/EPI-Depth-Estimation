import numpy as np
import imageio.v2 as imageio
import os
from glob import glob

def load_hci_lightfield(scene_path, U=9, V=9):
    """
    Loads the 9x9 HCI 4D light field dataset into a numpy array.
    
    Returns:
        lf: numpy array of shape (H, W, U, V, C)
    """
    lf = None

    # Load all input_Cam### images
    files = sorted(glob(os.path.join(scene_path, "input_Cam*.png")))
    
    if len(files) != U * V:
        raise ValueError(f"Expected {U*V} images, found {len(files)} instead.")

    # Loop through the angular grid
    for idx, filename in enumerate(files):
        img = imageio.imread(filename).astype(np.float32) / 255.0
        
        if lf is None:
            H, W, C = img.shape
            lf = np.zeros((H, W, U, V, C), dtype=np.float32)

        # Compute angular indices
        u = idx % U
        v = idx // U
        
        # Save image into proper location
        lf[:, :, u, v, :] = img

    return lf