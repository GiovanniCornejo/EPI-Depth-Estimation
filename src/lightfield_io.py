import numpy as np
import imageio.v2 as imageio
import os
from glob import glob

def load_hci_lightfield(scene_path, U=9, V=9):
    """
    Loads a 9x9 HCI 4D light field dataset into a numpy array.
    
    Parameters
    ----------
    scene_path : str
        Path to the folder containing the light field images.
    U, V : int
        Number of horizontal (U) and vertical (V) views in the light field grid.
    
    Returns
    -------
    lf : numpy.ndarray, shape (H, W, U, V, C)
        4D light field volume where H,W = spatial dimensions, 
        U,V = angular dimensions, C = number of channels (typically 3 for RGB)
    """

    # Get images from the scene folder
    files = sorted(glob(os.path.join(scene_path, "input_Cam*.png")))    
    if len(files) != U * V:
        raise ValueError(f"Expected {U*V} images, found {len(files)} instead.")
    
    lf = None  # 4D light field array

    # Loop through images in the angular grid
    for idx, filename in enumerate(files):
        img = imageio.imread(filename).astype(np.float32) / 255.0
        
        # Initialize LF array
        if lf is None:
            H, W, C = img.shape
            lf = np.zeros((H, W, U, V, C), dtype=np.float32)

        # Compute angular indices based on image index
        u = idx % U  # horizontal
        v = idx // U # vertical
        
        # Store image in correct position of LF array
        lf[:, :, u, v, :] = img

    assert lf is not None
    return lf
