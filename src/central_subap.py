def extract_central_sub_aperture(lf):
    """
    Extracts the central sub-aperture image from the 4D light field volume.
    
    The central sub-aperture image is the view corresponding to the 
    center angular indices (u0, v0) in the U x V view grid.
    
    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input 4D light field volume.
    
    Returns
    -------
    central_view : ndarray, shape (H, W, C)
        The central sub-aperture image.
    """
    
    # 1. Get the shape of the light field
    H, W, U, V, C = lf.shape
    
    # 2. Compute the central angular indices
    # This assumes U and V are odd (e.g., 9x9), which is typical for HCI datasets.
    # The integer division '//' correctly finds the center index (e.g., (9-1)//2 = 4).
    u0 = (U - 1) // 2  # center horizontal angular index
    v0 = (V - 1) // 2  # center vertical angular index
    
    # 3. Extract the slice at the central indices
    # We use slicing [:, :, u0, v0, :] to select all spatial pixels and channels 
    # for the specific central (u0, v0) view, dropping the U and V dimensions.
    central_view = lf[:, :, u0, v0, :]
    
    return central_view