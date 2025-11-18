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
    
    # Get the shape of the light field
    _, _, U, V, _ = lf.shape
    
    # Compute central angular indices
    u0 = (U - 1) // 2  # Center horizontal angular index
    v0 = (V - 1) // 2  # Center vertical angular index

    # Extract the slice at central indices
    central_view = lf[:, :, u0, v0, :]
    
    return central_view