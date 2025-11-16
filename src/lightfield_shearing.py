import numpy as np
from scipy.ndimage import map_coordinates

def refocus_shear_single(lf, alpha, clip_coords=True):
    """
    Refocus a 4D light field using angular shear.
    Produces a full sheared volume (H, W, U, V, C) using bilinear interpolation.

    This implements the refocusing formula:
        L_alpha(x, y, u, v) = L_F(x + u*(1 - 1/alpha), y + v*(1 - 1/alpha), u, v)

    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input 4D light field
    alpha : float
        Refocus parameter (shear amount). alpha=1 corresponds to no shear.
    clip_coords : bool
        If True, coordinates outside image bounds are clipped.
        Otherwise, values outside bounds are ignored (currently only clipping used).

    Returns
    -------
    out : ndarray, shape (H, W, U, V, C)
        Refocused light field
    """
    H, W, U, V, C = lf.shape

    # Angular coordinates centered at zero
    u_coords = np.arange(U) - (U - 1) / 2
    v_coords = np.arange(V) - (V - 1) / 2

    # Meshgrid of pixel coordinates
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    # Initialize output volume
    out = np.zeros_like(lf)

    # Loop over each angular view
    for u_idx, du in enumerate(u_coords):
        for v_idx, dv in enumerate(v_coords):
            # Compute shifted coordinates based on shear formula
            Xs = X + du * (1 - 1/alpha)
            Ys = Y + dv * (1 - 1/alpha)

            # Optionally clip coordinates to image bounds
            if clip_coords:
                Xs = np.clip(Xs, 0, W - 1)
                Ys = np.clip(Ys, 0, H - 1)

            # Interpolate each color channel using bilinear interpolation
            for c in range(C):
                out[:, :, u_idx, v_idx, c] = map_coordinates(
                    lf[:, :, u_idx, v_idx, c],  # input image for this view
                    [Ys, Xs],             # y,x coordinates to sample
                    order=1,                          # bilinear interpolation
                    mode='reflect'                    # handle boundaries by reflection
                )

    return out


def refocus_shear(lf, alphas, clip_coords=True):
    """
    Refocus a light field over multiple alpha values.
    Returns the full sheared volume for each alpha.

    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input 4D light field
    alphas : list or array
        List of shear values (relative depths) to refocus at
    clip_coords : bool
        Whether to clip coordinates at image boundaries

    Returns
    -------
    stack : ndarray, shape (A, H, W, U, V, C)
        Sheared LF for each alpha value (A = len(alphas))
    (u0, v0) : int tuple
        Center angular coordinates used for shearing
    """
    H, W, U, V, C = lf.shape
    u0 = (U - 1) // 2  # center horizontal angular index
    v0 = (V - 1) // 2  # center vertical angular index

    # Initialize output stack for all alphas
    stack = np.zeros((len(alphas), H, W, U, V, C), dtype=lf.dtype)

    # Compute sheared LF for each alpha
    for i, alpha in enumerate(alphas):
        stack[i] = refocus_shear_single(lf, alpha, clip_coords=clip_coords)
    
    return stack, (u0, v0)
