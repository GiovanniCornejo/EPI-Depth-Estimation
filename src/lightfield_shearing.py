import numpy as np
from scipy.ndimage import map_coordinates

def refocus_shear_single(lf, alpha, clip_coords=True):
    """
    Refocus a 4D light field using angular shear.
    Produces a full refocused volume (H, W, U, V, C) using bilinear interpolation.

    Parameters
    ----------
    lf : ndarray (H, W, U, V, C)
    alpha : float
        Refocus parameter.
    clip_coords : bool
        If True, clip coordinates instead of skipping them.

    Returns
    -------
    out : ndarray (H, W, U, V, C)
        Sheared light field volume.
    """
    H, W, U, V, C = lf.shape
    u_coords = np.arange(U) - (U - 1) / 2
    v_coords = np.arange(V) - (V - 1) / 2

    # create meshgrid for output image
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    out = np.zeros_like(lf)

    for u_idx, du in enumerate(u_coords):
        for v_idx, dv in enumerate(v_coords):
            # compute shifted coordinates
            Xs = X + du * (1 - 1/alpha)
            Ys = Y + dv * (1 - 1/alpha)

            if clip_coords:
                Xs = np.clip(Xs, 0, W - 1)
                Ys = np.clip(Ys, 0, H - 1)

            for c in range(C):
                # map_coordinates expects coordinates as (rows, cols)
                out[:, :, u_idx, v_idx, c] = map_coordinates(
                    lf[:, :, u_idx, v_idx, c],
                    [Ys, Xs],
                    order=1,  # bilinear interpolation
                    mode='reflect'
                )
    return out


def refocus_shear(lf, alphas, return_refocused_volumes=True, clip_coords=True):
    """
    Refocus a light field over multiple alpha values.

    Parameters
    ----------
    lf : ndarray (H, W, U, V, C)
    alphas : list or array of alpha values
    return_refocused_volumes : bool
        If True: return the whole stack (A, H, W, U, V, C)
        If False: return only refocused central view (A, H, W, C)
    clip_coords : bool

    Returns
    -------
    stack : ndarray
        If return_refocused_volumes=True:
            shape (A, H, W, U, V, C)
        Else:
            shape (A, H, W, C)
    (u0, v0) : int tuple
        center angular coordinates used for shearing
    """
    H, W, U, V, C = lf.shape
    alphas = np.asarray(alphas)

    u0 = (U - 1) // 2
    v0 = (V - 1) // 2

    if return_refocused_volumes:
        stack = np.zeros((len(alphas), H, W, U, V, C), dtype=lf.dtype)
        for i, alpha in enumerate(alphas):
            stack[i] = refocus_shear_single(lf, alpha, clip_coords=clip_coords)
        return stack, (u0, v0)

    stack = np.zeros((len(alphas), H, W, C), dtype=lf.dtype)
    for i, alpha in enumerate(alphas):
        vol = refocus_shear_single(lf, alpha, clip_coords=clip_coords)
        stack[i] = vol[:, :, u0, v0, :]
    return stack, (u0, v0)
