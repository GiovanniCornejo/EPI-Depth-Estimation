import numpy as np

# ---------------------------------------------------------------------------- #
#                                EPI Extraction                                #
# ---------------------------------------------------------------------------- #

def extract_horizontal_epi(lf, y, v):
    """
    Extract a single horizontal (x-u) EPI from a 4D light field.

    A horizontal EPI is a 2D slice where the horizontal spatial dimension (x)
    and horizontal angular dimension (u) vary, while the row y and vertical 
    angular index v are fixed.

    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input light field
    y : int
        Spatial row index to fix
    v : int
        Vertical angular index to fix

    Returns
    -------
    epi : ndarray, shape (W, U, C)
        Extracted horizontal EPI
    """
    H, _, _, V, _ = lf.shape
    if y < 0 or y >= H:
        raise ValueError(f"Invalid y index: {y}")
    if v < 0 or v >= V:
        raise ValueError(f"Invalid vertical angular index: {v}")
    
    # Slice all x (columns) and u (horizontal views)
    epi = lf[y, :, :, v, :]  # shape: (W, U, C)
    return epi


def extract_vertical_epi(lf, x, u):
    """
    Extract a single vertical (y-v) EPI from a 4D light field.

    A vertical EPI is a 2D slice where the vertical spatial dimension (y)
    and vertical angular dimension (v) vary, while the column x and horizontal
    angular index u are fixed.

    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input light field
    x : int
        Spatial column index to fix
    u : int
        Horizontal angular index to fix

    Returns
    -------
    epi : ndarray, shape (H, V, C)
        Extracted vertical EPI
    """
    _, W, U, _, _ = lf.shape
    if x < 0 or x >= W:
        raise ValueError(f"Invalid x index: {x}")
    if u < 0 or u >= U:
        raise ValueError(f"Invalid horizontal angular index: {u}")
    
    # Slice all y (rows) and v (vertical views)
    epi = lf[:, x, u, :, :]  # shape: (H, V, C)
    return epi


def extract_all_horizontal_epis(lf, v):
    """
    Extract all horizontal EPIs for a given vertical angular index.

    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input light field
    v : int
        Vertical angular index to fix

    Returns
    -------
    epis : list of ndarray
        Length H, each element is a horizontal EPI of shape (W, U, C)
    """
    H, _, _, _, _ = lf.shape
    return [extract_horizontal_epi(lf, y, v) for y in range(H)]


def extract_all_vertical_epis(lf, u):
    """
    Extract all vertical EPIs for a given horizontal angular index.

    Parameters
    ----------
    lf : ndarray, shape (H, W, U, V, C)
        Input light field
    u : int
        Horizontal angular index to fix

    Returns
    -------
    epis : list of ndarray
        Length W, each element is a vertical EPI of shape (H, V, C)
    """
    _, W, _, _, _ = lf.shape
    return [extract_vertical_epi(lf, x, u) for x in range(W)]

# ---------------------------------------------------------------------------- #
#                                KLD Computation                               #
# ---------------------------------------------------------------------------- #

def _two_side_histograms_1d(epi_2d, center_idx, window_radius=2, nbins=32):
    """
    Compute normalized histograms for the two sides of a vertical center line
    in a 2D EPI.

    Parameters
    ----------
    epi_2d : ndarray, shape (N, M)
        2D EPI slice (spatial × angular), already converted to grayscale.
        axis 0 is the spatial axis we split into left/right.
    center_idx : int
        Index along axis 0 where the vertical center line passes.
    window_radius : int
        Half-size of the local window along the spatial axis.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    h_left, h_right : ndarray, shape (nbins,)
        Normalized histograms for left and right sides. If one side is empty,
        it will be filled with a near-uniform distribution.
    """
    N, M = epi_2d.shape  # spatial, angular

    # Window along spatial axis (0)
    start = max(0, center_idx - window_radius)
    end   = min(N - 1, center_idx + window_radius)

    # Left side: indices [start, center_idx)
    # Right side: indices [center_idx, end]
    left_region  = epi_2d[start:center_idx, :]
    right_region = epi_2d[center_idx:end + 1, :]

    # Flatten to 1D
    left_vals  = left_region.reshape(-1)
    right_vals = right_region.reshape(-1)

    # If one side has no pixels (edge cases), use all pixels as fallback
    if left_vals.size == 0:
        left_vals = epi_2d.reshape(-1)
    if right_vals.size == 0:
        right_vals = epi_2d.reshape(-1)

    # Assume epi_2d is in [0, 1]; if not, we clip it.
    left_vals  = np.clip(left_vals,  0.0, 1.0)
    right_vals = np.clip(right_vals, 0.0, 1.0)

    # Compute histograms
    h_left, _  = np.histogram(left_vals,  bins=nbins, range=(0.0, 1.0))
    h_right, _ = np.histogram(right_vals, bins=nbins, range=(0.0, 1.0))

    # Normalize to get probabilities
    h_left  = h_left.astype(np.float32)
    h_right = h_right.astype(np.float32)

    if h_left.sum()  > 0: h_left  /= h_left.sum()
    if h_right.sum() > 0: h_right /= h_right.sum()

    # Avoid zeros to keep log well-defined
    eps = 1e-8
    h_left  = np.clip(h_left,  eps, 1.0)
    h_right = np.clip(h_right, eps, 1.0)

    # Re-normalize after clipping
    h_left  /= h_left.sum()
    h_right /= h_right.sum()

    return h_left, h_right


def _symmetrized_kld(p, q):
    """
    Symmetrized Kullback-Leibler Divergence:
        0.5 * ( KL(p||q) + KL(q||p) )
    """
    # Assume p, q are already normalized and > 0
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    return 0.5 * (kl_pq + kl_qp)


def compute_horizontal_kld_tensor(sheared_stack, v_center,
                                  window_radius=2, nbins=32):
    """
    Compute horizontal depth tensor Dh(y, x, alpha_idx) using KLD on (x-u) EPIs.

    Parameters
    ----------
    sheared_stack : ndarray, shape (A, H, W, U, V, C)
        Stack of refocused light fields for different alphas.
        This is the output of refocus_shear(lf, alphas)[0].
        A is the number of depth labels (alphas).
    v_center : int
        Vertical angular index v* for extracting horizontal EPIs.
        Usually the center view: (V - 1) // 2.
    window_radius : int
        Half-size of the local spatial window (along x) in the EPI.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    Dh : ndarray, shape (H, W, A)
        Horizontal KLD tensor for each pixel and each alpha.
    """
    A, H, W, U, V, C = sheared_stack.shape
    Dh = np.zeros((H, W, A), dtype=np.float32)

    for a in range(A):
        lf_a = sheared_stack[a]  # (H, W, U, V, C)
        # For each row y, extract (x-u) EPI at fixed v_center
        for y in range(H):
            # epi_xyu: shape (W, U, C)
            epi_xyu = lf_a[y, :, :, v_center, :]
            # Convert to grayscale: average over color channels
            epi_gray = epi_xyu.mean(axis=-1)  # shape (W, U)

            for x in range(W):
                h_left, h_right = _two_side_histograms_1d(
                    epi_gray, center_idx=x,
                    window_radius=window_radius,
                    nbins=nbins
                )
                Dh[y, x, a] = _symmetrized_kld(h_left, h_right)

    return Dh


def compute_vertical_kld_tensor(sheared_stack, u_center,
                                window_radius=2, nbins=32):
    """
    Compute vertical depth tensor Dv(y, x, alpha_idx) using KLD on (y-v) EPIs.

    Parameters
    ----------
    sheared_stack : ndarray, shape (A, H, W, U, V, C)
        Stack of refocused light fields for different alphas.
    u_center : int
        Horizontal angular index u* for extracting vertical EPIs.
        Usually the center view: (U - 1) // 2.
    window_radius : int
        Half-size of the local spatial window (along y) in the EPI.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    Dv : ndarray, shape (H, W, A)
        Vertical KLD tensor for each pixel and each alpha.
    """
    A, H, W, U, V, C = sheared_stack.shape
    Dv = np.zeros((H, W, A), dtype=np.float32)

    for a in range(A):
        lf_a = sheared_stack[a]  # (H, W, U, V, C)
        # For each column x, extract (y-v) EPI at fixed u_center
        for x in range(W):
            # epi_yvu: shape (H, V, C)
            epi_yvu = lf_a[:, x, u_center, :, :]
            epi_gray = epi_yvu.mean(axis=-1)  # shape (H, V)

            for y in range(H):
                h_left, h_right = _two_side_histograms_1d(
                    epi_gray, center_idx=y,
                    window_radius=window_radius,
                    nbins=nbins
                )
                Dv[y, x, a] = _symmetrized_kld(h_left, h_right)

    return Dv


def argmax_depth_from_tensor(tensor, alphas):
    """
    Convert a KLD depth tensor D(y, x, alpha_idx) into a raw depth map
    by taking the alpha with maximum KLD at each pixel.

    Parameters
    ----------
    tensor : ndarray, shape (H, W, A)
        KLD tensor (horizontal or vertical).
    alphas : ndarray, shape (A,)
        The list/array of alpha values corresponding to the stack.

    Returns
    -------
    depth_map : ndarray, shape (H, W)
        Raw depth map where each pixel stores the alpha of the maximum KLD.
    """
    # index of best alpha per pixel
    best_idx = np.argmax(tensor, axis=-1)  # shape (H, W)
    alphas = np.asarray(alphas)
    return alphas[best_idx]