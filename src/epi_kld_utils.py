import numpy as np

def rgb_to_gray(epi):
    """
    Converts RGB EPI to grayscale using standard NTSC coefficients.
    epi: shape (..., 3)
    Returns grayscale epi: shape (...)
    """
    if epi.ndim < 3 or epi.shape[-1] != 3:
        raise ValueError("Expected RGB EPI with last dimension = 3")
    return 0.299*epi[...,0] + 0.587*epi[...,1] + 0.114*epi[...,2]


def extract_window(epi, center, h, w):
    """
    Extracts an h x w window around center pixel (cx, cu) in an EPI.

    Parameters
    ----------
    epi : 2D numpy array (grayscale), shape (size_x, size_u)
    center : tuple (cx, cu) pixel coordinate
    h, w : int, window height and width

    Returns
    -------
    window : ndarray of shape (h, w)
    """
    cx, cu = center
    X, U = epi.shape

    half_h = h // 2
    half_w = w // 2

    x_min = max(0, cx - half_h)
    x_max = min(X, cx + half_h + 1)
    u_min = max(0, cu - half_w)
    u_max = min(U, cu + half_w + 1)

    # initialize window
    win = np.zeros((h, w), dtype=epi.dtype)

    # compute placement inside window
    win_x_min = half_h - (cx - x_min)
    win_u_min = half_w - (cu - u_min)

    win[win_x_min:win_x_min + (x_max - x_min),
        win_u_min:win_u_min + (u_max - u_min)] = epi[x_min:x_max, u_min:u_max]

    return win


def split_window_two_sides(window):
    """
    Splits window into left/right halves around the center column.

    This matches the paper's approach for (x-u) horizontal EPIs.

    Parameters
    ----------
    window : ndarray (h, w)

    Returns
    -------
    left_win : (h, w//2)
    right_win : (h, w//2)
    """
    _, w = window.shape
    mid = w // 2
    left = window[:, :mid]
    right = window[:, mid:]
    return left, right


def window_histograms(left_win, right_win, bins=32, intensity_range=(0,1e-6)):
    """
    Computes normalized histograms for left and right windows.

    For grayscale values assumed to be in [0,1].

    Parameters
    ----------
    left_win, right_win : ndarray
    bins : int, number of histogram bins

    Returns
    -------
    h1, h2 : probability distributions (sum to 1)
    """
    # If input has integer intensities 0-255, adjust range accordingly
    if intensity_range == (0,1e-6):
        # auto-detect
        if np.max(left_win) > 1.0 or np.max(right_win) > 1.0:
            # assume 0–255
            intensity_range = (0,255)
        else:
            intensity_range = (0.0,1.0)

    h1, _ = np.histogram(left_win.flatten(), bins=bins, range=intensity_range, density=False)
    h2, _ = np.histogram(right_win.flatten(), bins=bins, range=intensity_range, density=False)

    # Normalize to probability distributions
    h1 = h1.astype(np.float32)
    h2 = h2.astype(np.float32)

    # Avoid divide-by-zero
    if h1.sum() == 0:
        h1 += 1e-6
    if h2.sum() == 0:
        h2 += 1e-6

    h1 /= h1.sum()
    h2 /= h2.sum()

    return h1, h2
