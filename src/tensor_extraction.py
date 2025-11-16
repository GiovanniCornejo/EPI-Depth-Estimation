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