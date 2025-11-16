def extract_horizontal_epi(lf, y, v):
    """
    Extracts a horizontal (x-u) EPI.

    Parameters
    ----------
    lf : numpy array of shape (H, W, U, V, C)
    y  : int, spatial row index
    v  : int, angular v index (vertical viewpoint)

    Returns
    -------
    epi : numpy array of shape (W, U, C)
    """
    # lf[x, y, u, v, c] => we fix y, v and vary x, u
    H, W, U, V, C = lf.shape
    if y < 0 or y >= H:
        raise ValueError("Invalid y index")
    if v < 0 or v >= V:
        raise ValueError("Invalid v index")
    
    # Extract: all x, all u, fixed y, fixed v
    # shape: (W, U, C)
    epi = lf[y, :, :, v, :]  # shape (W, U, C)
    return epi


def extract_vertical_epi(lf, x, u):
    """
    Extracts a vertical (y-v) EPI.

    Parameters
    ----------
    lf : numpy array (H, W, U, V, C)
    x  : int, spatial column index
    u  : int, angular u index (horizontal viewpoint)

    Returns
    -------
    epi : numpy array of shape (H, V, C)
    """
    H, W, U, V, C = lf.shape
    if x < 0 or x >= W:
        raise ValueError("Invalid x index")
    if u < 0 or u >= U:
        raise ValueError("Invalid u index")

    # Extract: all y, all v, fixed x, fixed u
    # shape: (H, V, C)
    epi = lf[:, x, u, :, :]  # shape (H, V, C)
    return epi


def extract_all_horizontal_epis(lf, v):
    """
    Extracts all horizontal EPIs for a given angular index v.

    Returns
    -------
    epis : list of arrays, length H
        Each element has shape (W, U, C)
    """
    H, _, _, _, _ = lf.shape
    return [extract_horizontal_epi(lf, y, v) for y in range(H)]


def extract_all_vertical_epis(lf, u):
    """
    Extracts all vertical EPIs for a given angular index u.

    Returns
    -------
    epis : list of arrays, length W
        Each element has shape (H, V, C)
    """
    _, W, _, _, _ = lf.shape
    return [extract_vertical_epi(lf, x, u) for x in range(W)]
