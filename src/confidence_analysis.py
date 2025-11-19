import numpy as np

def compute_confidence_from_tensor(D, delta=2):
    """
    Compute confidence map from a KLD depth tensor.

    Parameters
    ----------
    D : ndarray, shape (H, W, A)
        KLD tensor for one direction (horizontal or vertical).
        H, W : spatial size
        A    : number of depth labels (alphas)
    delta : int
        Radius of the window around the peak index.

    Returns
    -------
    conf : ndarray, shape (H, W)
        Confidence map. Larger value means sharper KLD peak
        → more reliable depth at that pixel.
    """
    H, W, A = D.shape
    conf = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            curve = D[y, x, :]          # KLD curve over all alphas
            k = np.argmax(curve)        # index of the peak
            start = max(0, k - delta)
            end   = min(A - 1, k + delta)
            window = curve[start:end+1]

            if window.size > 1:
                conf[y, x] = np.var(window)
            else:
                conf[y, x] = 0.0

    return conf


def compute_confidences(Dh, Dv, delta=2):
    """
    Convenience function to get horizontal & vertical confidences
    in one call.

    Parameters
    ----------
    Dh : ndarray, shape (H, W, A)
        Horizontal KLD tensor.
    Dv : ndarray, shape (H, W, A)
        Vertical KLD tensor.
    delta : int
        Radius of the window around each peak.

    Returns
    -------
    conf_h, conf_v : ndarray, shape (H, W)
        Confidence maps for horizontal and vertical directions.
    """
    conf_h = compute_confidence_from_tensor(Dh, delta=delta)
    conf_v = compute_confidence_from_tensor(Dv, delta=delta)
    return conf_h, conf_v