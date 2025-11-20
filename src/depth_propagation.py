import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_depth_map(depth, sigma=1.0):
    """
    Apply Gaussian smoothing to the depth map.
    """
    return gaussian_filter(depth, sigma=sigma)

def fuse_depth_maps(depth_h, depth_v, conf_h, conf_v, eps=1e-6):
    """
    Fuse horizontal and vertical raw depth maps using confidence weights.
    """
    depth_h = depth_h.astype(np.float32)
    depth_v = depth_v.astype(np.float32)
    conf_h = conf_h.astype(np.float32)
    conf_v = conf_v.astype(np.float32)

    w_sum = conf_h + conf_v + eps
    depth_fused = (conf_h * depth_h + conf_v * depth_v) / w_sum
    return depth_fused

def propagate_depth(depth_h, depth_v, conf_h, conf_v, smooth_sigma=1.0):
    """
    Fuse depths and apply smoothing to obtain the final depth map.
    """
    depth_fused = fuse_depth_maps(depth_h, depth_v, conf_h, conf_v)
    depth_final = smooth_depth_map(depth_fused, sigma=smooth_sigma)
    return depth_final