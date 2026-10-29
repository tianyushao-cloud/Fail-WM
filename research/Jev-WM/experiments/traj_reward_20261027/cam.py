import numpy as np
POS = np.array([.9, -1.4, 1.9]); _x = np.array([.83205, .5547, 0]); _y = np.array([-.2691, .40365, .8745])
_x /= np.linalg.norm(_x); _y /= np.linalg.norm(_y); R = np.stack([_x, _y, np.cross(_x, _y)])
F = 240 / np.tan(np.deg2rad(24))
def project(p):
    """world xyz (...,3) -> packed 320x256 pixel (...,2)"""
    c = (np.asarray(p) - POS) @ R.T
    return np.stack([(320 + F * c[..., 0] / -c[..., 2]) * .5, (240 - F * c[..., 1] / -c[..., 2]) * 256 / 480], -1)
