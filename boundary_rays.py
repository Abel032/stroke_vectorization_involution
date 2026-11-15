"""Batched normal-ray utility ONLY for boundary topology reconstruction.
Never used by scalar segment-width estimation.
"""
import numpy as np
import cv2
def unit_rows(a):
    a=np.asarray(a,dtype=float)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def cast_exit_rays(binary: np.ndarray, origins: np.ndarray, directions: np.ndarray,
                   limits: np.ndarray | float, step: float = .5, batch: int = 768):
    """Batch signed rays. Returns (distance, valid) at first ink->background exit.

    Uses OpenCV remap across ALL pixels/steps of each batch, never Python per ray.
    Mask boundaries are sampled bilinearly with threshold 0.5 (subpixel approx).
    A ray reaching its safety limit without an exit is invalid, not an EDT width.
    """
    origins = np.asarray(origins, dtype=np.float32).reshape(-1, 2)
    directions = unit_rows(np.asarray(directions, np.float32).reshape(-1, 2)).astype(np.float32)
    lim = np.broadcast_to(np.asarray(limits, dtype=np.float32), (len(origins),))
    distance = np.full(len(origins), np.nan, dtype=np.float32)
    valid = np.zeros(len(origins), dtype=bool)
    if not len(origins) or not np.any(binary):
        return distance, valid
    img = np.asarray(binary, dtype=np.float32)
    stride = max(.2, float(step))
    for a in range(0, len(origins), batch):
        b = min(len(origins), a+batch)
        length = np.maximum(lim[a:b], stride)
        max_steps = int(np.ceil(np.max(length)/stride))
        if max_steps < 1:
            continue
        ds = stride*np.arange(1, max_steps+1, dtype=np.float32)
        loc = origins[a:b, None, :] + directions[a:b, None, :] * ds[None, :, None]
        ink = cv2.remap(img, loc[..., 0], loc[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= .5
        ended = (~ink) & (ds[None, :] <= length[:, None] + 1e-5)
        any_exit = ended.any(axis=1)
        first = np.argmax(ended, axis=1)
        indices = np.flatnonzero(any_exit)
        if len(indices):
            # The contour lies between last foreground and first background sample.
            distance[a+indices] = np.maximum(.25*stride, (first[indices]+.5)*stride)
            valid[a+indices] = True
    return distance, valid

