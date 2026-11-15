"""CPU-only lightweight integration with provided gnn_common / GNN v6 graph.

No plots, SVG, files, torch or per-pixel PCHIP fitting. Uses GNN's OWN Line
IDs/keypoint segmentation, never silently reindexes a neural network graph.
"""
from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
from scipy.ndimage import label, minimum, maximum, distance_transform_edt
from width_profile import (SegmentMeanWidth, compute_distance_map,
                           estimate_segment_width, means_from_line_label_map)

_FOUR = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)


def _region_partition_preserved(ink: np.ndarray, new: np.ndarray) -> bool:
    """Fast one-to-one 4-connected background check without N×2 pair sorting."""
    old_labels, old_num=label(~ink,structure=_FOUR)
    new_labels, new_num=label(~new,structure=_FOUR)
    if old_num!=new_num:return False
    if not old_num:return True
    indices=np.arange(1,old_num+1)
    mn=np.asarray(minimum(new_labels,labels=old_labels,index=indices))
    mx=np.asarray(maximum(new_labels,labels=old_labels,index=indices))
    return bool(np.all(mn==mx) and np.all(mn>0) and len(np.unique(mn))==old_num)


def skeletonize_lineart_rgba(lineart_rgba: np.ndarray, *, config=None,
                              fallback:Callable|None=None,
                              partition_guard:bool=True) -> np.ndarray:
    """Drop-in gnn_common.skeletonize_lineart replacement, CPU compute only.

    Input and output are RGBA uint8, ink pixels have alpha>0.
    Color is directly inherited for skeleton pixels inside the original ink;
    only newly extrapolated out-of-ink points need a nearest-neighbor lookup.
    """
    rgba=np.asarray(lineart_rgba)
    if rgba.dtype!=np.uint8 or rgba.ndim!=3 or rgba.shape[-1]!=4:
        raise ValueError('input must be preprocessed RGBA uint8')
    ink=rgba[...,3]>0
    if not np.any(ink):return np.zeros_like(rgba)
    # Lazy import: width-only callers do not pay vectorizer import overhead.
    from vectorizer import vectorize, Config
    cfg=config or Config(threshold=127,min_component=1,compute_widths=False)
    gray=np.where(ink,0,255).astype(np.uint8)
    sk=vectorize(gray,cfg,output_mode='skeleton').refined_skeleton
    if partition_guard and not _region_partition_preserved(ink,sk):
        if fallback is None:
            raise ValueError('new centerline would alter GNN Region partition; supply fallback')
        warnings.warn('Region partition changed: falling back to legacy skeleton',RuntimeWarning,stacklevel=2)
        return fallback(rgba)
    out=np.zeros_like(rgba)
    kept=sk & ink
    out[kept]=rgba[kept]
    outside=sk & ~ink
    if np.any(outside):
        # Rare boundary-extrapolated pixels only: nearest foreground color.
        _,nearest=distance_transform_edt(~ink,return_indices=True)
        yy,xx=np.nonzero(outside)
        out[yy,xx]=rgba[nearest[0,yy,xx],nearest[1,yy,xx]]
    out[sk,3]=255
    return out


def parameterize_gnn_line_graph(graph:dict, *,ink_mask:np.ndarray|None=None,
                                distance_map:np.ndarray|None=None,
                                return_profiles:bool=False,
                                trim_fraction:float=.12,
                                max_samples:int=96) -> dict:
    """Return a SINGLE average width for each GNN Line keypoint segment.

    Fast path: GNN's line_label_map and one cv2 EDT, using two bincounts.
    Fallback: batch one EDT, then at most `max_samples` points / segment.
    Always returns line_ids, line_mean_width_px, line_width_quality and
    line_sample_count; optional sparse SegmentMeanWidth records. No image IO.
    Width values are a *sidecar*, NOT changes to existing Line 5D features.
    """
    if ink_mask is None:
        rgba=graph.get('lineart_rgb')
        if rgba is None:raise ValueError('ink_mask or graph["lineart_rgb"] required')
        ink_mask=np.asarray(rgba)[...,3]>0
    ink=np.asarray(ink_mask,dtype=bool)
    info=graph['line_info']
    ids=np.asarray(sorted(int(i) for i in info),dtype=np.int64)
    n=len(ids)
    if distance_map is None:distance_map=compute_distance_map(ink)
    if distance_map.shape!=ink.shape:raise ValueError('distance_map shape mismatch')
    means=np.zeros(n,dtype=np.float32)
    counts=np.zeros(n,dtype=np.int32)
    line_label_map=graph.get('line_label_map')
    fast=(line_label_map is not None and np.shape(line_label_map)==ink.shape
          and np.array_equal(ids,np.arange(n)))
    if fast:
        means,counts=means_from_line_label_map(ink,line_label_map,n,distance_map=distance_map)
    # Fallback for unlabeled graph OR lines absent from raster label map.
    for i in (np.flatnonzero(counts==0) if fast else range(n)):
        lid=int(ids[i]); rec=info[lid]
        px,py=rec.get('bfs_order',((),()))
        xy=np.column_stack((px,py)) if len(px) else np.array([rec['kp1'],rec['kp2']])
        # No PCHIP, no normals, no ray casts; all samples reuse one EDT map.
        prof=estimate_segment_width(ink,xy,distance_map=distance_map,
                         seg_id=f'L{lid}',source_segment_id=lid,
                         trim_fraction=trim_fraction,max_samples=max_samples,
                         source_kind='gnn_line')
        means[i]=prof.mean_width_px
        counts[i]=prof.samples_used
    quality=np.minimum(1.,counts.astype(np.float32)/8.)
    result={'schema_version':'2.0','line_ids':ids,
            'line_mean_width_px':means,
            'line_width_quality':quality,
            'line_sample_count':counts}
    if return_profiles:
        result['profiles']={int(lid):SegmentMeanWidth(
            f'L{int(lid)}',int(lid),None,None,float(means[i]),float(quality[i]),
            int(counts[i]),float(info[int(lid)].get('path_len_raw',0.)),
            source_kind='gnn_line',method='label_edt_mean' if fast else 'trimmed_edt_mean')
            for i,lid in enumerate(ids)}
    return result


__all__=['skeletonize_lineart_rgba','parameterize_gnn_line_graph']
