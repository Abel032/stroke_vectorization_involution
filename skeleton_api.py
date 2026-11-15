"""Stable public skeleton / keypoint-segment service for downstream GNN clients.

The 'segment' unit is one active graph edge between two detected keypoint
nodes; reconstructed bridges are internal and are not width segments.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from vectorizer import Config, Result, vectorize, preprocess, skeletonize, skeleton_graph


@dataclass
class SkeletonBundle:
    mask: np.ndarray
    initial_skeleton: np.ndarray
    refined_skeleton: np.ndarray
    keypoints: list[dict]
    segments: list[dict]
    widths: dict
    topology: dict
    result: Result


def extract_skeleton_bundle(image: np.ndarray, config: Config | None = None,
                            *, include_semantic_strokes: bool=False) -> SkeletonBundle:
    """Pure CPU geometry and scalar widths; semantic assembly is OPT-IN."""
    config=config or Config()
    result=vectorize(image,config,
        output_mode='full' if include_semantic_strokes else 'segments')
    graph=result.source_graph
    nodes=[{'id':int(nid), 'xy':tuple(map(float,xy)),
            'is_junction':bool(graph['junction'][nid])}
            for nid,xy in graph['nodes'].items()]
    paths={s.sid:s.points for s in graph['segments']}
    records=[]
    for key,prof in result.segment_widths.items():
        records.append({'id':key,'start_node':prof.start_node,'end_node':prof.end_node,
                        'path_xy': paths.get(prof.source_segment_id,np.empty((0,2))),
                        'length_px':prof.length_px, 'mean_width_px':prof.mean_width_px,
                        'width_profile':prof,
                        'source_active':True,
                        'closed':bool(prof.closed)})
    return SkeletonBundle(result.binary,result.raw_skeleton,result.refined_skeleton,
                          nodes,records,result.segment_widths,result.topology,result)


__all__=['Config','Result','SkeletonBundle','extract_skeleton_bundle',
         'preprocess','skeletonize','skeleton_graph','vectorize', 'extract_skeleton_mask']


def extract_skeleton_mask(image: np.ndarray, config: Config | None = None) -> np.ndarray:
    """Fast CPU-only skeleton backend: returns bool H×W and no visualization."""
    cfg=config or Config(compute_widths=False)
    return vectorize(image,cfg,output_mode='skeleton').refined_skeleton
