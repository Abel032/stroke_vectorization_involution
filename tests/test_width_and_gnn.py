"""CPU scalar Segment width + GNN compatibility tests; no visualization calls."""
from pathlib import Path
import importlib.util
import cv2
import numpy as np
import pytest
from width_profile import (compute_distance_map, estimate_segment_width,
                           means_from_line_label_map)
from vectorizer import Config, vectorize, _neighbors
from skeleton_api import extract_skeleton_bundle,extract_skeleton_mask
from gnn_adapter import skeletonize_lineart_rgba,parameterize_gnn_line_graph
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('thickness',[5,11,23])
def test_single_scalar_width(thickness):
    ink=np.zeros((140,200),np.uint8)
    cv2.line(ink,(20,70),(180,70),255,thickness,cv2.LINE_8)
    p=np.array([[x,70.] for x in range(33,166)],float)
    dt=compute_distance_map(ink>0)
    prof=estimate_segment_width(ink>0,p,distance_map=dt,seg_id='G1',start_node=0,end_node=1)
    assert abs(prof.mean_width_px-(thickness+1))<2
    assert prof.samples_used<=96 and prof.samples_used>=6
    assert prof.width_at(0.)==prof.width_at(.5,normalized=True)==prof.mean_width_px
    assert 'parameterization' not in prof.to_dict()
    assert set(prof.to_dict())>= {'start_keypoint','end_keypoint','mean_width_px'}


def test_two_segment_means_are_independent():
    ink=np.zeros((160,240),np.uint8)
    cv2.line(ink,(15,35),(225,35),255,5)
    cv2.line(ink,(15,115),(225,115),255,19)
    dt=compute_distance_map(ink>0)
    thin=estimate_segment_width(ink>0,np.c_[np.arange(35,205),np.full(170,35)],distance_map=dt)
    thick=estimate_segment_width(ink>0,np.c_[np.arange(35,205),np.full(170,115)],distance_map=dt)
    assert 4<thin.mean_width_px<9
    assert 18<thick.mean_width_px<23


def test_gnn_label_map_vectorized_aggregation():
    ink=np.zeros((96,128),bool)
    ink[20:30,10:116]=True
    ink[64:82,10:116]=True
    labels=np.full(ink.shape,-1,np.int32)
    labels[25,20:105]=0
    labels[73,20:105]=1
    graph={'line_info':{
        0:{'kp1':(20,25),'kp2':(104,25),'bfs_order':(np.arange(20,105),np.full(85,25))},
        1:{'kp1':(20,73),'kp2':(104,73),'bfs_order':(np.arange(20,105),np.full(85,73))}},
        'line_label_map':labels,
        'lineart_rgb':np.dstack((np.zeros((*ink.shape,3),np.uint8),ink.astype(np.uint8)*255))}
    result=parameterize_gnn_line_graph(graph)
    assert result['line_ids'].tolist()==[0,1]
    assert result['line_mean_width_px'].shape==(2,)
    assert result['line_mean_width_px'][1]>result['line_mean_width_px'][0]+5
    assert 'profiles' not in result
    also=parameterize_gnn_line_graph(graph,return_profiles=True)
    assert also['profiles'][0].width_at(.9)==also['profiles'][0].mean_width_px
    assert not hasattr(also['profiles'][0],'knot_s_px')


def test_gnn_path_fallback_and_reuse_dt(monkeypatch):
    ink=np.zeros((80,120),bool);ink[30:40,10:110]=True
    gr={'line_info':{7:{'kp1':(18,35),'kp2':(100,35),
                        'bfs_order':(np.arange(18,101),np.full(83,35))}}}
    dt=compute_distance_map(ink)
    import gnn_adapter
    def forbid(*args,**kwargs): raise AssertionError('should reuse cached EDT')
    monkeypatch.setattr(gnn_adapter,'compute_distance_map',forbid)
    d=parameterize_gnn_line_graph(gr,ink_mask=ink,distance_map=dt)
    assert d['line_ids'].tolist()==[7]
    assert d['line_mean_width_px'][0]>7


def test_full_segment_ids_no_pchip():
    img=cv2.imread(str(ROOT/'examples/x_junction.png'))
    result=vectorize(img,Config())
    assert set(result.segment_widths)=={'G0','G1','G2','G3'}
    assert len(result.part_widths)==0
    assert len(result.topology['geometric_segments'])==4
    assert all(p.mean_width_px>0 for p in result.segment_widths.values())
    assert all(p.samples_used<=96 for p in result.segment_widths.values())
    assert result.meta['widths']['schema_version']=='2.0'


def test_skeleton_mode_does_not_assemble_strokes_or_width(monkeypatch):
    from pathlib import Path
    import vectorizer
    def fail(*args,**kwargs):raise AssertionError('No SVG, annotation or width estimation in the skeleton API')
    monkeypatch.setattr(vectorizer,'assemble_strokes',fail)
    monkeypatch.setattr(vectorizer,'estimate_segment_width',fail)
    monkeypatch.setattr(vectorizer,'save_svg',fail)
    image=cv2.imread(str(ROOT/'examples/t_junction.png'))
    result=vectorizer.vectorize(image,Config(),output_mode='skeleton')
    assert result.refined_skeleton.any()
    assert not result.strokes and not result.semantic_strokes
    assert result.segment_widths=={}
    assert result.boundaries.size==0
    assert np.array_equal(extract_skeleton_mask(image),result.refined_skeleton)


def test_bundle_exposes_mean_per_keypoint_segment():
    img=cv2.imread(str(ROOT/'examples/t_junction.png'))
    b=extract_skeleton_bundle(img)
    assert len(b.segments)==len(b.widths)>0
    assert all(s['mean_width_px']>0 and s['path_xy'].ndim==2 for s in b.segments)
    assert all(s['width_profile'].mean_width_px==s['mean_width_px'] for s in b.segments)


def test_no_false_diagonal_cycles():
    pts={(10,10),(11,10),(11,11)}
    assert (11,11) not in _neighbors((10,10),pts,'topology_safe')
    assert (11,11) in _neighbors((10,10),pts,'redundant_only')
    pts={(10,10),(11,11),(12,12)}
    assert (11,11) in _neighbors((10,10),pts,'topology_safe')


def test_gnn_rgba_output_and_legacy_signature(monkeypatch):
    rgba=np.zeros((90,100,4),np.uint8)
    rgba[30:43,15:85]=[80,120,210,255]
    out=skeletonize_lineart_rgba(rgba,partition_guard=False)
    assert out.shape==rgba.shape and out.dtype==np.uint8
    assert (out[...,3]>0).any()
    assert (out[out[...,3]>0,:3]==[80,120,210]).all()
    p=ROOT/'gnn_integration/gnn_common.py'
    spec=importlib.util.spec_from_file_location('gnn_common_cpu_test',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('LINE_SKELETON_BACKEND','legacy')
    assert np.array_equal(module.skeletonize_lineart(rgba),module._legacy_skeletonize_lineart(rgba))
    d=module.extract_line_segment_mean_widths({'line_info':{0:{'kp1':(16,36),'kp2':(80,36),
          'bfs_order':(np.arange(16,81),np.full(65,36))}}},ink_mask=rgba[...,3]>0)
    assert d['line_mean_width_px'].shape==(1,)


def test_segment_bundle_skips_semantic_assembly(monkeypatch):
    import vectorizer
    def fail(*args,**kwargs):raise AssertionError('segment API must not assemble semantic render annotations')
    monkeypatch.setattr(vectorizer,'assemble_strokes',fail)
    bundle=extract_skeleton_bundle(cv2.imread(str(ROOT/'examples/y_junction.png')))
    assert bundle.widths and bundle.segments
    assert bundle.topology=={}
    assert bundle.result.strokes==[]
    assert bundle.result.boundaries.size==0
