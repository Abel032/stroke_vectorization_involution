"""Fast regression checks for the paper's canonical topology patterns."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from vectorizer import Config, preprocess, skeleton_graph, skeletonize, vectorize, save_svg
from evaluate import evaluate

BASE=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('case,min_strokes',[
    ('t_junction',3),('x_junction',4),('y_junction',3),
    ('closed_loop',1),('varying_width',1),
])
def test_canonical_cases(case,min_strokes):
    image=cv2.imread(str(BASE/'examples'/f'{case}.png'))
    result=vectorize(image)
    assert len(result.strokes)>=min_strokes
    assert result.refined_skeleton.any()
    assert result.raw_skeleton.any()
    assert result.boundaries.any()
    assert result.binary.shape==image.shape[:2]
    gt=cv2.imread(str(BASE/'examples'/f'{case}_gt.png'),0)<128
    metrics=evaluate(result.refined_skeleton,gt)
    assert metrics['chamfer']<.018, (case,metrics)
    assert metrics['occupancy_percent']>80., (case,metrics)


def test_topology_preservation():
    for name,edges,junctions in [('t_junction',3,1),('x_junction',4,1),('closed_loop',1,0)]:
        img=cv2.imread(str(BASE/'examples'/f'{name}.png'))
        graph,meta=skeleton_graph(skeletonize(preprocess(img,Config())),Config())
        assert len(graph)==edges
        assert sum(meta['junction'].values())==junctions
        if name=='closed_loop':
            assert graph[0].closed


def test_export(tmp_path):
    im=cv2.imread(str(BASE/'examples'/'t_junction.png'))
    result=vectorize(im)
    svg=tmp_path/'result.svg'
    save_svg(result,svg,Config())
    text=svg.read_text()
    assert '<svg' in text and '<path' in text


def test_blank_image():
    result=vectorize(np.full((80,80),255,np.uint8))
    assert len(result.strokes)==0
    assert result.segment_count==0


def test_white_on_black():
    im=cv2.imread(str(BASE/'examples'/'t_junction.png'),0)
    a=vectorize(255-im,Config(invert=True))
    assert len(a.strokes)>0

@pytest.mark.parametrize('kind', ['spike','connector'])
def test_unreliable_segment_criteria(kind):
    from vectorizer import Segment, classify_segments
    # Deterministic test double for a fully boundary-inconsistent segment.
    class InconsistentBoundaries:
        cfg=Config()
        def initiated(self,p,t,binary,edt):
            return (0.,None,None)
        def boundary_initiated(self,p,t,radius):
            return 0.
    s=Segment(0,0,1,np.array([[6.,6.],[7.,7.],[8.,8.],[9.,9.]]))
    graph={'junction':{0:True,1:kind=='connector'},
           'nodes':{0:np.array([6.,6.]),1:np.array([9.,9.])}}
    report=classify_segments([s],graph,InconsistentBoundaries(),
                             np.ones((20,20),bool),np.ones((20,20)),Config())
    assert not s.active
    assert report['removed_spikes']==int(kind=='spike')
    assert report['removed_connectors']==int(kind=='connector')


def test_semantic_stroke_ids_and_x_continuity():
    image = cv2.imread(str(BASE/'examples'/'x_junction.png'))
    result = vectorize(image)
    topology = result.topology
    assert topology['stroke_count'] == 2
    assert len(result.semantic_strokes) == 2
    assert set(s['id'] for s in topology['strokes']) == {'S0','S1'}
    assert all(len(s['segment_ids']) == 2 for s in topology['strokes'])
    assert topology['endpoint_count'] == 4
    assert len(topology['junctions']) == 1
    j = topology['junctions'][0]
    assert j['id'] == 'J0' and j['kind'] == 'X_crossing'
    assert j['degree'] == 4
    assert len(j['continuations']) == 2
    assert set(p['stroke_id'] for p in j['continuations']) == {'S0','S1'}
    assert all(p['outgoing_cosine'] < -.9 for p in j['continuations'])


def test_semantic_t_junction_and_closed_loop():
    tj = vectorize(cv2.imread(str(BASE/'examples'/'t_junction.png'))).topology
    assert tj['stroke_count'] == 2
    assert tj['endpoint_count'] == 3
    assert tj['junctions'][0]['kind'] == 'T_junction'
    assert len(tj['junctions'][0]['continuations']) == 1
    stopped = tj['junctions'][0]['ending_strokes']
    assert len(stopped) == 1
    stroke = next(s for s in tj['strokes'] if s['id'] == stopped[0])
    assert 'J0' in (stroke['start_node'],stroke['end_node'])

    loop = vectorize(cv2.imread(str(BASE/'examples'/'closed_loop.png'))).topology
    assert loop['stroke_count'] == 1 and loop['endpoint_count'] == 0
    assert loop['strokes'][0]['closed']
    assert loop['strokes'][0]['start_node'] is None


def test_y_junction_geometry_not_mislabelled_as_t():
    topology = vectorize(cv2.imread(str(BASE/'examples'/'y_junction.png'))).topology
    assert topology['junctions'][0]['kind'] == 'Y_or_three_way_junction'
    assert topology['junctions'][0]['degree'] == 3
    # The Y reconstruction may pair one smoothly interpreted branch,
    # but a bent pair is *not* labelled as a geometric T junction.
    assert any('J0' in (s['start_node'], s['end_node']) for s in topology['strokes'])


def test_topology_files_and_annotations(tmp_path):
    import json
    from vectorizer import export_result
    result = vectorize(cv2.imread(str(BASE/'examples'/'x_junction.png')))
    prefix = tmp_path/'annotated'
    export_result(result, Config(), prefix)
    obj=json.loads((tmp_path/'annotated_topology.json').read_text(encoding='utf-8'))
    assert obj['stroke_count']==2
    assert obj['junctions'][0]['kind']=='X_crossing'
    assert obj['strokes'][0]['polyline_xy']
    svg=(tmp_path/'annotated_topology.svg').read_text(encoding='utf-8')
    assert 'stroke-S0' in svg and 'stroke-S1' in svg and 'J0' in svg
    img=cv2.imread(str(tmp_path/'annotated_topology.png'))
    assert img is not None and img.shape[0]==result.binary.shape[0]*2
    assert (tmp_path/'annotated.svg').exists()
