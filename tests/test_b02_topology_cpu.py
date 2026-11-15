"""Dense-junction CPU regression. Does not export visualization or alter GNN features."""
from pathlib import Path
import cv2
import numpy as np
from vectorizer import (Config, Segment, preprocess, skeleton_graph,
                        collapse_degree2_junctions, demote_pseudojunctions,
                        trim_to_reliable, vectorize)

ROOT = Path(__file__).resolve().parents[1]


def test_transparent_bgra_uses_alpha_and_not_rgb_black():
    bgra = np.zeros((30, 40, 4), dtype=np.uint8)
    bgra[10:13, 5:25] = [220, 50, 40, 255]
    out = preprocess(bgra, Config(min_component=1, alpha_threshold=127))
    assert out.sum() == 60 and out[11, 10]
    assert not out[0, 0]
    bgra[11, 10, 3] = 126
    assert not preprocess(bgra, Config(min_component=1))[11, 10]


def test_neighbor_junctions_must_be_graph_connected_before_merging():
    # Two separate "T"s have centers 3px apart, but none of their
    # foreground pixels is 8-connected to the other T.
    ink = np.zeros((30, 30), bool)
    pts = [(10, 10), (9, 10), (8, 10), (10, 9), (10, 8), (10, 11), (10, 12),
           (13, 10), (14, 10), (15, 10), (13, 9), (13, 8), (13, 11), (13, 12)]
    for x, y in pts:
        ink[y, x] = True
    _, safe = skeleton_graph(ink, Config(junction_merge_mode='geodesic'))
    _, unsafe = skeleton_graph(ink, Config(junction_merge_mode='euclidean'))
    assert sum(safe['junction'].values()) == 2
    assert sum(unsafe['junction'].values()) == 1  # archived failure reproducer


def test_attached_microcycle_is_not_counted_as_a_stroke():
    ink = np.zeros((30, 30), bool)
    for x, y in [(10, 10), (11, 10), (11, 11), (10, 11), (9, 10), (8, 10)]:
        ink[y, x] = True
    safe, _ = skeleton_graph(ink, Config())
    assert len(safe) == 1 and not safe[0].closed
    disabled, _ = skeleton_graph(ink, Config(junction_microcycle_max_length=0.))
    assert len(disabled) >= 2 and any(s.closed for s in disabled)


def test_real_isolated_closed_loop_is_preserved():
    ink = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(ink, (20, 20), 9, 255, 1)
    segs, _ = skeleton_graph(ink > 0, Config())
    assert any(s.closed and s.start < 0 and len(s.points)>10 for s in segs)


def _sampled(points, start, end, sid):
    pts = np.asarray(points, float)
    s = Segment(sid, start, end, pts, raw=pts.copy(), dense=pts.copy())
    s.consistency = np.ones(len(pts))
    s.left = [None] * len(pts)
    s.right = [None] * len(pts)
    return s


def test_reliable_degree_two_is_spliced_not_two_extrapolations():
    a = _sampled([(3, 12), (4, 12), (5, 12), (6, 12)], 0, 1, 0)
    b = _sampled([(6, 12), (7, 12), (8, 12), (9, 12)], 1, 2, 1)
    graph = {'nodes': {0: np.array([3., 12.]), 1: np.array([6., 12.]),
                       2: np.array([9., 12.])},
             'junction': {0: False, 1: True, 2: False}}
    n = collapse_degree2_junctions([a, b], graph, Config())
    assert n == 1
    remaining = [s for s in (a, b) if s.active]
    assert len(remaining) == 1
    assert (remaining[0].start, remaining[0].end) == (0, 2)
    assert 1 not in graph['nodes']
    assert len(remaining[0].dense) == 7


def test_ambiguous_acute_degree2_is_demoted_not_forced_to_connect():
    a = _sampled([(5, 5), (4, 5), (3, 5)], 0, 1, 0)
    b = _sampled([(5, 5), (6, 5), (7, 5)], 1, 2, 1)
    # Both outward from node are opposite, so would actually merge.
    # Use two branches traveling in same direction from the same node.
    a = _sampled([(3, 5), (4, 5), (5, 5)], 0, 1, 0)
    b = _sampled([(5, 5), (4, 6), (3, 7)], 1, 2, 1)
    g = {'nodes': {0: np.array([3., 5.]), 1: np.array([5., 5.]),
                   2: np.array([3., 7.])},
         'junction': {0: False, 1: True, 2: False}}
    cfg = Config()
    assert collapse_degree2_junctions([a, b], g, cfg) == 0
    assert demote_pseudojunctions([a, b], g, cfg) == 1
    assert sum(s.active for s in (a, b)) == 2
    assert not g['junction'][1]


def test_trim_cap_protects_short_segments():
    pts = np.c_[np.linspace(0., 8., 17), np.full(17, 10.)]
    seg = _sampled(pts, 0, 1, 0)
    seg.consistency = np.zeros(len(pts))
    edt = np.full((40, 40), 4.0)
    cfg = Config(max_trim=22., trim_fraction_limit=.20)
    cut = trim_to_reliable(seg, 0, edt, cfg)
    assert cut <= 4  # <= ~1.6 px in an 8px segment, allowing sample step


def test_b02_dense_topology_regression():
    file = ROOT / 'examples' / 'B02_transparent.png'
    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    assert img.shape == (730, 2048, 4)
    result = vectorize(img, Config(compute_widths=False))
    assert result.refined_skeleton.any()
    assert result.meta['collapsed_degree2_junctions'] >= 30
    assert result.meta['demoted_pseudojunctions'] > 0
    assert result.topology['stroke_count'] <= 300
    assert all(j['degree'] >= 3 for j in result.topology['junctions'])
    assert result.topology['stroke_count'] == len(result.semantic_strokes)
    # At least some non-trivial closed strokes must remain.
    assert sum(s['closed'] for s in result.topology['strokes']) >= 2


def test_refined_mask_rasterization_is_hard_binary():
    from vectorizer import rasterize
    p = np.asarray([(3.2, 3.4), (26.4, 26.1)], np.float64)
    im = rasterize([p], (32, 32))
    # Anti-alias values 1..254 must never turn into extra GNN graph pixels.
    assert set(np.unique(im).tolist()) <= {0, 255}
    assert np.count_nonzero(im) <= 26
