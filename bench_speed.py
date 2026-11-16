# -*- coding: utf-8 -*-
"""Timing after optimization: vectorize crop/full + aiColorize 侧的 btp 分组耗时。"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, r'E:\ai\boundary_topology_python')
sys.path.insert(0, r'E:\ai\aiColorize')
from vectorizer import Config, vectorize

B02 = r'E:\ai\aiColorize\input_B\predict\B02.png'
img = cv2.imread(B02, cv2.IMREAD_UNCHANGED)
crop = img[214:1014, 900:1700]

for tag, im in (('crop800', crop), ('full', img)):
    for mode in ('skeleton', 'segments', 'full'):
        t0 = time.perf_counter()
        r = vectorize(im, Config(), output_mode=mode)
        print(f'{tag:8s} {mode:8s} {time.perf_counter() - t0:5.2f}s  '
              f'segments={r.segment_count} strokes={len(r.semantic_strokes)}')

import gnn_common as G
raw = cv2.imdecode(np.fromfile(r'E:\ai\aiColorize\input_B\train\NotPainted\B01.png',
                              dtype=np.uint8), cv2.IMREAD_UNCHANGED)
rgba = G.prepare_lineart_for_graph(G.process_alpha(raw))
t0 = time.perf_counter()
grp = G.extract_btp_stroke_group_map(rgba)
print(f'extract_btp_stroke_group_map(B01 full 1393x3903): {time.perf_counter() - t0:.2f}s  '
      f'groups={grp["n_groups"]} segments={grp["n_segments"]} mapped={grp["n_mapped_segments"]}')
