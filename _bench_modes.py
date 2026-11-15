import time

import cv2
import numpy as np

from vectorizer import Config, vectorize

p = r'E:\ai\aiColorize\input_Gcami_03_01_A\train\NotPainted\0325.png'
raw = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
if raw.ndim == 2:
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
elif raw.shape[2] == 3:
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
crop = raw[100:700, 100:700]
ink = crop[..., 3] > 0
gray = np.where(ink, 0, 255).astype(np.uint8)
for mode in ('skeleton', 'segments', 'full'):
    t0 = time.perf_counter()
    r = vectorize(gray, Config(threshold=127, min_component=1, compute_widths=False),
                  output_mode=mode)
    dt = time.perf_counter() - t0
    nseg = len(r.source_graph.get('segments', []))
    print(f'{mode:9s} {dt:6.2f}s  strokes={len(r.strokes)} sem={len(r.semantic_strokes)} '
          f'segs={nseg} topo_keys={len(r.topology)}')
