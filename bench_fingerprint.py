# -*- coding: utf-8 -*-
"""Output fingerprint: proves an optimization keeps the vectorizer bit-identical."""
import hashlib
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, r'E:\ai\boundary_topology_python')
from vectorizer import Config, vectorize


def fingerprint(image_path, crop=None):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if crop:
        y0, y1, x0, x1 = crop
        img = img[y0:y1, x0:x1]
    res = vectorize(img, Config(), output_mode='full')
    t = res.topology
    h = hashlib.sha256()
    for p in res.strokes:
        h.update(np.asarray(p, np.float64).tobytes())
    seg_h = hashlib.sha256()
    for s in res.source_graph.get('segments', []):
        seg_h.update(np.asarray(s.points, np.float64).tobytes())
    widths = {str(k): round(float(v.mean_width_px), 6) for k, v in res.segment_widths.items()}
    return {
        'segments': res.segment_count,
        'active': int(res.meta['active_segments']),
        'closed': sum(1 for s in res.source_graph['segments'] if s.closed),
        'strokes': len(res.semantic_strokes),
        'stroke_closed': sum(1 for s in t['strokes'] if s['closed']),
        'junctions': res.junctions,
        'bridges': res.bridges,
        'endpoints': sum(1 for n in t['nodes'] if n['type'] == 'endpoint'),
        'removed_spikes': res.meta['removed_spikes'],
        'removed_connectors': res.meta['removed_connectors'],
        'widths': widths,
        'strokes_sha': h.hexdigest(),
        'segments_sha': seg_h.hexdigest(),
    }


if __name__ == '__main__':
    out = sys.argv[1]
    data = {}
    data['crop'] = fingerprint(r'E:\ai\aiColorize\input_B\predict\B02.png', (214, 1014, 900, 1700))
    data['full'] = fingerprint(r'E:\ai\aiColorize\input_B\predict\B02.png')
    json.dump(data, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k, v in data.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != 'widths'})
