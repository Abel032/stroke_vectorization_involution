"""Explicit CPU-only benchmark: no PNG/SVG or topology visualization export."""
from pathlib import Path
from time import perf_counter
from statistics import median
import argparse
import json
import cv2
import numpy as np
from vectorizer import Config, vectorize


def run(image_path: str, repeats: int = 3):
    img=cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)
    if img is None:raise ValueError(f'Unable to load image {image_path}')
    report={'shape':list(img.shape),'runs':{}}
    for mode in ('skeleton','segments','full'):
        times=[];last=None
        for _ in range(repeats):
            start=perf_counter()
            last=vectorize(img,Config(compute_widths=mode!='skeleton'),output_mode=mode)
            times.append(perf_counter()-start)
        row={'median_seconds':round(median(times),4), 'sample_seconds':[round(t,4) for t in times],
             'initial_segments':last.segment_count,'active_segments':last.meta['active_segments'],
             'junctions':last.junctions,'bridges':last.bridges,
             'collapsed_degree2':last.meta.get('collapsed_degree2_junctions',0),
             'demoted_pseudo':last.meta.get('demoted_pseudojunctions',0)}
        if mode=='segments':
            widths=[p.mean_width_px for p in last.segment_widths.values()]
            row.update({'segment_width_count':len(widths),
                        'median_width_px':round(float(np.median(widths)),3) if widths else None})
        if mode=='full':
            lengths=[s['length_px'] for s in last.topology['strokes']]
            row.update({'stroke_count':len(lengths),
                        'median_stroke_length_px':round(float(np.median(lengths)),2) if lengths else None,
                        'deg2_junctions':sum(j['degree']==2 for j in last.topology['junctions'])})
        report['runs'][mode]=row
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('image',nargs='?',default=str(Path(__file__).parent/'examples/B02_transparent.png'))
    ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args()
    print(json.dumps(run(args.image,args.repeats),ensure_ascii=False,indent=2))
