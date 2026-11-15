"""Explicit benchmark only (never imported by GNN compute interfaces)."""
from __future__ import annotations
import argparse
import time
import numpy as np
import cv2
from gnn_adapter import parameterize_gnn_line_graph


def make_synthetic_graph(size:int, n_lines:int=32):
    mask=np.zeros((size,size),dtype=np.uint8)
    labels=np.full((size,size),-1,dtype=np.int32)
    info={}
    for lid,y in enumerate(np.linspace(24,size-25,n_lines,dtype=int)):
        thickness=int(2+(lid%7))
        if y-thickness//2<0 or y+thickness//2>=size:continue
        cv2.line(mask,(8,int(y)),(size-9,int(y)),255,thickness,cv2.LINE_8)
        # Synthetic middle skeleton labeling: existing GNN generates this.
        labels[int(y),16:size-16]=int(lid)
        p=np.arange(16,size-16)
        info[int(lid)]={'kp1':(16,int(y)),'kp2':(size-17,int(y)),
             'bfs_order':(p,np.full(len(p),int(y))), 'path_len_raw':size-33}
    return {'line_label_map':labels,'line_info':info},mask>0


def bench(size:int,n_lines:int,repeats:int):
    graph,ink=make_synthetic_graph(size,n_lines)
    times=[]
    for _ in range(repeats):
        t=time.perf_counter()
        result=parameterize_gnn_line_graph(graph,ink_mask=ink)
        times.append((time.perf_counter()-t)*1000)
    print(f'{size}²; {len(result["line_ids"])} segments; '
          f'mean={sum(times)/len(times):.1f}ms; '
          f'min={min(times):.1f}ms; '
          f'width_range=({result["line_mean_width_px"].min():.2f},'
          f'{result["line_mean_width_px"].max():.2f})')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sizes',type=int,nargs='+',default=[512,1024,2048])
    parser.add_argument('--lines',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    for size in args.sizes:
        bench(size,args.lines,args.repeats)
