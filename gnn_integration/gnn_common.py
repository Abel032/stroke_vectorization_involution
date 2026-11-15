"""
============================================================================
 gnn_common.py -- 共享工具库 (预处理 / 可视化 / 工具函数)
 供 gnn_colorizer_v5.py 和 gnn_colorizer_v6_edge_ll.py 共同引用
============================================================================
"""

import os
import math
import pickle
import random
from typing import List, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import thin

import torch


# ==================== 1. 工具函数 ====================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 1.5 颜色统计工具 ====================
def _mode_rgb_packed(colors: np.ndarray) -> Tuple[int, int]:
    """计算颜色数组的众数 (向量化, 比 Counter 更快)
    
    Args:
        colors: (N, 3) uint8 RGB 或 (N, 4) uint8 RGBA 颜色数组
    Returns:
        (packed_rgb, count): 众数颜色的 packed 整数 (R<<16|G<<8|B, 忽略 alpha) 和出现次数
    """
    colors = np.asarray(colors)
    if colors.size == 0:
        return 0, 0

    flat = colors.reshape(-1, colors.shape[-1])
    if flat.shape[1] >= 3:
        flat = flat[:, :3]
    flat = flat.astype(np.uint32, copy=False)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    unique, first_idx, counts = np.unique(packed, return_index=True, return_counts=True)
    best_count = int(counts.max())
    candidates = np.flatnonzero(counts == best_count)
    best_pos = int(candidates[np.argmin(first_idx[candidates])])
    return int(unique[best_pos]), best_count


# ==================== 2. Scatter 消息传递工具 ====================
def _expand_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    shape = [index.size(0)] + [1] * (src.ndim - 1)
    return index.view(shape).expand_as(src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out_shape = [dim_size] + list(src.shape[1:])
    out = src.new_zeros(out_shape)
    idx_exp = _expand_index(index, src)
    out.scatter_add_(0, idx_exp, src)
    count = src.new_zeros(dim_size)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count_shape = [dim_size] + [1] * (src.ndim - 1)
    return out / count.view(count_shape).clamp(min=1)

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out_shape = [dim_size] + list(src.shape[1:])
    out = src.new_zeros(out_shape)
    idx_exp = _expand_index(index, src)
    out.scatter_add_(0, idx_exp, src)
    return out

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """按 index 取最大值；空分组返回 0，非空分组保留真实的 -inf。"""
    out_shape = [dim_size] + list(src.shape[1:])
    if src.is_floating_point():
        fill_value = float('-inf')
    elif src.dtype == torch.bool:
        fill_value = False
    else:
        fill_value = torch.iinfo(src.dtype).min
    out = src.new_full(out_shape, fill_value)
    idx_exp = _expand_index(index, src)
    out.scatter_reduce_(0, idx_exp, src, reduce='amax')

    # 不能仅通过 out == -inf 判定空分组，因为输入本身也可能是 -inf。
    seen = torch.zeros(dim_size, dtype=torch.int64, device=src.device)
    seen.scatter_add_(0, index, torch.ones_like(index, dtype=torch.int64))
    seen_shape = [dim_size] + [1] * (src.ndim - 1)
    return torch.where(seen.view(seen_shape) > 0, out, torch.zeros_like(out))


# ==================== 拓扑构图专用共享函数 ====================

def normalize_lineart_rgba(lineart_rgba, white_threshold=250, min_alpha=1):
    """把 RGBA 线稿规范成前景 alpha>0 的表示；默认不截断半透明线像素。

    全不透明且大面积接近白色时识别为白底黑线。仅利用背景 *位置*，
    不会把彩色线稿中的蓝色/红色笔画误当背景。
    """
    rgba = np.asarray(lineart_rgba)
    if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError('线稿必须是 RGBA uint8')
    out = rgba.copy()
    a = rgba[..., 3]
    if np.all(a == 255):
        # 仅在绝大多数图像像素为接近纯白时识别白底，避免误判满版不透明彩稿。
        white = np.all(rgba[..., :3] >= int(white_threshold), axis=-1)
        if float(np.mean(white)) >= 0.5:
            out[white] = (0, 0, 0, 0)
    if min_alpha > 1:
        out[out[..., 3] < int(min_alpha)] = (0, 0, 0, 0)
    return out


def trace_skeleton_component(px, py):
    """沿 8 邻接像素图连续访问每个骨架像素；无任意 `set.pop()` 空间跳跃。

    贪心走到死胡同后，经已访问像素的最短路径抵达最近未访问节点，
    因而返回的相邻采样点始终 8 连通。分叉处允许重复经过回溯像素；
    应用 distinct_edges=True 计算弧长避免回溯重复计入几何长度。
    返回 (walk_x, walk_y, n_backtrack_steps)。
    """
    from collections import deque
    points = {(int(x), int(y)) for x,y in zip(px,py)}
    if not points:
        return np.asarray([],dtype=np.int32),np.asarray([],dtype=np.int32),0
    offsets = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    neighbors = {p: tuple((p[0]+dx,p[1]+dy) for dx,dy in offsets
                          if (p[0]+dx,p[1]+dy) in points) for p in points}
    cur = (int(px[0]),int(py[0]))
    unvisited = points - {cur}
    walk = [cur]
    backtracks=0
    while unvisited:
        opts = [q for q in neighbors[cur] if q in unvisited]
        if opts:
            nxt=opts[0]
            walk.append(nxt);unvisited.remove(nxt);cur=nxt
            continue
        # BFS 只遍历同一像素邻接图，绝不制造不连续的虚假 Line 片段。
        queue=deque([cur]); parent={cur:None}; goal=None
        while queue and goal is None:
            p=queue.popleft()
            for q in neighbors[p]:
                if q in parent: continue
                parent[q]=p
                if q in unvisited:
                    goal=q;break
                queue.append(q)
        if goal is None:
            raise ValueError('path_labels 存在不连通的像素组，请检查骨架连通性')
        route=[]; node=goal
        while node != cur:
            route.append(node);node=parent[node]
        route.reverse()
        for q in route:
            walk.append(q)
            if q in unvisited:unvisited.remove(q)
            else:backtracks += 1
        cur=goal
    xs=np.fromiter((p[0] for p in walk),dtype=np.int32,count=len(walk))
    ys=np.fromiter((p[1] for p in walk),dtype=np.int32,count=len(walk))
    return xs,ys,backtracks




def junction_keypoint_anchors(keypoints, crossing_numbers, shape, *, max_span=2):
    """为 8 邻接的多像素交点簇提供统一亚像素几何中心，不合并拓扑节点。

    CN>=3 的相邻像素在粗线/斜线交叉处有时同属一个物理交点。
    若簇的 X/Y 范围均不超过 max_span，则几何拟合使用簇质心；
    原有 Keypoint ID、CN 标签、Line 关联和 Region 拓扑全部保持原样。
    返回 (N_k,2) float 几何锚点，以及 (N_k,) 同簇成员数。
    """
    from scipy.ndimage import label as cc_label
    pts=np.asarray(keypoints,dtype=np.int32).reshape(-1,2)
    cn=np.asarray(crossing_numbers,dtype=np.int32).reshape(-1)
    if len(pts)!=len(cn):
        raise ValueError('keypoints / crossing_numbers 数量不一致')
    h,w=shape
    anchors=pts.astype(np.float64).copy()
    sizes=np.ones(len(pts),dtype=np.int32)
    chosen=np.flatnonzero(cn>=3)
    if not chosen.size:return anchors,sizes
    mask=np.zeros((h,w),dtype=bool)
    mask[pts[chosen,1],pts[chosen,0]]=True
    groups,n=cc_label(mask,structure=np.ones((3,3),dtype=np.uint8))
    grouped={}
    for i in chosen:
        group=int(groups[pts[i,1],pts[i,0]])
        grouped.setdefault(group,[]).append(int(i))
    for members in grouped.values():
        coords=pts[members].astype(float)
        if len(coords)<2 or np.any(np.ptp(coords,axis=0)>max_span):
            continue
        center=coords.mean(axis=0)
        anchors[members]=center
        sizes[members]=len(members)
    return anchors,sizes

def estimate_keypoint_branch_tangent(path_x, path_y, keypoint, *, radius=12.0):
    """在关键点之外、沿单条骨架分支拟合向外的切向与可信度。

    与“最近点前后对称截取 PCA”不同：此处只追随一个向外的路径分支，
    不会把回溯段或连接处另一臂混入同一窗口。对 45° 阶梯锯齿用
    total-least-squares（正交回归）平滑，并用特征值各向异性估算可信度。

    返回 (unit_dx, unit_dy), confidence[0,1]；极短/退化分支返回零方向与 0。
    注意：保留输入骨架像素的位置和连通性，只优化供图提取使用的几何量。
    """
    xy = np.column_stack((np.asarray(path_x, np.float64),
                          np.asarray(path_y, np.float64)))
    if len(xy) < 2:
        return (0.0, 0.0), 0.0
    kp = np.asarray(keypoint, dtype=np.float64)
    distances = np.linalg.norm(xy-kp, axis=1)
    # 从最接近关键点的路径像素出发，分别尝试沿轨迹的两个方向。
    # 复杂分支的 Euler 回溯可能重访像素，选实际“向外走得最远”的分支。
    nearest = int(np.argmin(distances))
    proposals = []
    for direction in (1, -1):
        cur = nearest
        cumulative = 0.0
        chosen = [xy[cur]]
        visited = {tuple(xy[cur])}
        max_radial = float(distances[cur])
        while 0 <= cur + direction < len(xy) and cumulative < radius:
            nxt = cur + direction
            step = float(np.linalg.norm(xy[nxt] - xy[cur]))
            if step > 1.5 + 1e-6 or step < 1e-9:
                break
            cumulative += step
            cur = nxt
            point = tuple(xy[cur])
            if point in visited:
                # 回溯段不应再进入正交拟合。
                break
            radial = float(distances[cur])
            if radial + 2.0 < max_radial:
                # 已经折返回交点附近，不能污染本分支方向。
                break
            visited.add(point)
            chosen.append(xy[cur])
            max_radial = max(max_radial, radial)
        if len(chosen) > 1:
            proposals.append((max_radial, cumulative, np.asarray(chosen)))
    if not proposals:
        return (0.0, 0.0), 0.0
    _, travel, pts = max(proposals, key=lambda v:(v[0],v[1],len(v[2])))
    # 丢弃紧靠 junction 的 0~1 像素，弱化中心处的不稳定锯齿。
    away = np.linalg.norm(pts-kp,axis=1)
    take = away >= 1.25
    if np.count_nonzero(take) >= 3:
        pts=pts[take]
    if len(pts) < 2:
        return (0.0, 0.0), 0.0
    delta = pts[-1]-pts[0]
    outward = pts.mean(axis=0)-kp
    if np.linalg.norm(outward) < 1e-8:
        outward=delta
    # 对关键点附近赋稍高权重，但不让首像素的单点方向主导拟合。
    arc = np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    weight = 1.0/(1.0+(arc/max(radius*.75,1.0))**2)
    centroid = np.average(pts,axis=0,weights=weight)
    centered=pts-centroid
    covariance=(centered*weight[:,None]).T@centered/max(float(weight.sum()),1e-12)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    v=eigenvectors[:,-1]
    if np.dot(v,outward)<0:
        v=-v
    # eig 线性度: 1 表示理想直线，0 表示方向歧义（如环形）。
    anisotropy=max(0.0,float(eigenvalues[-1]-eigenvalues[0]))/max(float(eigenvalues[-1]+eigenvalues[0]),1e-12)
    confidence=anisotropy*min(1.0,max(0.0,travel)/6.0)
    return (float(v[0]),float(v[1])),float(np.clip(confidence,0.0,1.0))


def infer_line_plus_region_a(path_x, path_y, direction, region_a, region_b, markers,
                             *, sample_count=9, radii=(2.0, 3.0)):
    """从多处局部切向两侧取样，判断面积排序的 Region A 是否位于 +n 侧。

    +n 的有向定义始终与主程序 feat_dir 一致，绝不自行翻转 Side 标签。
    返回 (plus_is_a: bool, confidence: float)。证据不足时返回 (True,0)；
    调用方可用旧中点采样作兜底。rA/rB 为 0-based，markers 为 1-based。
    """
    px=np.asarray(path_x,dtype=np.float64)
    py=np.asarray(path_y,dtype=np.float64)
    vec=np.asarray(direction,dtype=np.float64)
    h,w=markers.shape
    if len(px)<2 or np.linalg.norm(vec)<1e-8 or region_a==region_b:
        return True,0.0
    vec=vec/np.linalg.norm(vec)
    plus_a=plus_b=0.0
    n=len(px)
    # 不使用恰在 KP 的首尾样本：该位置易含其他分支/边界混叠。
    span=max(1,int(n*.13))
    lo=min(span,n-1);hi=max(lo,n-1-span)
    indices=np.unique(np.linspace(lo,hi,min(sample_count,max(2,hi-lo+1)),dtype=int))
    for k in indices:
        i=max(0,int(k)-5);j=min(n-1,int(k)+5)
        if j-i<2:continue
        part=np.column_stack((px[i:j+1],py[i:j+1]))
        start_end=part[-1]-part[0]
        if np.linalg.norm(start_end)<1.5:continue
        # 局部 TLS 使斜线锯齿收敛为连续切向；依据全局方向固定符号。
        centered=part-part.mean(axis=0)
        cov=centered.T@centered
        values,vectors=np.linalg.eigh(cov)
        tang=vectors[:,-1]
        align=float(np.dot(tang,vec))
        if abs(align)<.25:continue  # 剧烈回折或接近自环，方向证据不足
        tang*=1 if align>0 else -1
        normal=np.array([-tang[1],tang[0]])
        for radius in radii:
            p0=np.rint(np.array([px[k],py[k]])+radius*normal).astype(int)
            m0=np.rint(np.array([px[k],py[k]])-radius*normal).astype(int)
            rp=int(markers[p0[1],p0[0]]) if 0<=p0[0]<w and 0<=p0[1]<h else 0
            rm=int(markers[m0[1],m0[0]]) if 0<=m0[0]<w and 0<=m0[1]<h else 0
            a,b=int(region_a)+1,int(region_b)+1
            if rp==a:plus_a+=1
            if rm==b:plus_a+=1
            if rp==b:plus_b+=1
            if rm==a:plus_b+=1
    evidence=plus_a+plus_b
    if evidence<1e-8 or abs(plus_a-plus_b)<1e-8:
        return True,0.0
    return plus_a>plus_b,float(abs(plus_a-plus_b)/evidence)

def skeleton_contact_pairs(line_label_map, skeleton_mask):
    """线条像素之间的真实 8 邻接，按无向线 ID 去重；不逐像素 Python 循环。"""
    lab=np.asarray(line_label_map)
    sk=np.asarray(skeleton_mask,dtype=bool)
    if lab.shape != sk.shape:raise ValueError('line_label_map / skeleton_mask 尺寸不一致')
    found=[]
    # 只扫八邻接的一半，每条像素邻接边恰访问一次。
    for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
        h,w=lab.shape
        ya,yb=max(0,-dy),min(h,h-dy)
        xa,xb=max(0,-dx),min(w,w-dx)
        a=lab[ya:yb,xa:xb];b=lab[ya+dy:yb+dy,xa+dx:xb+dx]
        mask=(sk[ya:yb,xa:xb] & sk[ya+dy:yb+dy,xa+dx:xb+dx]
              & (a>=0) & (b>=0) & (a!=b))
        if mask.any():
            aa=a[mask];bb=b[mask]
            found.append(np.stack([np.minimum(aa,bb),np.maximum(aa,bb)],axis=1))
    if not found:return set()
    return set(map(tuple,np.unique(np.concatenate(found,axis=0),axis=0).tolist()))


def unify_target_by_regions(lineart_rgba, target_rgba, min_fraction=0.15):
    """用线稿 4 连通 Region 中的非透明颜色众数统一 GT；仅扫描图像有限次数。

    保持主程序旧规则：分母为整个 Region 面积；若众数像素占比不足阈值，
    设为透明；并列众数选光栅首个颜色，墨线位置保持原 GT 不变。
    返回 (统一后的 RGBA, 1-based region_markers, num_regions)。
    """
    from scipy.ndimage import label as cc_label
    rgba=np.asarray(lineart_rgba)
    target=np.asarray(target_rgba)
    if rgba.shape!=target.shape or rgba.ndim!=3 or rgba.shape[-1]!=4:
        raise ValueError('线稿与 GT 必须同尺寸 RGBA')
    inside=(rgba[...,3]==0)
    cross=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    markers,n_regions=cc_label(inside,structure=cross)
    valid=inside & (target[...,3]>0)
    r=target.reshape(-1,4).astype(np.uint32)
    packed=(r[:,0]<<24)|(r[:,1]<<16)|(r[:,2]<<8)|r[:,3]
    modes,mode_counts=_metric_modes_by_region(markers,valid,packed)
    areas=np.bincount(markers.ravel(),minlength=n_regions+1)
    accepted=(mode_counts.astype(np.float64)>=float(min_fraction)*areas)
    accepted[0]=False
    lut=np.where(accepted,modes,0).astype(np.uint32)
    selected=lut[markers[inside]]
    out=target.copy()
    out[inside,0]=((selected>>24)&255).astype(np.uint8)
    out[inside,1]=((selected>>16)&255).astype(np.uint8)
    out[inside,2]=((selected>>8)&255).astype(np.uint8)
    out[inside,3]=(selected&255).astype(np.uint8)
    return out,markers,int(n_regions)

# ==================== 3. 图像预处理 ====================
def process_alpha(img: np.ndarray) -> np.ndarray:
    """把 OpenCV 读取的灰度/BGR/BGRA 图转换为 RGBA uint8，并保留 alpha。

    支持 (H,W)、(H,W,1)、(H,W,3)、(H,W,4) 的 uint8 / uint16 图像。
    16 位图像按 0..65535 映射到 0..255，以保证后续颜色统计的位宽一致。
    """
    if img is None or img.ndim not in (2, 3):
        raise ValueError('img 必须是二维或三维图像数组')
    if img.dtype == np.uint16:
        img = ((img.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f'不支持的图像 dtype: {img.dtype}（需要 uint8 或 uint16）')

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    channels = img.shape[2]
    if channels == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    if channels == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    raise ValueError(f'不支持的图像通道数: {channels}（需要 1、3 或 4）')


# 小封闭透明区域的最大面积：只用于预处理，不删除骨架短枝。
SMALL_CLOSED_REGION_MAX_AREA = 4


def _fill_small_components(lineart_rgb: np.ndarray,
                           max_size: int = SMALL_CLOSED_REGION_MAX_AREA) -> np.ndarray:
    """删除 4 连通、未接触画框的微小透明 Region；各像素复制最近的原始墨线 RGBA。

    不以“主线色”代替相邻笔画色，也不从其它透明像素采样颜色。实际的最近邻
    只在原始 alpha>0 的笔画像素中寻找（欧氏距离），避免旧版 EDT(fill_mask)
    把非目标透明背景当作颜色来源、导致小孔依然透明。

    返回副本；不修改调用方输入；边界连通的透明区域及大于 max_size 的区域不动。
    """
    result = np.asarray(lineart_rgb).copy()
    ink = result[..., 3] > 0
    if max_size < 1 or not ink.any() or not (~ink).any():
        return result

    # 4 连通与图提取/RegionAcc 一致。标签 0 为墨线。
    bg_labels, n_bg = label(~ink, structure=_CROSS_STRUCT)
    if n_bg == 0:
        return result
    areas = np.bincount(bg_labels.ravel(), minlength=n_bg + 1)
    small = areas <= max_size
    small[0] = False
    # 任何接触画框的透明连通区都不能填，否则会损伤边界分区。
    border_labels = np.unique(np.concatenate((
        bg_labels[0, :], bg_labels[-1, :],
        bg_labels[:, 0], bg_labels[:, -1],
    )))
    small[border_labels] = False
    fill_mask = small[bg_labels]
    if not fill_mask.any():
        return result

    # 距离变换的 0 像素是“原始墨线”：只从 ink 中读取颜色。
    # 注意：不能使用 distance_transform_edt(fill_mask)，它的 0 还包含透明背景。
    _, nearest_ink = distance_transform_edt(~ink, return_indices=True)
    result[fill_mask] = lineart_rgb[
        nearest_ink[0][fill_mask], nearest_ink[1][fill_mask]]
    return result


def prepare_lineart_for_graph(lineart_rgba: np.ndarray) -> np.ndarray:
    """训练、推理、调色板和骨架化的统一线稿预处理入口（幂等）。

    先规范白底与透明背景，再将极小封闭透明区填成邻近原墨线颜色。
    只改变原线稿的像素掩码，不在原图上原位修改。适用于 BGRA 经过
    process_alpha() 转成的 RGBA 数组。
    """
    return _fill_small_components(normalize_lineart_rgba(lineart_rgba))


# 4-连通结构: 与 gnn_colorizer_v6_edge_ll.py 的 region 分割一致, 防止对角线穿过线条交叉点导致区域合并
_CROSS_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _border_side_ink_runs(ink: np.ndarray):
    """分别枚举上/右/下/左四条边的接触段；不能把跨角的接触合成一个整体。

    同一 ink 连通笔触可能在不同边有多个锚点，仅保证其中一个触边是不充分的。
    返回 [(side, [(y,x), ...]), ...]。
    """
    h, w = ink.shape
    if not h or not w:
        return []
    sides = (
        ('top', [(0, x) for x in range(w)]),
        ('right', [(y, w - 1) for y in range(h)]),
        ('bottom', [(h - 1, x) for x in range(w)]),
        ('left', [(y, 0) for y in range(h)]),
    )
    result = []
    for side, pts in sides:
        active = np.fromiter((ink[y, x] for y, x in pts), bool, count=len(pts))
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        result.extend((side, pts[int(a):int(b)+1]) for a, b in zip(starts, ends))
    return result


def _region_partition_status(ink: np.ndarray, skeleton: np.ndarray):
    """比较*具体的* 4-连通背景分区，而非只比较 Region 总数。

    若两个原始 Region 被并入同一骨架背景区域，同时骨架内部凭空出现另一个
    透明洞，Region 总数可能碰巧相等；仅比较 n_regions 无法发现。
    返回 (是否一致, 原Region数, 骨架Region数, 合并数, 凭空出现的Region数)。
    """
    orig, no = label(~ink, structure=_CROSS_STRUCT)
    new, nn = label(~skeleton, structure=_CROSS_STRUCT)
    corresponding = np.unique(new[orig > 0])
    orphan = int(nn - np.count_nonzero(np.isin(np.arange(1, nn+1), corresponding)))
    merged = int(no - len(corresponding))
    return merged == 0 and orphan == 0 and no == nn, int(no), int(nn), merged, orphan


def _restore_boundary_contacts(original_ink: np.ndarray, skeleton: np.ndarray,
                               *, reference_regions: int = None) -> np.ndarray:
    """将四边的每个原始触边段接到同一墨线分量的骨架主干。

    与 v64 不同，不因为某个*跨角点的大接触段*已经有一个骨架像素就放弃
    底边/侧边的端点；也不认为孤立的贴边骨架像素等价于连到笔画主干。
    只沿原始 ink 像素做最短路径连接，并拒绝新增不存在于原图的背景分区。
    """
    from collections import deque
    ink = np.asarray(original_ink, dtype=bool)
    out = np.asarray(skeleton, dtype=bool).copy()
    if ink.ndim != 2 or ink.shape != out.shape:
        raise ValueError('原始墨线和骨架需要尺寸相同的二维布尔数组')
    if not ink.any():
        return out
    h, w = ink.shape
    ink_cc, _ = label(ink, structure=np.ones((3, 3), dtype=np.uint8))
    n_ref = (int(reference_regions) if reference_regions is not None
             else int(label(~ink, structure=_CROSS_STRUCT)[1]))
    n_cur = int(label(~out, structure=_CROSS_STRUCT)[1])
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0),
               (-1, -1), (-1, 1), (1, -1), (1, 1))

    for side, run in _border_side_ink_runs(ink):
        orig_component = int(ink_cc[run[len(run) // 2]])
        if orig_component == 0:
            continue
        sk_cc, ns = label(out, structure=np.ones((3, 3), dtype=np.uint8))
        # 按每个原始墨线连通分量选出最大的骨架主干，避免仅连接孤立的边缘毛刺。
        branch_ids = sk_cc[(ink_cc == orig_component) & out]
        if branch_ids.size == 0:
            continue
        counts = np.bincount(branch_ids, minlength=ns+1)
        counts[0] = 0
        trunk = int(np.argmax(counts))
        if any(sk_cc[y, x] == trunk for y, x in run):
            continue
        anchor = run[len(run)//2]
        queue = deque([anchor]); parent = {anchor: None}; target = None
        while queue:
            y, x = queue.popleft()
            if sk_cc[y, x] == trunk:
                target = (y, x)
                break
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                q = (ny, nx)
                if (0 <= ny < h and 0 <= nx < w
                    and ink_cc[ny, nx] == orig_component and q not in parent):
                    parent[q] = (y, x)
                    queue.append(q)
        if target is None:
            continue
        cand = out.copy()
        cur = target
        while cur is not None:
            cand[cur] = True
            cur = parent[cur]
        # 增加原图墨线不应新增原图没有的背景岛；允许逐步修复原分区合并。
        status = _region_partition_status(ink, cand)
        new_count = status[2]
        if new_count <= n_ref and status[4] == 0 and new_count >= n_cur:
            out = cand
            n_cur = new_count
    return out


def _legacy_skeletonize_lineart(image_rgb: np.ndarray) -> np.ndarray:
    """无剪枝骨架化：先合并 1～4px 的微小闭区，再保护四边接触及分区。

    被并入邻近线条的微小闭区不再作为 Region；大 Region、触边区域保持原样。
    对预处理后的线稿比较 4 连通 Region 分区映射，而不是比较原始未填洞图。
    """
    import warnings
    from skimage.morphology import skeletonize

    image_rgb = prepare_lineart_for_graph(image_rgb)
    ink = image_rgb[..., 3] > 0
    if not np.any(ink):
        return np.zeros_like(image_rgb)

    def _thin_with(mode, method):
        pad = np.pad(ink, 2, mode=mode)
        sk = thin(pad) if method == 'thin' else skeletonize(pad)
        return sk[2:-2, 2:-2]

    # 首先保留 Guo-Hall + edge 边缘延拓，异常时再试其它细化算法。
    candidates = [('edge', 'thin'), ('reflect', 'thin'),
                  ('edge', 'zhang'), ('reflect', 'zhang'),
                  ('constant', 'thin'), ('constant', 'zhang')]
    best = None
    best_score = None
    for mode, method in candidates:
        raw = _thin_with(mode, method)
        restored = _restore_boundary_contacts(ink, raw)
        ok, orig_n, sk_n, merged, orphan = _region_partition_status(ink, restored)
        # 保留与原图分区吻合的结果，且优先使用接近原来的候选骨架。
        score = (0 if ok else 1, merged + orphan, abs(orig_n - sk_n),
                 int(np.count_nonzero(restored & ~raw)))
        if best_score is None or score < best_score:
            best, best_score = restored, score
        if ok:
            break

    ok, orig_n, sk_n, merged, orphan = _region_partition_status(ink, best)
    if not ok:
        warnings.warn(
            f'骨架化 Region 拓扑未完全一致: {orig_n} -> {sk_n}，'
            f'合并={merged}，虚增={orphan}；请检查具体边界。',
            RuntimeWarning, stacklevel=2)
    result = np.zeros_like(image_rgb)
    result[best] = image_rgb[best]
    return result


def skeletonize_lineart(image_rgb: np.ndarray) -> np.ndarray:
    """Existing GNN signature, opt-in backend. Legacy behavior is default.

    LINE_SKELETON_BACKEND=boundary activates boundary-guided reconstruction.
    Region topology guards fallback to the unmodified GNN skeletonizer.
    """
    backend=os.environ.get('LINE_SKELETON_BACKEND','legacy').strip().lower()
    if backend == 'legacy':
        return _legacy_skeletonize_lineart(image_rgb)
    if backend != 'boundary':
        raise ValueError(f'unsupported LINE_SKELETON_BACKEND={backend}')
    import sys
    from pathlib import Path
    project_dir=str(Path(__file__).resolve().parents[1])
    if project_dir not in sys.path:
        sys.path.insert(0,project_dir)
    from gnn_adapter import skeletonize_lineart_rgba
    return skeletonize_lineart_rgba(prepare_lineart_for_graph(image_rgb),
                                   fallback=_legacy_skeletonize_lineart,
                                   partition_guard=True)


def crossing_number_map(skeleton: np.ndarray) -> np.ndarray:
    """8 邻接环的 crossing number 图 (int16), 非骨架像素为 0。

    定义: cn = (|r0-r1| + |r1-r2| + ... + |r7-r0|) / 2, ring 顺序 N,NE,E,SE,S,SW,W,NW。
    该函数此前在 `detect_keypoints` 与 `_detect_skeleton_corners` 中各复制了一份 (逐字相同),
    抽到这里统一维护; 输入兼容 0/1 与 0/255 骨架。
    """
    skel_mask = skeleton.astype(bool)
    h, w = skel_mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h + 1, 1:w + 1] = skel_mask.astype(np.uint8)
    ring = (padded[0:h, 1:w + 1], padded[0:h, 2:w + 2], padded[1:h + 1, 2:w + 2],
            padded[2:h + 2, 2:w + 2], padded[2:h + 2, 1:w + 1], padded[2:h + 2, 0:w],
            padded[1:h + 1, 0:w], padded[0:h, 0:w])
    diffs = np.zeros((h, w), dtype=np.int16)
    for i in range(8):
        diffs += np.abs(ring[i].astype(np.int16) - ring[(i + 1) % 8].astype(np.int16))
    return np.where(skel_mask, (diffs // 2).astype(np.int16), np.int16(0))


def detect_keypoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    # 关键点检测不修改输入，同时兼容 0/1 和 0/255 骨架。
    skeleton = skeleton.astype(bool, copy=True)
    h, w = skeleton.shape
    skel_u8 = skeleton.astype(np.uint8)
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = skel_u8

    # Crossing number map (与 _detect_skeleton_corners 共用同一实现)
    skel_mask = skeleton.astype(bool)
    cn = crossing_number_map(skeleton)

    ep_y, ep_x = np.where(skel_mask & (cn == 1))
    jn_y, jn_x = np.where(skel_mask & (cn >= 3))

    endpoints = list(zip(ep_x.tolist(), ep_y.tolist()))
    junctions = list(zip(jn_x.tolist(), jn_y.tolist()))

    # all_points 基于修改前骨架 (用于 fallback 采样)
    y_all, x_all = np.where(skel_mask)
    all_points = list(zip(x_all.tolist(), y_all.tolist()))

    # 移除紧邻 junction 的 endpoint (向量化: 膨胀 junction 掩码)
    junction_mask = np.zeros((h, w), dtype=bool)
    for jx, jy in junctions:
        junction_mask[jy, jx] = True

    remove_mask = np.zeros((h, w), dtype=bool)
    if np.any(junction_mask):
        # 对 junction 掩码做 3x3 膨胀
        jp = np.zeros((h + 2, w + 2), dtype=bool)
        jp[1:h+1, 1:w+1] = junction_mask
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                remove_mask |= jp[1+dy:h+1+dy, 1+dx:w+1+dx]

    remove_endpoints = set()
    for ex, ey in endpoints:
        if remove_mask[ey, ex]:
            remove_endpoints.add((ex, ey))
    endpoints = [p for p in endpoints if p not in remove_endpoints]

    for ex, ey in remove_endpoints:
        skeleton[ey, ex] = False

    corners = _detect_skeleton_corners(skeleton, endpoints, junctions,
                                        removed_endpoints=remove_endpoints)

    keypoints = set(endpoints + junctions + corners)
    if len(keypoints) < 4 and len(all_points) >= 4:
        # 只在尚未选中的骨架像素中补充，避免 random.sample 抽中已有关键点。
        remaining = [p for p in all_points if p not in keypoints and p not in remove_endpoints]
        n_extra = min(4 - len(keypoints), len(remaining))
        keypoints.update(remaining[:n_extra])
    # 关键点排序固定，避免 Python hash/set 顺序使同一张图的特征发生漂移。
    return sorted(keypoints)


def _detect_skeleton_corners(skeleton: np.ndarray, endpoints: List[Tuple[int, int]],
                             junctions: List[Tuple[int, int]],
                             walk_dist: int = 2,
                             corner_angle_thresh: float = 40.0,
                             min_dist_junction: int = 7,
                             nms_radius: int = 5,
                             removed_endpoints: set = None) -> List[Tuple[int, int]]:
    h, w = skeleton.shape
    skel_mask = skeleton.astype(bool)
    ring_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                    (1, 0), (1, -1), (0, -1), (-1, -1)]

    kp_exclude = list(junctions) + list(endpoints)
    if removed_endpoints:
        kp_exclude += list(removed_endpoints)

    def _trace(sx, sy, first_nx, first_ny, max_steps):
        """沿骨架追踪, 用方向评分选择最佳邻居 (避免斜线锯齿)"""
        init_dx = first_nx - sx
        init_dy = first_ny - sy
        visited = {(sx, sy)}
        cx, cy = first_nx, first_ny
        for _ in range(max_steps):
            visited.add((cx, cy))
            best_score = -999.0
            best_pos = None
            for dy, dx in ring_offsets:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < w and 0 <= ny < h
                        and skeleton[ny, nx] and (nx, ny) not in visited):
                    # 评分: 与初始方向的一致性
                    score = dx * init_dx + dy * init_dy
                    if score > best_score:
                        best_score = score
                        best_pos = (nx, ny)
            if best_pos is not None:
                cx, cy = best_pos
            else:
                break
        return cx - sx, cy - sy

    # --- 向量化预筛 (CN 定义与逐像素版本 / detect_keypoints 完全一致) ---
    # 8 邻域 (ring 顺序: N, NE, E, SE, S, SW, W, NW)
    # 0/255 的骨架输入也必须按 0/1 计算 CN。
    skel_u8 = skel_mask.astype(np.uint8)
    cn = crossing_number_map(skeleton)


    # kp_near 掩码: 对 kp_exclude 做 L1 球 (菱形) 膨胀, 与逐像素循环等价
    kp_exclude_mask = np.zeros((h, w), dtype=bool)
    for kx, ky in kp_exclude:
        if 0 <= kx < w and 0 <= ky < h:
            kp_exclude_mask[ky, kx] = True
    # 等价替换: 单次 15×15 菱形(structure=diamond) 膨胀 == 默认 3×3 十字结构迭代 min_dist_junction 次
    # (两者均得 L1 半径=min_dist_junction 的菱形球, 输出逐位一致), 但后者每像素比较数 225→5×7=35, 更快.
    # scipy 的 iterations=0 表示迭代到稳定，而非“不膨胀”。
    kp_near_mask = (ndimage.binary_dilation(kp_exclude_mask, iterations=min_dist_junction)
                    if min_dist_junction > 0 else kp_exclude_mask)

    # 候选: 仅内部像素 & 骨架 & CN==2 & 不靠近关键点
    interior = np.zeros((h, w), dtype=bool)
    interior[1:h - 1, 1:w - 1] = True
    cand_mask = skel_mask & (cn == 2) & (~kp_near_mask) & interior
    cand_y, cand_x = np.where(cand_mask)

    candidates = []
    for y, x in zip(cand_y.tolist(), cand_x.tolist()):
        ring = np.zeros(8, dtype=np.uint8)
        for i, (dy, dx) in enumerate(ring_offsets):
            ny, nx = y + dy, x + dx
            if skeleton[ny, nx]:
                ring[i] = 1

        nbrs = []
        for i, (dy, dx) in enumerate(ring_offsets):
            if ring[i]:
                nbrs.append((x + dx, y + dy))
        if len(nbrs) < 2:
            continue

        # 取前两个邻居; 若 >2 个, 用方向评分选最佳两个分支
        if len(nbrs) == 2:
            n1_px, n2_px = nbrs[0], nbrs[1]
        else:
            # 按相对第一个邻居的方向评分, 分成两组
            ref_dx = nbrs[0][0] - x
            ref_dy = nbrs[0][1] - y
            scored = []
            for nb in nbrs:
                s = (nb[0] - x) * ref_dx + (nb[1] - y) * ref_dy
                scored.append((s, nb))
            scored.sort(key=lambda t: t[0])
            n1_px = scored[-1][1]   # 最一致 (同方向)
            n2_px = scored[0][1]    # 最不一致 (另一分支)

        v1 = _trace(x, y, n1_px[0], n1_px[1], walk_dist)
        v2 = _trace(x, y, n2_px[0], n2_px[1], walk_dist)

        n1_len = math.hypot(*v1)
        n2_len = math.hypot(*v2)
        if n1_len < 1.0 or n2_len < 1.0:
            continue

        cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1_len * n2_len)
        cos_a = max(-1.0, min(1.0, cos_a))
        vec_angle = math.degrees(math.acos(cos_a))
        bend_angle = 180.0 - vec_angle

        if bend_angle > corner_angle_thresh:
            candidates.append((x, y, bend_angle))

    candidates.sort(key=lambda c: -c[2])
    corners = []
    used = set()
    for cx, cy, ca in candidates:
        if (cx, cy) in used:
            continue
        corners.append((cx, cy))
        for rdy in range(-nms_radius, nms_radius + 1):
            for rdx in range(-nms_radius, nms_radius + 1):
                used.add((cx + rdx, cy + rdy))

    return corners


# --- 区域形状特征提取 (保留全部 17 维计算) ---
# 性能优化: 缓存 mask_u8 避免重复转换, 向量化 Hu 矩计算
def compute_region_shape_features(mask: np.ndarray, region_scale: float,
                                   area_ref: float, coord_offset: tuple = (0, 0)) -> List[float]:
    """区域形状特征 (18D).

    Args:
        region_scale: 该区域自身的长度尺度 (调用方传 sqrt(area)),
                      仅用于 perimeter_norm 的长度归一化。禁止传全局统计量
                      (如全图 pos_scale): 它与本区域的几何无关, 且随裁剪/构图变化。
        area_ref:     面积归一化的参考值 (调用方传"同图区域面积的中位数"),
                      即同类节点集合自身的统计量。禁止传画幅量 (如 h*w):
                      同一内容改变留白/裁幅会让它整体漂移。
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return [0.0] * 18

    area = len(ys)
    # 面积归一化用"同类节点(区域)集合的中位数面积", 而非面积占画幅的比例:
    # 后者 (area / (h*w)) 是画幅量, 留白/裁幅一变, 所有区域的取值整体漂移
    area_norm = np.clip(area / max(area_ref, 1e-6), 0.0, 2.0)
    r_cx = float(np.mean(xs)) + coord_offset[0]
    r_cy = float(np.mean(ys)) + coord_offset[1]

    # 默认特征（在 cv2 操作失败时返回）
    default_feats = [
        area_norm, 0.5, r_cx, r_cy,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0   # 末位 = moment_compactness 默认值
    ]

    try:
        mask_u8 = mask.astype(np.uint8)

        # 注: 原 distanceTransform 计算结果 (dt_core) 从未进入输出特征, 属死代码, 已删除。

        # 二阶中心矩 → 协方差特征值 (λ1 ≥ λ2)
        # 用于伸长率/离心率: 二阶矩是解析量, 旋转下特征值不变 ⇒ 严格旋转不变,
        # 且对像素离散化稳健 (不像包围盒宽高比随旋转改变, 也不像 fitEllipse 依赖轮廓离散化)
        m = cv2.moments(mask_u8)
        if m['m00'] > 0:
            mu20 = m['mu20'] / m['m00']
            mu02 = m['mu02'] / m['m00']
            mu11 = m['mu11'] / m['m00']
            eigvals = np.linalg.eigvalsh(
                np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64))
            lam1 = max(float(eigvals[1]), 1e-12)
            lam2 = max(float(eigvals[0]), 1e-12)
        else:
            lam1 = lam2 = 1e-12

        # 伸长率: sqrt(λ2/λ1) — 圆=1, 越细长越小 (与旧版 min(w/h, h/w) 同向)
        elongation = min(math.sqrt(max(lam2 / lam1, 0.0)), 1.0)

        # Hu 矩
        hu = cv2.HuMoments(m).flatten()
        abs_hu = np.abs(hu)
        valid_mask = abs_hu > 1e-15
        hu_features = np.zeros(7, dtype=np.float64)
        if np.any(valid_mask):
            hu_features[valid_mask] = -np.sign(hu[valid_mask]) * np.log10(abs_hu[valid_mask])

        # 轮廓提取（可能触发内存错误）
        # 注: perimeter / compactness / boundary_roughness / angle 已优化移除 —
        #     仅保留 contours → convexHull → hull_area 用于 solidity。
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 未使用特征以默认值填充 (保持 18D 返回签名向后兼容)
        perimeter_norm = 0.0
        compactness = 1.0
        boundary_roughness = 1.0
        angle = 0.0

        if len(contours) > 0:
            all_pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
            hull = cv2.convexHull(all_pts)
            hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
            solidity = area / (hull_area + 1e-6)
            solidity = min(solidity, 1.0)
        else:
            solidity = 1.0

        # 离心率 (二阶矩版): sqrt(1 - λ2/λ1) — 圆=0, 越细长越接近 1
        # 旧版用 fitEllipse 拟合距离变换轮廓, 依赖轮廓离散化: 连 90° 旋转都有 ~0.018 偏差
        eccentricity = math.sqrt(max(0.0, 1.0 - lam2 / lam1))

        # 矩类紧凑度 (旋转不变, 对离散化稳健): 面积 / 等价椭圆面积
        #   等价椭圆半轴由二阶矩给出: a = 2√λ1, b = 2√λ2 ⇒ 椭圆面积 = 4π√(λ1λ2)
        #   椭圆≈1.0, 矩形≈1.27, 细长/凹陷/多分支的形状更大 → 与"圆度"同向区分。
        #   用途: 取代依赖离散轮廓周长的 compactness (4πA/P², 旋转漂移 0.28),
        #         保留其对"形状是否紧凑"的判别力, 但完全由矩决定 ⇒ 旋转不变。
        moment_compactness = area / (4.0 * math.pi * math.sqrt(lam1 * lam2) + 1e-6)
        moment_compactness = min(moment_compactness, 3.0)

        return [
            area_norm, elongation, r_cx, r_cy,
            float(hu_features[0]), float(hu_features[1]), float(hu_features[2]), float(hu_features[3]),
            float(hu_features[4]), float(hu_features[5]), float(hu_features[6]),
            perimeter_norm, compactness, boundary_roughness, angle, solidity, eccentricity,
            moment_compactness
        ]

    except cv2.error as e:
        # 只打印一次警告，避免刷屏
        if not hasattr(compute_region_shape_features, '_warned'):
            compute_region_shape_features._warned = True
            print(f"[警告] compute_region_shape_features cv2.error: {e}，使用默认特征")
        return default_feats

# ==================== 4. 图数据可视化 (Cytoscape.js 交互式) ====================

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


# ============================================================================
#  HTML 可视化模板
#  用于交互式展示线图结构 (节点 Node=Line / 区域 Region + Line-Line / Line-Region / Region-Region 边)
#  依赖: Cytoscape.js (CDN)
#  数据注入: %%GRAPH_DATA%% 占位符由 Python 端 JSON 替换
# ============================================================================
_VIS_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GNN Graph Visualizer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<style>
/* ========== 全局布局 ========== */
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  display: flex; height: 100vh;
  background: #1a1a2e; color: #e0e0e0; overflow: hidden;
}

/* ========== 左侧边栏 ========== */
#sidebar {
  width: 340px; min-width: 340px;
  background: #16213e;
  display: flex; flex-direction: column;
  border-right: 1px solid #0f3460;
  overflow-y: auto;
}
#sidebar h2 {
  padding: 16px 18px 8px;
  font-size: 15px; color: #e94560; letter-spacing: 1px;
}

/* --- 统计信息 --- */
.stat-row { display: flex; justify-content: space-between; padding: 4px 18px; font-size: 12px; }
.stat-row .val { color: #7ec8e3; font-weight: 600; }

/* --- 区块标题 --- */
.section-title {
  padding: 12px 18px 4px; font-size: 13px; color: #999;
  border-top: 1px solid #0f3460; margin-top: 8px;
}

/* --- 显示控制 --- */
#controls { padding: 10px 18px; }
#controls label {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 12px; margin: 3px 8px 3px 0;
  cursor: pointer; user-select: none;
}
#controls input[type=checkbox] { accent-color: #e94560; }
#controls button {
  background: #0f3460; color: #e0e0e0;
  border: 1px solid #1a1a5e; padding: 5px 12px;
  border-radius: 4px; cursor: pointer; font-size: 12px;
  margin: 3px 4px 3px 0;
}
#controls button:hover { background: #1a1a5e; }
#controls button.active { background: #e94560; border-color: #e94560; }

/* --- 详情面板 --- */
#info-panel { flex: 1; padding: 14px 18px; overflow-y: auto; font-size: 12px; line-height: 1.7; }
#info-panel .hint { color: #666; font-style: italic; }
#info-panel .node-id { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
#info-panel .feat-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
#info-panel .feat-table td { padding: 3px 6px; border-bottom: 1px solid #0f3460; }
#info-panel .feat-table td:first-child { color: #7ec8e3; width: 45%; font-weight: 500; }
#info-panel .feat-table td:last-child { text-align: right; font-family: 'Cascadia Code', monospace; }

/* --- 投票列表样式 --- */
.vote-section { margin-top: 10px; }
.vote-section h4 {
  font-size: 12px; color: #8899bb; margin-bottom: 6px;
  border-bottom: 1px solid #0f3460; padding-bottom: 4px;
}
.vote-list {
  display: flex; flex-direction: column; gap: 3px;
}
.vote-item {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 8px; border-radius: 4px;
  background: rgba(26, 26, 46, 0.6);
}
.vote-item .color-swatch {
  width: 20px; height: 20px; border-radius: 3px;
  border: 1px solid #444466; flex-shrink: 0;
}
.vote-item .vote-info {
  flex: 1; display: flex; justify-content: space-between;
  align-items: center; font-size: 12px;
}
.vote-item .vote-label { color: #aab; }
.vote-item .vote-prob { color: #88bbdd; font-family: monospace; font-weight: 600; }
.boundary-line {
  margin: 4px 0; padding: 4px 8px; border-radius: 4px;
  background: rgba(30, 30, 60, 0.55);
  border-left: 2px solid #50c878;
}
.boundary-line .bl-head {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 12px; margin-bottom: 3px;
}
.boundary-line .bl-head .vote-label { color: #cde; font-weight: 600; }
.boundary-line .bl-true { color: #99aacc; font-size: 12px; }
.vote-bar-bg {
  width: 60px; height: 4px; background: #2a2a4e;
  border-radius: 2px; overflow: hidden; flex-shrink: 0;
}
.vote-bar-fill {
  height: 100%; border-radius: 2px;
  transition: width 0.3s ease;
}

.vote-pair {
  display: flex; gap: 20px; flex-wrap: wrap;
}
.vote-pair .vote-side {
  flex: 1; min-width: 120px;
}

/* ========== 右侧画布区域 ========== */
#cy-container {
  flex: 1; position: relative;
  /* 透明底显示网格: 所有透明层露出此网格背景 */
  background-color: #1a1a2e;
  background-image:
    linear-gradient(rgba(80,90,120,0.18) 1px, transparent 1px),
    linear-gradient(90deg, rgba(80,90,120,0.18) 1px, transparent 1px);
  background-size: 24px 24px;
}
#cy { width: 100%; height: 100%; }
#cy canvas {
  image-rendering: pixelated;
  image-rendering: -moz-crisp-edges;
  image-rendering: crisp-edges;
}
/* 独立背景 canvas: 叠在 Cytoscape 下方, 最近邻绘制原始像素, 只画可视区域 */
#bg-canvas {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;   /* 不拦截鼠标, 交给 cytoscape */
  image-rendering: pixelated;
  image-rendering: -moz-crisp-edges;
  image-rendering: crisp-edges;
  z-index: 0;
}
#cy { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }

/* --- 图例 --- */
#legend {
  position: absolute; top: 12px; right: 12px;
  background: rgba(22, 33, 62, 0.92);
  padding: 10px 14px; border-radius: 8px;
  font-size: 11px; line-height: 1.9;
  border: 1px solid #0f3460;
}
#legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
#legend .line-dot { background: #4a9eff; }
#legend .region-dot { background: #50c878; }
#legend .edge-line { display: inline-block; width: 18px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }

/* --- 背景切换按钮 --- */
#bg-controls { padding: 6px 18px 10px; display: flex; gap: 6px; flex-wrap: wrap; }
#bg-controls button {
  background: #0f3460; color: #e0e0e0;
  border: 1px solid #1a1a5e; padding: 4px 10px;
  border-radius: 4px; cursor: pointer; font-size: 11px;
}
#bg-controls button:hover { background: #1a1a5e; }
#bg-controls button.active { background: #e94560; border-color: #e94560; }

/* --- 右下角浮窗: 学习用图 (1-shot 训练样本) --- */
#train-panel {
  position: absolute; right: 12px; bottom: 12px; z-index: 30;
  background: rgba(22, 33, 62, 0.95);
  border: 1px solid #0f3460; border-radius: 8px;
  padding: 8px 10px 10px; width: 230px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.45);
}
#train-panel.collapsed .tp-body { display: none; }
#train-panel.collapsed { width: auto; }
#train-panel.big { width: 470px; }
#train-panel .tp-head {
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; color: #9fb3d1; margin-bottom: 6px;
}
#train-panel .tp-title { flex: 1; font-weight: 600; color: #e0e0e0; }
#train-panel .tp-head button {
  background: #0f3460; color: #e0e0e0; border: 1px solid #1a1a5e;
  padding: 2px 7px; border-radius: 4px; cursor: pointer; font-size: 11px;
}
#train-panel .tp-head button:hover { background: #1a1a5e; }
#train-panel .tp-body { display: flex; flex-direction: column; gap: 6px; }
#train-panel .tp-item { position: relative; cursor: zoom-in; }
#train-panel .tp-item img {
  display: block; width: 100%; border-radius: 4px; background: #fff;
  image-rendering: auto;
}
#train-panel .tp-item span {
  position: absolute; left: 4px; top: 4px;
  background: rgba(0,0,0,0.6); color: #fff;
  font-size: 10px; padding: 1px 5px; border-radius: 3px;
}
/* 点击缩略图后的全屏放大层 */
#train-zoom {
  position: fixed; inset: 0; z-index: 9999; display: none;
  background: rgba(0,0,0,0.88);
  align-items: center; justify-content: center; cursor: zoom-out;
}
#train-zoom.show { display: flex; }
#train-zoom img { max-width: 94vw; max-height: 92vh; border-radius: 6px; background: #fff; }
</style>
</head>
<body>

<!-- ==================== 左侧控制面板 ==================== -->
<div id="sidebar">
  <h2>GNN Graph Visualizer</h2>
  <div id="stats-area"></div>
  <div id="eval-metrics-area"></div>
  <div class="section-title">显示控制</div>
  <div id="controls">
    <label><input type="checkbox" id="show-l" checked> Line 节点</label>
    <label><input type="checkbox" id="show-r"> Region 节点</label>
    <label><input type="checkbox" id="show-ll" checked> Edge 边 (共享 Keypoint)</label>
    <label><input type="checkbox" id="show-lr" checked> Line-Region 边</label>
    <label><input type="checkbox" id="show-rr" checked> Region-Region 边</label>
    <br>
    <button id="btn-spatial" class="active">空间布局</button>
    <button id="btn-force">力导向</button>
    <button id="btn-fit">适应画布</button>
  </div>
  <div class="section-title">背景图</div>
  <div id="bg-controls">
    <button id="bg-skel">骨架线稿</button>
    <button id="bg-region" class="active">区域着色</button>
    <button id="bg-conf">最终颜色支持度</button>
    <button id="bg-disagreement">线间分歧</button>
    <button id="bg-err">成图 Region 对错</button>
    <button id="bg-pixel-err">逐像素 RGBA 对错</button>
  </div>
  <div class="section-title">节点 / 边详情</div>
  <div id="info-panel"><span class="hint">点击图中的节点或边查看特征信息</span></div>
</div>
<div id="cy-container">
  <canvas id="bg-canvas"></canvas>
  <div id="cy"></div>
  <div id="legend">
    <div><span class="dot line-dot"></span>Line 节点</div>
    <div><span class="dot region-dot"></span>Region 节点</div>
    <div><span class="edge-line" style="background:#ff8c42"></span>Edge (两 Line 共享 Keypoint)</div>
    <div><span class="edge-line" style="background:#9b59b6"></span>Line-Region (线→区域)</div>
    <div><span class="edge-line" style="background:#555"></span>Region-Region (共享边界)</div>
  </div>
  <div id="train-panel" style="display:none;">
    <div class="tp-head">
      <span class="tp-title">学习用图 (1-shot)</span>
      <button id="tp-big" title="放大 / 还原">放大</button>
      <button id="tp-toggle" title="隐藏 / 显示">隐藏</button>
    </div>
    <div class="tp-body" id="tp-body"></div>
  </div>
</div>
<div id="train-zoom"><img id="train-zoom-img" alt=""></div>
<script>
const DATA = %%GRAPH_DATA%%;

function buildElements(){
  const els = [];
  DATA.line_nodes.forEach(n => {
    els.push({group:'nodes',data:{id:n.id,type:'line',features:n.features,endpoints:n.endpoints,lineColor:n.color||'#ff6b6b',strokeId:n.stroke_id},position:n.pos});
  });
  DATA.region_nodes.forEach(n => {
    els.push({group:'nodes',data:{id:n.id,type:'region',features:n.features,area:n.area,size:n.size},position:n.pos});
  });
  DATA.edges.forEach(e => {
    els.push({group:'edges',data:{id:e.id,source:e.source,target:e.target,edgeType:e.edge_type,features:e.features}});
  });
  return els;
}

const spatialPositions = {};
buildElements().forEach(el => { if(el.position) spatialPositions[el.data.id] = {...el.position}; });

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: buildElements(),
  style: [
    {selector:'node[type="line"]',style:{
      'background-color':'data(lineColor)','width':6,'height':6,
      'border-width':2,'border-color':'#333',
      'label':'','font-size':'7px','color':'#c0392b','text-valign':'top','text-margin-y':-2,
      'z-index':10
    }},
    {selector:'node[type="region"]',style:{
      'background-color':'rgba(80,200,120,0.25)','width':'mapData(size,10,40,16,38)','height':'mapData(size,10,40,16,38)',
      'border-width':2,'border-color':'#27ae60','border-opacity':0.8,
      'background-opacity':0.25,
      'label':'data(id)','font-size':'8px','color':'#1e8449','text-valign':'center',
      'text-halign':'center','text-wrap':'wrap','text-max-width':'45px',
      'text-background-color':'rgba(255,255,255,0.7)','text-background-opacity':1,'text-background-padding':2,
      'z-index':10
    }},
    {selector:'edge[edgeType="Line-Line"]',style:{
      'line-color':'#ff8c42','target-arrow-color':'#ff8c42','width':2,'opacity':0.6,'curve-style':'bezier','z-index':5
    }},
    {selector:'edge[edgeType="Line-Region"]',style:{
      'line-color':'#9b59b6','target-arrow-color':'#9b59b6','width':1.5,'opacity':0.35,'curve-style':'bezier','z-index':3
    }},
    {selector:'edge[edgeType="Region-Region"]',style:{
      'line-color':'#666','target-arrow-color':'#666','width':1.5,'opacity':0.3,'curve-style':'bezier','z-index':2
    }},
    {selector:':selected',style:{
      'border-width':3,'border-color':'#e94560','line-color':'#e94560','width':4,'opacity':1,'z-index':999
    }},
    {selector:'node.highlighted',style:{'border-width':3,'border-color':'#ffd700','z-index':998}},
    {selector:'edge.highlighted',style:{'width':4,'opacity':1,'z-index':998}}
  ],
  layout:{name:'preset'},
  minZoom:0.05, maxZoom:15, wheelSensitivity:0.3
});

const bgUrls = {
  'bg-skel': 'data:image/png;base64,' + DATA.bg_skel_b64,
  'bg-region': 'data:image/png;base64,' + DATA.bg_region_b64
};
// 决策支持度/线间分歧为不同概念，都不是校准过的上色正确率。
if (DATA.bg_conf_b64) bgUrls['bg-conf'] = 'data:image/png;base64,' + DATA.bg_conf_b64;
if (DATA.bg_disagreement_b64) bgUrls['bg-disagreement'] = 'data:image/png;base64,' + DATA.bg_disagreement_b64;
if (DATA.bg_err_b64) bgUrls['bg-err'] = 'data:image/png;base64,' + DATA.bg_err_b64;
if (DATA.bg_pixel_err_b64) bgUrls['bg-pixel-err'] = 'data:image/png;base64,' + DATA.bg_pixel_err_b64;
const BG_KEYS = ['bg-skel', 'bg-region', 'bg-conf', 'bg-disagreement', 'bg-err', 'bg-pixel-err'].filter(k => !!bgUrls[k]);
// 没有对应背景的按钮直接隐藏, 避免点了没反应
['bg-skel', 'bg-region', 'bg-conf', 'bg-disagreement', 'bg-err', 'bg-pixel-err'].forEach(function(k){
  if (BG_KEYS.indexOf(k) < 0) {
    const el = document.getElementById(k);
    if (el) el.style.display = 'none';
  }
});

// ==================== 学习用图浮窗 (右下角, 可放大/隐藏) ====================
(function initTrainPanel(){
  const previews = DATA.train_previews || [];
  if (!previews.length) return;
  const panel = document.getElementById('train-panel');
  const body = document.getElementById('tp-body');
  previews.forEach(function(p){
    const item = document.createElement('div');
    item.className = 'tp-item';
    item.title = '点击放大';
    const img = new Image();
    img.src = 'data:image/png;base64,' + p.b64;
    img.alt = p.label;
    const tag = document.createElement('span');
    tag.textContent = p.label;
    item.appendChild(img);
    item.appendChild(tag);
    item.addEventListener('click', function(){
      document.getElementById('train-zoom-img').src = img.src;
      document.getElementById('train-zoom').classList.add('show');
    });
    body.appendChild(item);
  });
  panel.style.display = '';
  const btnBig = document.getElementById('tp-big');
  const btnToggle = document.getElementById('tp-toggle');
  btnBig.addEventListener('click', function(e){
    e.stopPropagation();
    panel.classList.toggle('big');
    btnBig.textContent = panel.classList.contains('big') ? '还原' : '放大';
  });
  btnToggle.addEventListener('click', function(e){
    e.stopPropagation();
    panel.classList.toggle('collapsed');
    btnToggle.textContent = panel.classList.contains('collapsed') ? '显示' : '隐藏';
  });
  const zoom = document.getElementById('train-zoom');
  zoom.addEventListener('click', function(){ zoom.classList.remove('show'); });
  document.addEventListener('keydown', function(ev){
    if (ev.key === 'Escape') zoom.classList.remove('show');
  });
})();

// ==================== 独立背景 canvas: 最近邻绘制原始像素 ====================
const bgCanvas = document.getElementById('bg-canvas');
const bgCtx = bgCanvas.getContext('2d');
let currentBgKey = 'bg-region';
const bgImages = {};   // key -> {img, loaded}
let bgCanvasW = 0, bgCanvasH = 0;

function loadBgImages(){
  Object.keys(bgUrls).forEach(function(key){
    const img = new Image();
    img.onload = function(){
      bgImages[key] = img;
      if (key === currentBgKey) requestBgDraw();
    };
    img.src = bgUrls[key];
  });
}

// 把世界坐标 (图像像素坐标) 映射到屏幕坐标: 用 cy.extent() 求当前可视世界范围。
function drawBg(){
  const img = bgImages[currentBgKey];
  const W = bgCanvas.clientWidth || bgCanvas.width;
  const H = bgCanvas.clientHeight || bgCanvas.height;
  if (W !== bgCanvasW || H !== bgCanvasH) {
    bgCanvas.width = W;
    bgCanvas.height = H;
    bgCanvasW = W; bgCanvasH = H;
  }
  bgCtx.clearRect(0, 0, W, H);
  if (!img) return;

  const ext = cy.extent();   // {x1,y1,x2,y2} 世界坐标下可视范围
  const iw = DATA.stats.img_w, ih = DATA.stats.img_h;
  // 图像在世界坐标 = [0,0] ~ [iw,ih]
  const wx1 = Math.max(0, ext.x1), wy1 = Math.max(0, ext.y1);
  const wx2 = Math.min(iw, ext.x2), wy2 = Math.min(ih, ext.y2);
  if (wx2 <= wx1 || wy2 <= wy1) return;   // 无交集

  // 从完整像素边界取样，并把这些边界映射到同一世界坐标。
  // 旧实现的源区间取 floor/ceil、目标区间却用小数边界，平移时会产生偏移和拉伸。
  const srcX = Math.floor(wx1), srcY = Math.floor(wy1);
  const srcW = Math.ceil(wx2) - srcX, srcH = Math.ceil(wy2) - srcY;
  const scaleX = W / (ext.x2 - ext.x1), scaleY = H / (ext.y2 - ext.y1);
  const dstX = (srcX - ext.x1) * scaleX, dstY = (srcY - ext.y1) * scaleY;

  bgCtx.imageSmoothingEnabled = false;   // 最近邻, 放大显示原始像素
  bgCtx.drawImage(img, srcX, srcY, srcW, srcH,
                  dstX, dstY, srcW * scaleX, srcH * scaleY);
}

let bgRafPending = false;
function requestBgDraw(){
  if (bgRafPending) return;
  bgRafPending = true;
  requestAnimationFrame(function(){
    bgRafPending = false;
    drawBg();
  });
}

function applyBg(key){
  currentBgKey = key;
  requestBgDraw();
}
loadBgImages();
cy.on('render viewport zoom pan', requestBgDraw);
applyBg('bg-region');

const s = DATA.stats;
document.getElementById('stats-area').innerHTML =
  `<div class="stat-row"><span>节点 Node (Line)</span><span class="val">${s.n_nodes}</span></div>`+
  `<div class="stat-row"><span>区域 Region</span><span class="val">${s.n_regions}</span></div>`+
  `<div class="stat-row"><span>边 Edge (共享 Keypoint)</span><span class="val">${s.n_edges}</span></div>`+
  `<div class="stat-row"><span>Line-Region 边</span><span class="val">${s.n_line_region}</span></div>`+
  `<div class="stat-row"><span>Region-Region 边</span><span class="val">${s.n_region_region}</span></div>`+
  `<div class="stat-row"><span>图像尺寸</span><span class="val">${s.img_w} x ${s.img_h}</span></div>`;

// 图结构节点个数采用骨架图；下列成图 RegionAcc 采用原线稿 4 连通分区，两者不同。
if (DATA.image_metrics) {
  const m = DATA.image_metrics;
  const fmt = (v) => (v == null || !Number.isFinite(Number(v))) ? 'N/A' : (Number(v) * 100).toFixed(2) + '%';
  const count = (v) => (v == null) ? '-' : String(v);
  const row = (label, value) => `<div class="stat-row"><span>${label}</span><span class="val">${value}</span></div>`;
  let output = '<div class="section-title">最终成图指标（非 GNN 图节点）</div>';
  output += row('成图 RegionAcc（众数）', m.region_metric_defined === false ? 'N/A' : fmt(m.region_acc));
  output += row('Region 正确/总数', `${count(m.region_correct)}/${count(m.region_total)}`);
  output += row('PixelAcc（调色板）', m.pixel_metric_defined === false ? 'N/A' : fmt(m.pixel_acc));
  output += row('Pixel mIoU（有效类宏平均）', m.pixel_metric_defined === false ? 'N/A' : fmt(m.pixel_miou));
  output += row('严格 RGBA PixelAcc', Number(m.pixel_total_exact_rgba) === 0 ? 'N/A' : fmt(m.pixel_acc_exact_rgba));
  output += row('像素覆盖率（已知 GT）', fmt(m.pixel_coverage));
  output += row('排除未知 GT 像素', count(m.excluded_unknown_gt_pixels));
  output += `<div style="padding:5px 18px;color:#8f9cbb;font-size:11px;line-height:1.5">`
          + `区域对错：每个原线稿连通区域的 RGBA 众数是否一致；逐像素对错：严格 RGBA 相等。`
          + `红=错，绿=对，黑=墨线，灰=忽略。Region 众数正确不代表区域内每个像素都正确。`
          + `像素准确率/mIoU 排除不在调色板中的 GT，请同时检查覆盖率。`
          + `</div>`;
  document.getElementById('eval-metrics-area').innerHTML = output;
}

function applyVisibility(){
  const sl=document.getElementById('show-l').checked, sr=document.getElementById('show-r').checked;
  const sll=document.getElementById('show-ll').checked, slr=document.getElementById('show-lr').checked, srr=document.getElementById('show-rr').checked;
  cy.nodes('[type="line"]').style('display',sl?'element':'none');
  cy.nodes('[type="region"]').style('display',sr?'element':'none');
  cy.edges('[edgeType="Line-Line"]').style('display',sll?'element':'none');
  cy.edges('[edgeType="Line-Region"]').style('display',slr?'element':'none');
  cy.edges('[edgeType="Region-Region"]').style('display',srr?'element':'none');
}
['show-l','show-r','show-ll','show-lr','show-rr'].forEach(id=>{
  document.getElementById(id).addEventListener('change',applyVisibility);
});
applyVisibility();

function setBg(key){
  BG_KEYS.forEach(b=>document.getElementById(b).classList.remove('active'));
  document.getElementById(key).classList.add('active');
  applyBg(key);
}
BG_KEYS.forEach(id=>{
  document.getElementById(id).addEventListener('click',()=>setBg(id));
});

document.getElementById('btn-spatial').addEventListener('click',function(){
  this.classList.add('active');
  document.getElementById('btn-force').classList.remove('active');
  cy.nodes().forEach(n=>{if(spatialPositions[n.id()])n.position(spatialPositions[n.id()]);});
  cy.fit(undefined,40);
});
document.getElementById('btn-force').addEventListener('click',function(){
  this.classList.add('active');
  document.getElementById('btn-spatial').classList.remove('active');
  cy.layout({name:'cose',animate:true,animationDuration:800,nodeRepulsion:()=>8000,idealEdgeLength:()=>80,edgeElasticity:()=>50,gravity:0.3}).run();
});
document.getElementById('btn-fit').addEventListener('click',()=>cy.fit(undefined,40));

// ==================== 缩放时同步缩小 node/edge 尺寸 ====================
// 放大图时 (zoom>1), node/edge 的屏幕尺寸按 1/zoom 缩小, 保持相对背景图的比例。
function regionBaseSize(size){
  const v = Math.max(10, Math.min(40, size||10));
  return 16 + (v-10)/(40-10)*(38-16);   // 复现 style 的 mapData(size,10,40,16,38)
}
function applyZoomScale(){
  const z = cy.zoom();
  const s = Math.max(0.15, 1/z);
  cy.nodes('[type="line"]').style({'width':6*s,'height':6*s,'border-width':Math.max(1,2*s)});
  cy.nodes('[type="region"]').forEach(function(n){
    const b = regionBaseSize(n.data('size'));
    n.style({
      'width':b*s,'height':b*s,
      'border-width':Math.max(1,2*s),
      'font-size':Math.max(4,8*s),
      'text-background-padding':Math.max(1,2*s),   // label 白底框内边距随缩放缩小
      'text-margin-y':0
    });
  });
  cy.edges('[edgeType="Line-Line"]').style({'width':2*s});
  cy.edges('[edgeType="Line-Region"]').style({'width':1.5*s});
  cy.edges('[edgeType="Region-Region"]').style({'width':1.5*s});
  // 高亮/选中边框也随缩放缩小
  cy.nodes('.highlighted').style({'border-width':Math.max(1,3*s)});
  cy.edges('.highlighted').style({'width':Math.max(1,4*s)});
  cy.nodes(':selected').style({'border-width':Math.max(1,3*s)});
}
cy.on('zoom', applyZoomScale);
applyZoomScale();

const panel = document.getElementById('info-panel');

// ==================== 投票渲染辅助函数 ====================
function renderVoteList(votes, title, limit) {
  if (!votes || votes.length === 0) return '';
  // 按概率降序排序
  const sorted = [...votes].sort((a,b) => b.prob - a.prob);
  const shown = (limit && limit > 0) ? sorted.slice(0, limit) : sorted;
  const truncated = shown.length < sorted.length;
  const head = truncated ? `${title} (Top-${limit})` : title;
  let html = `<div class="vote-section"><h4>${head}</h4><div class="vote-list">`;
  const maxProb = Math.max(...shown.map(v => v.prob), 0.01);
  for (const v of shown) {
    const pct = (v.prob * 100).toFixed(1);
    const barWidth = (v.prob / maxProb * 100).toFixed(0);
    const color = window.colorPalette && window.colorPalette[v.label] ? window.colorPalette[v.label] : '#666';
    html += `<div class="vote-item">
      <div class="color-swatch" style="background:${color};"></div>
      <div class="vote-info">
        <span class="vote-label">颜色 ${v.label}</span>
        <span class="vote-prob">${pct}%</span>
      </div>
      <div class="vote-bar-bg"><div class="vote-bar-fill" style="width:${barWidth}%;background:${color};"></div></div>
    </div>`;
  }
  if (truncated) {
    // 只展示 Top-K 时, 明确剩余概率量, 避免读者把前 K 项当成归一化后的全部。
    const rest = sorted.slice(shown.length).reduce((s, v) => s + (v.prob || 0), 0);
    html += `<div class="vote-item">
      <span class="vote-label" style="color:#889;">其余 ${sorted.length - shown.length} 类</span>
      <span class="vote-prob">${(rest * 100).toFixed(1)}%</span>
    </div>`;
  }
  html += '</div></div>';
  return html;
}

// ==================== 周边线投票渲染 ====================
// 每条边界线：线 id + 面积 A/B 侧 + 显示图标签（非 GT）+ 该侧联合边际分布
function renderBoundaryLines(lines, title) {
  if (!lines || lines.length === 0) return '';
  const displayColor = (lbl) => window.colorPalette && window.colorPalette[lbl] ? window.colorPalette[lbl] : '#666';
  let html = `<div class="vote-section"><h4>${title} (${lines.length})</h4><div class="vote-list">`;
  for (const bl of lines) {
    html += `<div class="boundary-line">`;
    html += `<div class="bl-head">
      <span class="vote-label">${bl.line_id} <span style="color:#888;font-weight:normal;">(${bl.side}, ${bl.pixel_count}px)</span></span>
      <span class="bl-true">显示图标签 <span class="color-swatch" style="display:inline-block;width:10px;height:10px;background:${displayColor(bl.label_display)};"></span> ${bl.label_display}</span>
    </div>`;
    if (bl.votes && bl.votes.length > 0) {
      const maxProb = Math.max(...bl.votes.map(v => v.prob), 0.01);
      for (const v of bl.votes) {
        const pct = (v.prob * 100).toFixed(1);
        const barWidth = (v.prob / maxProb * 100).toFixed(0);
        const color = window.colorPalette && window.colorPalette[v.label] ? window.colorPalette[v.label] : '#666';
        html += `<div class="vote-item" style="margin-left:12px;">
          <div class="color-swatch" style="background:${color};"></div>
          <div class="vote-info"><span class="vote-label">颜色 ${v.label}</span><span class="vote-prob">${pct}%</span></div>
          <div class="vote-bar-bg"><div class="vote-bar-fill" style="width:${barWidth}%;background:${color};"></div></div>
        </div>`;
      }
    } else {
      html += `<div style="color:#556688;font-size:12px;margin-left:12px;padding:2px 0;">无投票数据</div>`;
    }
    html += `</div>`;
  }
  html += '</div></div>';
  return html;
}

// ==================== 显示节点详情 ====================
function showInfo(title, type, features, extra=''){
  let html = `<div class="node-id" style="color:${type==='line'?'#4a9eff':type==='region'?'#50c878':'#e94560'}">${title}</div>`;
  if(extra) html += `<div style="color:#999;margin-bottom:6px">${extra}</div>`;

  // --- 投票展示 ---
  if (type === 'region') {
    // 实际 MRF/Unary 决策与线长加权展示来自不同来源；不要混作校准正确率。
    if (features['MRF 最终颜色（非训练 GT）'] !== undefined) {
      const c = features['MRF 最终颜色（非训练 GT）'];
      const u = features['Unary 首选颜色（无 Pairwise）'];
      const support = Number(features['Decision Support（非正确率）']);
      const disagree = Number(features['Line 分歧 JS']);
      const pal = window.colorPalette && window.colorPalette[c] || '#666';
      html += `<div class="vote-section"><h4>真实求解结果（非校准正确概率）</h4>`;
      html += `<div class="vote-item"><div class="color-swatch" style="background:${pal}"></div>`;
      html += `<span class="vote-label">联合 MRF 最终颜色 ${c}；Unary 首选 ${u}</span></div>`;
      html += `<div class="vote-item"><span class="vote-label">最终颜色周边支持度</span><span class="vote-prob">${(support*100).toFixed(1)}%</span></div>`;
      html += `<div class="vote-item"><span class="vote-label">周边 Line 分歧（JS）</span><span class="vote-prob">${(disagree*100).toFixed(1)}%</span></div>`;
      if (c !== u) html += `<div style="color:#ffc06b;font-size:11px;padding:4px;">MRF 已改变 Unary 首选；请结合支持度与线间分歧排查。</div>`;
      html += `</div>`;
    }
    // Region 投票数据存放在 features.vote_probs
    if (features.vote_probs && features.vote_probs.length > 0) {
      html += renderVoteList(features.vote_probs, '周边线长度加权边际均值（非 MRF）');
    } else {
      html += `<div class="vote-section"><h4>边际均值（非 MRF）</h4><div style="color:#556688;font-size:13px;padding:4px 0;">无投票数据</div></div>`;
    }
    // 周边线投票
    html += renderBoundaryLines(features.boundary_lines, '周边线 A/B 证据（非 MRF 信念）');
  } else if (type === 'line') {
    // Line 后验边际：已把 +n/-n 映射到面积 A/B 侧；非 MRF 最终 belief。
    let hasVote = false;
    if (features.prob_a && features.prob_a.length > 0) {
      html += `<div class="vote-pair">`;
      html += `<div class="vote-side">${renderVoteList(features.prob_a, 'A侧 边际', 3)}</div>`;
      html += `<div class="vote-side">${renderVoteList(features.prob_b || [], 'B侧 边际', 3)}</div>`;
      html += `</div>`;
      hasVote = true;
    }
    if (!hasVote) {
      html += `<div class="vote-section"><h4>投票结果</h4><div style="color:#556688;font-size:13px;padding:4px 0;">无投票数据</div></div>`;
    }
    // 预测 Triple Top-3；Direction Head 已删除，没有 dir 字段。
    if (features.top3_triples) {
      html += `<div class="vote-section"><h4>预测 Triple (Top-3)</h4><div class="vote-list">`;
      if (features.top3_triples && features.top3_triples.length > 0) {
        const maxP = Math.max(...features.top3_triples.map(t => t.prob), 0.01);
        features.top3_triples.forEach(function(t, rank){
          const cc0 = window.colorPalette && window.colorPalette[t.c0] ? window.colorPalette[t.c0] : '#666';
          const cc1 = window.colorPalette && window.colorPalette[t.c1] ? window.colorPalette[t.c1] : '#666';
          const cc2 = window.colorPalette && window.colorPalette[t.c2] ? window.colorPalette[t.c2] : '#666';
          const pct = (t.prob*100).toFixed(1);
          const bw = (t.prob/maxP*100).toFixed(0);
          html += `<div class="vote-item">`;
          html += `<span class="vote-label" style="min-width:30px;">#${rank+1}</span>`;
          html += `<div class="color-swatch" style="background:${cc0};"></div><span class="vote-label">${t.c0}</span>`;
          html += `<div class="color-swatch" style="background:${cc1};"></div><span class="vote-label">${t.c1}</span>`;
          html += `<div class="color-swatch" style="background:${cc2};"></div><span class="vote-label">线${t.c2}</span>`;
          html += `<span class="vote-prob">${pct}%</span>`;
          html += `<div class="vote-bar-bg"><div class="vote-bar-fill" style="width:${bw}%;background:${cc0};"></div></div>`;
          html += `</div>`;
        });
      } else {
        html += `<div style="color:#556688;font-size:13px;padding:4px 0;">无 triple 数据</div>`;
      }
      html += `</div></div>`;
    }

    // ---- 协同输入: 两路各自 Top-4 (乘性协同真正使用的那组数据) ----
    if (features.top4_pairs || features.top4_side_a || features.top4_side_b) {
      const _pal = function(i){ return window.colorPalette && window.colorPalette[i] ? window.colorPalette[i] : '#666'; };
      const _bar = function(prob, col){
        const bw = Math.max(0, Math.min(100, prob*100)).toFixed(0);
        return `<span class="vote-prob">${(prob*100).toFixed(1)}%</span>`
             + `<div class="vote-bar-bg"><div class="vote-bar-fill" style="width:${bw}%;background:${col};"></div></div>`;
      };
      html += `<div class="vote-section"><h4>协同输入 · 两路 Top-4</h4><div class="vote-list">`;
      if (features.top4_pairs && features.top4_pairs.length) {
        html += `<div class="vote-item"><span class="vote-label">triple · 色对分布</span></div>`;
        features.top4_pairs.forEach(function(t, rank){
          const a = _pal(t.c0), b = _pal(t.c1);
          html += `<div class="vote-item"><span class="vote-label" style="min-width:30px;">#${rank+1}</span>`
               + `<div class="color-swatch" style="background:${a};"></div><span class="vote-label">${t.c0}</span>`
               + `<div class="color-swatch" style="background:${b};"></div><span class="vote-label">${t.c1}</span>`
               + _bar(t.prob, a) + `</div>`;
        });
      }
      [['side · +n', features.top4_side_a], ['side · -n', features.top4_side_b]].forEach(function(p){
        if (!p[1] || !p[1].length) return;
        html += `<div class="vote-item" style="margin-top:6px;"><span class="vote-label">${p[0]}</span></div>`;
        p[1].forEach(function(t, rank){
          const c = _pal(t.c);
          html += `<div class="vote-item"><span class="vote-label" style="min-width:30px;">#${rank+1}</span>`
               + `<div class="color-swatch" style="background:${c};"></div><span class="vote-label">${t.c}</span>`
               + _bar(t.prob, c) + `</div>`;
        });
      });
      html += `</div></div>`;
    }
  }

  // --- 其他特征表格 ---
  html += '<table class="feat-table">';
  for(const[k,v]of Object.entries(features)){
    // 跳过我们已展示的投票字段 + triple 已展示字段
    if (k === 'vote_probs' || k === 'prob_a' || k === 'prob_b' || k === 'boundary_lines'
        || k === 'pred_a' || k === 'pred_b' || k === 'pred_line'
        || k === 'top3_triples' || k === 'top4_pairs'
        || k === 'top4_side_a' || k === 'top4_side_b') continue;
    html += `<tr><td>${k}</td><td>${v}</td></tr>`;
  }
  html += '</table>';
  panel.innerHTML = html;
}

// 注册颜色调色板到全局（供渲染函数使用）
window.colorPalette = DATA.color_palette || {};
// 默认颜色补全
const defaultColors = ['#e74c3c','#3498db','#2ecc71','#f1c40f','#9b59b6','#1abc9c','#e67e22','#e84393','#00b894','#6c5ce7'];
for (let i=0; i<20; i++) {
  if (!window.colorPalette[i]) window.colorPalette[i] = defaultColors[i % defaultColors.length];
}

cy.on('tap','node',function(evt){
  const node = evt.target;
  if(node.data('type')==='bg') return;
  cy.elements().removeClass('highlighted');
  node.addClass('highlighted');
  node.connectedEdges().addClass('highlighted');
  const d = node.data();
  const extra = d.type==='line' ? `端点: (${d.endpoints[0].join(',')}) → (${d.endpoints[1].join(',')})` : `面积: ${d.area}px`;
  showInfo(d.id, d.type, d.features, extra);
});
cy.on('tap','edge',function(evt){
  cy.elements().removeClass('highlighted');
  const edge = evt.target;
  edge.addClass('highlighted');
  edge.source().addClass('highlighted');
  edge.target().addClass('highlighted');
  const d = edge.data();
  showInfo(`${d.edgeType}: ${d.source} → ${d.target}`, 'edge', d.features);
});
cy.on('tap',function(evt){
  if(evt.target===cy){cy.elements().removeClass('highlighted');panel.innerHTML='<span class="hint">点击图中的节点或边查看特征信息</span>';}
});

cy.fit(undefined,40);
</script></body></html>'''

# ==================== 5. 全局调色板构建 ====================

def build_global_palettes(lineart_paths, colorized_paths, version="v5", save_cache=True):
    """按文件内容构建颜色表；空输入兼容旧式“仅读取已存在缓存”的用法。

    缓存文件中的第一个 pickle 对象仍为 (line_palette, color_palette) 元组，
    以兼容直接 pickle.load 的已有调用方。第二个对象存储数据集指纹；
    不再创建额外的 .meta.json 附属文件，便于 GUI 按单文件备份/清理。
    """
    lineart_paths = list(lineart_paths)
    colorized_paths = list(colorized_paths)
    cache_path = f"Cache/global_palettes_gnn_{version}.pkl"
    cache_only = not lineart_paths and not colorized_paths

    def _file_fingerprint(paths):
        result = []
        for path in paths:
            full_path = os.path.abspath(os.fspath(path))
            try:
                st = os.stat(full_path)
                result.append((full_path, st.st_size, st.st_mtime_ns))
            except OSError:
                result.append((full_path, None, None))
        return result

    fingerprint = {
        'format': 7,   # 7: 微小闭区用最近原墨线色填补；调色板线色统计同步
        'lineart': _file_fingerprint(lineart_paths),
        'colorized': _file_fingerprint(colorized_paths),
    }
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                # 无路径时保持 GUI 旧模型读取缓存的行为，并支持老格式单对象缓存。
                if (isinstance(data, tuple) and len(data) == 2
                        and (cache_only or pickle.load(f) == fingerprint)):
                    return data
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, TypeError):
            pass  # 没有指纹的老缓存、缓存损坏或数据集变化时重新生成

    print("构建全局调色板...")
    import time
    t_start = time.time()
    
    # 构建线条调色板 (RGBA, 向量化, 按像素数降序排列)
    all_line_colors_list = []
    for l_path in lineart_paths:
        if not os.path.exists(l_path):
            continue
        l_raw = cv2.imdecode(np.fromfile(l_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if l_raw is None:
            continue
        l_rgba = prepare_lineart_for_graph(process_alpha(l_raw))
        line_mask = l_rgba[..., 3] > 0          # 非透明像素 = 线条
        line_px = l_rgba[line_mask].copy()
        if len(line_px) > 0:
            # 彩色线条的半透明边缘是抗锯齿覆盖率，不是额外的“线色类别”。
            # RGBA 每个 alpha 都建一类会用黑色的多种 alpha 占满 MAX_LINE_COLORS 个桶。
            # 仅对线色身份归一化 alpha，不修改源图、骨架或最终 RGBA 颜色。
            line_px[:, 3] = 255
            all_line_colors_list.append(line_px)
    
    if all_line_colors_list:
        all_line_colors = np.vstack(all_line_colors_list)
        # 统计每种颜色的像素数，按降序排列
        line_colors_unique, line_color_counts = np.unique(all_line_colors, axis=0, return_counts=True)
        sorted_indices = np.argsort(-line_color_counts)  # 降序
        line_colors_unique = line_colors_unique[sorted_indices]
    else:
        line_colors_unique = np.array([], dtype=np.uint8).reshape(0, 4)
    
    # 构建目标调色板 (RGBA, 过滤单张图中 <10 像素的噪声色; 透明背景不计入)
    #   口径: 填充色**只**来自彩图, 但要先剔除落在"线稿笔画"上的像素再统计 ——
    #   那些像素属于线 (其颜色由 line_palette 负责), 不应参与填充色域。
    #   注意这不是"减去线色": 若某个填充色恰好与线色相同 (如纯黑), 只要它在线稿之外的
    #   彩图区域出现, 就会被正常保留为填充类。
    all_target_colors_list = []
    min_pixel_count = 10

    for _idx, c_path in enumerate(colorized_paths):
        if not os.path.exists(c_path):
            continue

        c_raw = cv2.imdecode(np.fromfile(c_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if c_raw is None:
            continue

        c_rgba = process_alpha(c_raw)          # (H, W, 4) RGBA
        opaque_mask = c_rgba[..., 3] > 0        # 透明像素 = 背景, 不参与调色板
        # 剔除线稿笔画像素 (与彩图按序一一对应; 尺寸不一致则跳过剔除)
        if _idx < len(lineart_paths):
            _l_raw = cv2.imdecode(np.fromfile(lineart_paths[_idx], dtype=np.uint8),
                                  cv2.IMREAD_UNCHANGED)
            if _l_raw is not None:
                _l_rgba = prepare_lineart_for_graph(process_alpha(_l_raw))
                if _l_rgba.shape[:2] == c_rgba.shape[:2]:
                    ink = _l_rgba[..., 3] > 0
                    opaque_mask = opaque_mask & (~ink)
        c_opaque = c_rgba[opaque_mask]

        c_uint32 = (c_opaque[:, 0].astype(np.uint32) << 24) | \
                   (c_opaque[:, 1].astype(np.uint32) << 16) | \
                   (c_opaque[:, 2].astype(np.uint32) << 8) | \
                   c_opaque[:, 3].astype(np.uint32)
        unique_uint32, counts = np.unique(c_uint32, return_counts=True)

        # 仅保留出现 >= min_pixel_count 次的颜色
        unique_uint32 = unique_uint32[counts >= min_pixel_count]

        r = (unique_uint32 >> 24) & 0xFF
        g = (unique_uint32 >> 16) & 0xFF
        b = (unique_uint32 >> 8) & 0xFF
        a = unique_uint32 & 0xFF
        unique_t = np.stack([r, g, b, a], axis=1).astype(np.uint8)

        all_target_colors_list.append(unique_t)
    
    if all_target_colors_list:
        all_target_colors = np.vstack(all_target_colors_list)
        target_colors_unique = np.unique(all_target_colors, axis=0)
    else:
        target_colors_unique = np.array([], dtype=np.uint8).reshape(0, 4)
    
    # 目标调色板**只**来自彩图 (统计前剔除线稿笔画像素), 不做"减去线色"的处理:
    #   早期实现在统计完之后再把与线色相同的颜色整类删掉, 那等于用线稿信息修改填充色域 ——
    #   当某个填充色恰好等于线色 (最典型是纯黑) 时, 该填充类会被整类删除, 对应区域只能
    #   映到最近的非黑颜色, 造成标签系统性错误 (实测 input_B 彩图含 8.31% 纯黑填充)。
    #   现在改为"在像素层面剔除线稿位置", 颜色层面不再做任何删除。
    target_colors_filtered = target_colors_unique
    
    # 确保透明背景 (alpha=0) 固定为 index 0, 作为"背景/未上色"类。
    # 目标调色板仅统计 alpha>0 的像素, 故 target_colors_filtered 中不含透明色; 透明类 (0,0,0,0)
    # 单独置顶。这样训练时透明 region 的众数 (0,0,0,0) 自然命中 index 0, 推理时预测 0 → 输出透明,
    # 彻底消除旧"白色=背景"语义 (白色不再被强制置顶, 仅在 GT 真实出现时才作为普通颜色保留)。
    transparent = np.array([0, 0, 0, 0], dtype=np.uint8)
    sorted_target = np.vstack([transparent.reshape(1, 4), target_colors_filtered]) if len(target_colors_filtered) > 0 else transparent.reshape(1, 4)
    
    # 构建字典 (转纯 Python int, 便于嵌入 torch checkpoint 且 weights_only=True 可加载)
    line_palette = {i: tuple(int(x) for x in c) for i, c in enumerate(line_colors_unique)}
    color_palette = {i: tuple(int(x) for x in c) for i, c in enumerate(sorted_target)}
    
    t_build = time.time() - t_start
    print(f"调色板构建完成: {len(line_palette)} 线色, {len(color_palette)} 目标色, 耗时 {t_build:.1f}s")

    if save_cache:
        os.makedirs("Cache", exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((line_palette, color_palette), f)
            pickle.dump(fingerprint, f)
    else:
        print("[跳过保存] 调色板缓存 (save_cache=False)")
    return line_palette, color_palette


# ==================== 6. 独立成图颜色评价 ====================
# 不依赖 GNN / checkpoint：只需要原始线稿、最终成图、GT 三张同尺寸图片。
# 输入路径由 cv2.imread(..., IMREAD_UNCHANGED) 解码并转 RGBA；numpy 输入按 RGBA 解释。

def _metric_read_rgba(source, role):
    """独立读取 RGB(A) 图片；路径使用 OpenCV 的 BGR(A) 顺序，数组已为 RGBA。"""
    if isinstance(source, (str, os.PathLike)):
        p = os.fspath(source)
        raw = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f'无法读取{role}图片: {p}')
        return process_alpha(raw), raw.ndim == 3 and raw.shape[2] == 4
    a = np.asarray(source)
    if a.ndim != 3 or a.shape[-1] not in (3, 4) or a.dtype != np.uint8:
        raise ValueError(f'{role} 数组必须为 uint8 RGB/RGBA (H,W,3/4)')
    if a.shape[-1] == 3:
        return np.concatenate((a, np.full((*a.shape[:2], 1), 255, dtype=np.uint8)), axis=-1), False
    return a.copy(), True


def _metric_pack_rgba(rgba):
    a = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4).astype(np.uint32)
    r, g, b, alpha = (a[:, i] for i in range(4))
    # PNG 的透明色可能存储任意 RGB，统一为 00000000。
    return ((np.where(alpha == 0, 0, r) << 24)
            | (np.where(alpha == 0, 0, g) << 16)
            | (np.where(alpha == 0, 0, b) << 8) | alpha)


def _metric_unpacked_color(packed):
    p = int(packed)
    return [int((p >> 24) & 255), int((p >> 16) & 255),
            int((p >> 8) & 255), int(p & 255)]


def _metric_modes_by_region(markers, valid_mask, packed):
    """每个 Region 的 RGBA 众数，平票取区域内光栅顺序最先出现的颜色。

    这与主程序 extract_ll_graph 的 Region 真值众数平票规则一致，
    避免同一图像因为两套统计的 tie-break 不同而产生假错误。
    """
    selected = valid_mask.reshape(-1)
    region_ids = markers.reshape(-1)[selected].astype(np.uint64)
    color_ids = packed[selected].astype(np.uint64)
    best_colors = np.zeros(int(markers.max()) + 1, dtype=np.uint32)
    best_counts = np.zeros(len(best_colors), dtype=np.int64)
    if not region_ids.size:
        return best_colors, best_counts
    both = (region_ids << np.uint64(32)) | color_ids
    pair, first, counts = np.unique(both, return_index=True, return_counts=True)
    group = pair >> np.uint64(32)
    # 首要排序: Region；次要排序: 频次降序；平票: 首次像素位置升序。
    order = np.lexsort((first, -counts, group))
    ordered_region = group[order]
    first_of_each = np.r_[True, ordered_region[1:] != ordered_region[:-1]]
    chosen = order[first_of_each]
    best_colors[group[chosen].astype(np.intp)] = (pair[chosen] & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    best_counts[group[chosen].astype(np.intp)] = counts[chosen]
    return best_colors, best_counts


def region_accuracy_counts(pred, truth, voted_mask=None):
    """图节点 RegionAcc：所有有效 GT Region 都计入；未投票按透明类(0)计。

    同时返回仅投票准确率和覆盖率，防止只评价有投票的容易区域。
    支持 torch.Tensor 或 numpy.ndarray。用于训练/验证统计，不参与训练梯度。
    """
    p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    y = truth.detach().cpu().numpy() if isinstance(truth, torch.Tensor) else np.asarray(truth)
    p, y = np.asarray(p).reshape(-1), np.asarray(y).reshape(-1)
    if p.shape != y.shape:
        raise ValueError(f'Region 预测/真值维度不一致: {p.shape}, {y.shape}')
    if voted_mask is None:
        voted = np.ones_like(y, dtype=bool)
    else:
        voted = (voted_mask.detach().cpu().numpy() if isinstance(voted_mask, torch.Tensor)
                 else np.asarray(voted_mask)).reshape(-1).astype(bool)
        if voted.shape != y.shape:
            raise ValueError('voted_mask 维度必须与 region truth 相同')
    valid = y >= 0
    effective_pred = np.where(voted, p, 0)
    match = effective_pred == y
    return {
        'correct': int(np.count_nonzero(match & valid)),
        'total': int(np.count_nonzero(valid)),
        'voted_correct': int(np.count_nonzero(match & valid & voted)),
        'voted_total': int(np.count_nonzero(valid & voted)),
        'unvoted': int(np.count_nonzero(valid & ~voted)),
    }


def pixel_scores_from_confusion(confusion, num_classes=None):
    """按颜色类别 macro IoU；最后一列允许为未知预测类，排除于宏平均。"""
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('confusion 必须是方阵')
    c = cm.shape[0] - 1 if num_classes is None else int(num_classes)
    if c < 0 or c > cm.shape[0]:
        raise ValueError('num_classes 不合法')
    n = int(cm.sum())
    correct = int(np.trace(cm[:c, :c]))
    class_ious = {}
    for k in range(c):
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - tp)
        fn = int(cm[k, :].sum() - tp)
        union = tp + fp + fn
        if union:
            class_ious[k] = tp / union
    return ((correct / n) if n else 0.0,
            (float(np.mean(list(class_ious.values()))) if class_ious else 0.0), class_ious)


def _metric_palette_codes(color_palette, gt_codes, pred_codes):
    """无调色板时，使用全部实际 RGBA 类别（不做 RGB 近邻分类）。"""
    if color_palette is None:
        codes = np.unique(np.concatenate((gt_codes, pred_codes)))
        return codes, {int(i): _metric_unpacked_color(p) for i, p in enumerate(codes)}
    keys = sorted(color_palette)
    if keys != list(range(len(keys))):
        raise ValueError('color_palette 类别 ID 必须为 0..C-1')
    codes = []
    for i in keys:
        color = tuple(int(v) for v in color_palette[i])
        if len(color) == 3:
            color = (*color, 255)
        if len(color) != 4:
            raise ValueError('color_palette 颜色必须为 RGB 或 RGBA')
        codes.append(_metric_pack_rgba(np.asarray(color, dtype=np.uint8).reshape(1, 1, 4))[0])
    codes = np.asarray(codes, dtype=np.uint32)
    if np.unique(codes).size != len(keys):
        raise ValueError('color_palette 中存在重复 RGBA 颜色')
    return codes, {int(k): list(color_palette[k]) for k in keys}


def _metric_map_codes(values, color_codes):
    """RGBA packed → 精确类别编号，不在调色板内映射到 C (invalid)。"""
    c = len(color_codes)
    if not c:
        return np.zeros(len(values), dtype=np.int32)
    order = np.argsort(color_codes)
    codes = color_codes[order]
    locations = np.searchsorted(codes, values)
    good = locations < c
    good[good] &= codes[locations[good]] == values[good]
    out = np.full(len(values), c, dtype=np.int32)
    out[good] = order[locations[good]]
    return out


def _metric_save_rgb(path, img):
    import_dir = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(import_dir, exist_ok=True)
    success, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f'PNG 编码失败: {path}')
    buf.tofile(os.fspath(path))


def save_error_histogram(path, regions):
    """输出区域错误直方图: 左=面积分布(对/错堆叠), 右=各面积分箱的错误率。

    regions: [{'area': int, 'correct': bool}, ...] (evaluate_coloring_images 的产物)
    只依赖 matplotlib; 不可用时静默跳过, 不影响主流程。
    """
    if not regions:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False
    a_ok = np.array([r['area'] for r in regions if r['correct']], dtype=np.float64)
    a_bad = np.array([r['area'] for r in regions if not r['correct']], dtype=np.float64)
    amax = max(float(np.max(np.concatenate([a_ok, a_bad]))), 10.0)
    bins = np.logspace(0, np.log10(amax), 22)
    centers = np.sqrt(bins[:-1] * bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=120)
    axes[0].hist([a_ok, a_bad], bins=bins, stacked=True,
                 color=['#2e9e5b', '#dc3232'], label=['correct', 'wrong'])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('region area (px, log)')
    axes[0].set_ylabel('region count')
    axes[0].set_title('region count by area')
    axes[0].legend(fontsize=8, loc='upper right')

    cnt_ok, _ = np.histogram(a_ok, bins=bins)
    cnt_bad, _ = np.histogram(a_bad, bins=bins)
    tot = cnt_ok + cnt_bad
    rate = np.where(tot > 0, cnt_bad / np.maximum(tot, 1), np.nan)
    axes[1].bar(centers, np.nan_to_num(rate) * 100.0, width=np.diff(bins) * 0.8,
                color='#c0392b', alpha=0.85)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('region area (px, log)')
    axes[1].set_ylabel('wrong rate (%)')
    axes[1].set_ylim(0, 100)
    axes[1].set_title('wrong rate by area bin')
    for ax in axes:
        ax.grid(alpha=0.25)
    n = len(regions)
    nb = len(a_bad)
    fig.suptitle('regions=%d  wrong=%d (%.1f%%)  wrong_pixels=%.1f%%'
                 % (n, nb, 100.0 * nb / max(n, 1),
                    100.0 * float(a_bad.sum()) / max(float(a_ok.sum() + a_bad.sum()), 1.0)),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    d = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.fspath(path))
    plt.close(fig)
    return True


def evaluate_coloring_images(lineart, prediction, ground_truth, *, color_palette=None,
                             output_prefix=None, lineart_white_threshold=250,
                             fill_small_holes=True, min_region_area=1):
    """独立评估最终上色图（可从文件读取，不依赖 GNN）。

    参数：
      lineart: 原始未上色线稿 PNG 路径，或 uint8 RGB(A) 数组；优先用 alpha 确定墨线。
      prediction: 最终输出图（可以已包含原线条）PNG 路径或 uint8 RGB(A) 数组。
      ground_truth: GT 彩稿 PNG 路径或 uint8 RGB(A) 数组，必需。
      color_palette: 可选 {index: RGB/RGBA}；给定后仅评价 GT 精确匹配的像素，
          不认识的 GT 像素明确排除并记录；预测未知色计为错误。
          不提供时对所有非线条像素采用精确 RGBA 对比，mIoU 类别从两图自动提取。
      output_prefix: 若给定，保存 *_errors.png（整块 Region 对错）、
          *_metrics.json。
      fill_small_holes: 默认 True，与图提取一致：不评价 <=4px 的封闭透明小区；
          小区颜色从附近原始墨线采样。False 可用于检查未填洞的原始图。

    两个 RegionAcc 的边界口径必须区分：本函数从原始线稿的墨线掩码做 4 连通分割
    （透明底按 alpha；白底不透明图按亮度阈值）；与 GNN 骨架图的节点 Region 不同。
    线条不计入指标，但依然绘制成黑色以便定位；不按线的 RGB 值剔除有效填色。
    """
    import json
    from pathlib import Path
    line, has_alpha = _metric_read_rgba(lineart, '原始线稿')
    pred, _ = _metric_read_rgba(prediction, '预测成图')
    gt, _ = _metric_read_rgba(ground_truth, 'GT')
    if line.shape != pred.shape or line.shape != gt.shape:
        raise ValueError(f'线稿、预测图、GT 尺寸不一致: {line.shape}, {pred.shape}, {gt.shape}')
    if min_region_area < 1:
        raise ValueError('min_region_area 必须 >=1')

    # 非 alpha 线稿（如白底黑线 PNG/JPG）用近白色判断背景；有 alpha 时严格以 alpha 为准。
    if has_alpha and not np.all(line[..., 3] == 255):
        raw_ink = line[..., 3] > 0
        mask_source = 'alpha'
    else:
        raw_ink = np.any(line[..., :3] < int(lineart_white_threshold), axis=-1)
        mask_source = 'white_background_threshold'
    if fill_small_holes:
        raw_mask_rgba = np.zeros_like(line)
        raw_mask_rgba[raw_ink, 3] = 255
        filled = _fill_small_components(
            raw_mask_rgba, max_size=SMALL_CLOSED_REGION_MAX_AREA)
        ink = raw_ink | (filled[..., 3] > 0)
    else:
        ink = raw_ink
    region_pixels = ~ink
    # 4 连通，严格排除原始墨线+预处理小孔。若全图为线条则不能给出有意义指标。
    structure = np.asarray([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)
    markers, n_regions = label(region_pixels, structure=structure)
    line_flat = ink.reshape(-1)
    gt_packed = _metric_pack_rgba(gt)
    pred_packed = _metric_pack_rgba(pred)
    candidate = ~line_flat

    codes, palette = _metric_palette_codes(color_palette, gt_packed[candidate], pred_packed[candidate])
    c = len(codes)
    if c > 512:
        raise ValueError(f'像素颜色类太多 ({c})；请传入精确 color_palette，防止 mIoU 受抗锯齿噪声影响')
    gt_id = _metric_map_codes(gt_packed, codes)
    pred_id = _metric_map_codes(pred_packed, codes)
    known_gt = gt_id != c
    pixel_valid = candidate & known_gt
    cm = np.zeros((c+1, c+1), dtype=np.int64)
    if np.any(pixel_valid):
        flat_bins = gt_id[pixel_valid].astype(np.int64) * (c + 1) + pred_id[pixel_valid]
        pair_codes, pair_counts = np.unique(flat_bins, return_counts=True)
        cm.reshape(-1)[pair_codes] = pair_counts
    pixel_acc, miou, class_ious = pixel_scores_from_confusion(cm, c)
    # 与调色板无关的严格 RGBA 像素准确率。用于核查“GT 色未入调色板”造成的遗漏。
    # 与 mIoU 的调色板覆盖口径有意分开，不会把真实填色颜色静默丢掉。
    exact_equal = (gt_packed == pred_packed) & candidate
    exact_correct = int(np.count_nonzero(exact_equal))
    exact_total = int(np.count_nonzero(candidate))

    # 逐 Region 直接用原始 RGBA 众数判别，不依赖给定调色板；即使正确填色与墨线同色也能判对。
    valid_region_pixels = (markers > 0)
    gt_mode, gt_mode_count = _metric_modes_by_region(markers, valid_region_pixels, gt_packed)
    pred_mode, pred_mode_count = _metric_modes_by_region(markers, valid_region_pixels, pred_packed)
    areas = np.bincount(markers.ravel(), minlength=n_regions+1)
    # 精确像素正确率按 Region 一次性聚合，避免每个 Region 都完整扫描全图。
    exact_by_region = np.bincount(markers.reshape(-1)[exact_equal], minlength=n_regions + 1)
    regions = []
    region_scores = np.zeros(n_regions+1, dtype=np.int8)  # -1 忽略, 1正确, 2错误
    for ri in range(1, n_regions+1):
        area = int(areas[ri])
        if area < min_region_area:
            region_scores[ri] = -1
            continue
        truth_code, pred_code = int(gt_mode[ri]), int(pred_mode[ri])
        same = truth_code == pred_code
        region_scores[ri] = 1 if same else 2
        exact_pixels = int(exact_by_region[ri])
        regions.append({
            'id': ri, 'area': area, 'correct': bool(same),
            'gt_rgba': _metric_unpacked_color(truth_code),
            'prediction_rgba': _metric_unpacked_color(pred_code),
            'gt_modal_fraction': float(gt_mode_count[ri]) / area,
            'prediction_modal_fraction': float(pred_mode_count[ri]) / area,
            'strict_rgba_correct_pixels': exact_pixels,
            'strict_rgba_pixel_accuracy': float(exact_pixels / area),
        })
    correct_count = sum(1 for r in regions if r['correct'])
    total_region = len(regions)
    error_regions = [r for r in regions if not r['correct']]

    if output_prefix is not None:
        prefix = os.fspath(output_prefix)
        region_img = np.full((*markers.shape, 3), 235, dtype=np.uint8)
        region_img[markers > 0] = (np.asarray([[235,235,235], [30,190,80], [230,50,50]], dtype=np.uint8)
                                   [np.maximum(region_scores[markers[markers > 0]], 0)])
        region_img[ink] = [0, 0, 0]
        pix_img = np.full((*markers.shape, 3), 235, dtype=np.uint8)
        # 与指标相同的严格 RGBA 对错：排除原线稿墨线及已填补的微小闭区。
        pix_img.reshape(-1, 3)[exact_equal] = [30, 190, 80]
        pix_img.reshape(-1, 3)[candidate & ~exact_equal] = [230, 50, 50]
        pix_img[ink] = [0, 0, 0]
        _metric_save_rgb(prefix + '_errors.png', region_img)
        _metric_save_rgb(prefix + '_pixel_errors.png', pix_img)
        # 错误区域面积直方图 (与错误图配套, 便于判断错误是"大面积误判"还是"碎块误判")
        try:
            save_error_histogram(prefix + '_error_hist.png', regions)
        except Exception as _e:
            print(f'[错误直方图] 生成失败: {_e}')

    stats = {
        'pixel_acc': float(pixel_acc), 'pixel_miou': float(miou),
        'pixel_acc_exact_rgba': (exact_correct/exact_total if exact_total else 0.0),
        'pixel_correct_exact_rgba': exact_correct,
        'pixel_total_exact_rgba': exact_total,
        'pixel_total': int(np.count_nonzero(pixel_valid)),
        'pixel_correct': int(cm.diagonal()[:c].sum()),
        'pixel_coverage': float(np.count_nonzero(pixel_valid) / max(np.count_nonzero(candidate), 1)),
        'excluded_line_pixels': int(np.count_nonzero(ink)),
        'excluded_unknown_gt_pixels': int(np.count_nonzero(candidate & ~known_gt)),
        'unknown_prediction_pixels': int(np.count_nonzero(pixel_valid & (pred_id == c))),
        'total_fill_pixels': int(np.count_nonzero(candidate)),
        'line_mask_source': mask_source,
        'region_acc': float(correct_count / total_region) if total_region else 0.0,
        'region_correct': int(correct_count), 'region_total': int(total_region),
        'region_wrong': len(error_regions), 'region_ignored_small': int(n_regions - total_region),
        'pixel_metric_defined': bool(np.any(pixel_valid)),
        'region_metric_defined': bool(total_region),
        'metric_definitions': {
            'pixel_acc': '调色板内 GT 像素上的分类准确率；未知 GT 不参与，未知预测计错误',
            'pixel_miou': '调色板中在 GT 或预测中出现的类别的宏平均 IoU；未知预测计入 FN',
            'pixel_acc_exact_rgba': '全部非线稿像素上，逐像素精确 RGBA 相等比例（透明 RGB 归零）',
            'region_acc': '原线稿 4 连通 Region 的 GT/预测 RGBA 众数匹配比例；平票取光栅首色',
            'region_correct': '按 Region 众数正确计数，非该 Region 每一个像素都正确',
            'excluded_line_pixels': '按原始线稿位置及默认的微小闭区合并规则排除的像素数',
        },
        'regions': regions,
        'class_ious': {str(k): float(v) for k,v in class_ious.items()},
        'color_palette': {str(k): v for k,v in palette.items()},
        'confusion': cm,
    }
    if output_prefix is not None:
        dest = os.fspath(output_prefix) + '_metrics.json'
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(dest, 'w', encoding='utf-8') as f:
            json.dump({k: v for k,v in stats.items() if k != 'confusion'}, f,
                      ensure_ascii=False, indent=2)
    return stats


def _main_image_metrics_cli():
    """无需加载 GNN 的独立评价入口。"""
    import argparse
    import json
    parser = argparse.ArgumentParser(description='独立评估线稿上色：原始线稿 + 最终输出图 + GT')
    parser.add_argument('--lineart', required=True, help='原始未上色线稿路径')
    parser.add_argument('--prediction', required=True, help='最终上色图路径（可包含原线稿）')
    parser.add_argument('--gt', required=True, help='真值彩稿路径（计算准确率时必需）')
    parser.add_argument('--palette-json', default=None,
                        help='可选颜色调色板 JSON，如 {"0":[0,0,0,0],"1":[255,0,0,255]}')
    parser.add_argument('--out-prefix', default='Debug/coloring_metrics', help='错误图及 JSON 的输出路径前缀')
    parser.add_argument('--min-region-area', type=int, default=1)
    parser.add_argument('--white-threshold', type=int, default=250,
                        help='白底非透明线稿的白色背景检测阈值，默认250')
    args = parser.parse_args()
    palette = None
    if args.palette_json:
        with open(args.palette_json, encoding='utf-8') as f:
            palette = {int(k): tuple(v) for k,v in json.load(f).items()}
    s = evaluate_coloring_images(
        args.lineart, args.prediction, args.gt,
        color_palette=palette, output_prefix=args.out_prefix,
        lineart_white_threshold=args.white_threshold,
        min_region_area=args.min_region_area,
    )
    def _display(v, valid=True):
        return f'{v:.4f}' if valid else 'N/A'
    print('PixelAcc(调色板)={}, PixelAcc(严格RGBA)={}, Pixel mIoU={}, '
          '成图 RegionAcc={} ({}/{})'.format(
        _display(s['pixel_acc'], s['pixel_metric_defined']),
        _display(s['pixel_acc_exact_rgba'], s['pixel_total_exact_rgba'] > 0),
        _display(s['pixel_miou'], s['pixel_metric_defined']),
        _display(s['region_acc'], s['region_metric_defined']),
        s['region_correct'], s['region_total']))
    print('有效像素={}，覆盖率={:.2%}，排除线稿={}，未知GT={}，未知预测={}'.format(
        s['pixel_total'], s['pixel_coverage'], s['excluded_line_pixels'],
        s['excluded_unknown_gt_pixels'], s['unknown_prediction_pixels']))
    print(f'已输出: {args.out_prefix}_errors.png, {args.out_prefix}_pixel_errors.png, {args.out_prefix}_metrics.json')


if __name__ == '__main__':
    _main_image_metrics_cli()


def extract_line_segment_mean_widths(graph, *, ink_mask=None, distance_map=None):
    """CPU-only GNN Line widths, one scalar per Line ID (keypoint-to-keypoint).

    Explicit opt-in call after extract_ll_graph(). This does not change the
    cached GNN graph, feature dimensions, training labels or model weights.
    No drawing / file output. To avoid recomputing DT, pass distance_map.
    """
    from pathlib import Path
    import sys
    root=str(Path(__file__).resolve().parents[1])
    if root not in sys.path: sys.path.insert(0,root)
    from gnn_adapter import parameterize_gnn_line_graph
    return parameterize_gnn_line_graph(graph,ink_mask=ink_mask,
                                       distance_map=distance_map,
                                       return_profiles=False)
