# CPU Segment 均宽与 GNN 被调接口

## 模块结构（模块导入不会执行绘图）

| 文件 | 被调入口 | 计算 | 是否文件输出 |
|---|---|---|---|
| `width_profile.py` | `compute_distance_map(mask)` | 一次 OpenCV 精确 EDT | 否 |
| `width_profile.py` | `estimate_segment_width(mask, path_xy, distance_map=...)` | 单个 Keypoint Segment 的截尾均宽 | 否 |
| `width_profile.py` | `means_from_line_label_map(mask, line_label_map, N, distance_map=...)` | 所有带标签线的矢量化均宽 | 否 |
| `gnn_adapter.py` | `parameterize_gnn_line_graph(graph, ...)` | 保持 GNN Line ID 的宽度侧车数据 | 否 |
| `gnn_adapter.py` | `skeletonize_lineart_rgba(rgba, ...)` | RGBA 骨架、4 连通 Region 守护 | 否 |
| `skeleton_api.py` | `extract_skeleton_mask(image, config)` | bool 细化重建骨架 | 否 |
| `skeleton_api.py` | `extract_skeleton_bundle(image, config)` | 端点、交点、G Segment、每段均宽；语义 Stroke 组装可选 | 否 |
| `vectorizer.py` | `vectorize(image, config, output_mode='skeleton')` | 无宽度、无语义标注、无输出 | 否 |
| `vectorizer.py` | `vectorize(image, config, output_mode='segments')` | 仅骨架图和每 G 段均宽，免去 Stroke 组装 | 否 |
| `vectorizer.py` | `export_result(...)`、`save_svg(...)` | **仅调试/CLI** SVG/PNG/JSON | 是（显式调用） |

## 均宽定义与误差

`width ≈ 2*EDT(centerline_xy)-0.5`。传统 EDT 包括一像素边缘偏差；减去 0.5 px 是对细线宽度的经验折中。对于本项目内部 `G{sid}`，在该 segment 骨架像素中排除两端各 12%、最多取 96 个采样点，并丢弃宽度样本的上下 10% 极值，再取算术平均。这样减少交汇区共享墨线造成的高估，代价是只得到**代表性宽度值**，不保留沿路径的变宽信息。

GNN `line_label_map` 已具有 Keypoint 邻域排除规则，因此可对每一 Line 的已标记像素直接求均值：

```
w_i = sum_{pixels with label i}(2*EDT[p]-0.5) / count_{pixels with label i}
```

这是最快的默认路径，做一次全图 EDT 和两次 `np.bincount`。当某个 Line 缺少标签像素时，对该 Line 的 `bfs_order` 进行有限点位抽样回退（同样复用 EDT）。对重叠线段、极短线段或明显偏心骨架，不能保证估计与真实原始笔画粗细相等。

## 数据类型契约

```python
from gnn_adapter import parameterize_gnn_line_graph
result = parameterize_gnn_line_graph(graph, return_profiles=False)
```

`graph` 至少必须有 `line_info`，并传入 `ink_mask` 或让 `graph` 包含 `lineart_rgb` (RGBA)。可选的 `line_label_map` 应为 `(H,W)` 整数图，-1 表示背景，非负值是 Line ID；如不存在、ID 不连续或标签不足，将回退路径估计。

返回：

- `schema_version='2.0'`；
- `line_ids: np.int64[N_l]`，递增但不重新定义原始 Line ID；
- `line_mean_width_px: np.float32[N_l]`，唯一的宽度参数；
- `line_width_quality: np.float32[N_l]`，取样数量相关的 0–1 值；
- `line_sample_count: np.int32[N_l]`；
- `profiles` **仅当** `return_profiles=True` 时存在，为轻量 `SegmentMeanWidth`，每段一个恒宽，不能理解为 PCHIP。

`distance_map` 可用于一个调用链中复用已计算的 EDT：

```python
from width_profile import compute_distance_map
from gnn_adapter import parameterize_gnn_line_graph
ink = graph['lineart_rgb'][...,3] > 0
dist = compute_distance_map(ink)  # 只算一次
r = parameterize_gnn_line_graph(graph, ink_mask=ink, distance_map=dist)
```

返回值不含 tensor，不写 `graph`；如果后续网络要用宽度，需要由调用方显式构造新的特征，重训和更新缓存/checkpoint 版本。此处不会静默把第 6 维加进当前 Line 5D。

## RGBA skeleton drop-in

```python
from gnn_adapter import skeletonize_lineart_rgba
result = skeletonize_lineart_rgba(rgba, fallback=gnn_common._legacy_skeletonize_lineart)
```

`rgba` 是预处理过的 `uint8[H,W,4]`，Alpha>0 表示墨线。输出同 shape、同 dtype；Region partition guard 默认打开。若新骨架破坏原背景 4 连通分区，使用传入的 `fallback`，或在没有 fallback 时抛出 `ValueError`。该骨架后端不是“仅取均宽”的必要步骤；已有 GNN 图时，建议直接调用 `parameterize_gnn_line_graph()`。

## 性能优化细节

- **最高优先级：重用已有 GNN Line segmentation**。无须重复建骨架，`line_label_map` 直接汇总。
- EDT 用 OpenCV `distanceTransform(..., DIST_L2, DIST_MASK_PRECISE)`，C++ 实现，一次全图扫描。
- 对无 label 的备用分支，最多对单段 96 个样本做数组索引，不做数百步法线追踪。
- 对完整骨架 `output_mode='skeleton'` 或仅 Segment 均宽的 `output_mode='segments'`，不再运行 `assemble_strokes`，不构造 `boundary_mask`，不计算宽度。细化限于墨线 bounding box。
- 仅用于**拓扑正确性**的 `Boundary.initiated_batch()` 仍包含批量射线与 KDTree，不能和已移除的宽度拟合射线混淆。
- 纯计算 API 不 `import matplotlib`、不使用 `cv2.imwrite`、不调用 `save_svg`、不构造图片绘制任务。

## 不兼容变更（相对于上一版宽度方案）

移除 `width_sample_step`、`width_knot_step`、`width_ray_limit`、`width_smoothing` 等逐点宽度参数；替换为 `width_trim_fraction`、`width_max_samples`。不再存在 `.knot_s_px`、`.fitted_width_px`、`.radius_at()` 或变宽 `ribbon_polygon`。保留 `.width_at(...)` 为向后兼容方法，但始终返回**同一个均宽**。`--svg-mode original` 现在只是显式导出时按段赋固定笔画宽度，不是动态变宽多边形。此前 `line_width_stats[N,3]` 已改成分开输出的 1D 宽度、质量和数量数组。
