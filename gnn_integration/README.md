# 与上传的 GNN v6 / gnn_common 的 CPU 可选接口

本目录为上传的两个 GNN 文件提供可选修改副本，不建议直接覆盖现有训练项目。`LINE_SKELETON_BACKEND=legacy` 保持原有骨架化，`boundary` 时走父目录的 `gnn_adapter.skeletonize_lineart_rgba` **纯计算路径**，不会绘图/写文件；Region 4 连通背景分区改变时回退原函数。

当前最推荐的是 **在 `extract_ll_graph(...)` 返回图字典以后直接调用**：

```python
from gnn_common import extract_line_segment_mean_widths
w = extract_line_segment_mean_widths(graph)
line_ids = w['line_ids']
widths = w['line_mean_width_px']  # 与 GNN Line 对齐，形状 (N_l,)
```

该调用优先使用 GNN 自身已有的 `line_label_map`，一次 EDT 后按 ID 批量算均宽。不会重新骨架化，不修改 `graph`、模型 Line 5D/Edge 2D、缓存/checkpoint，也不会执行可视化逻辑。需要神经网络学习线宽，需要使用方显式修改特征和重新训练。

运行 GNN 需 `pip install -r requirements-gnn.txt`。完整用法见 `../docs/CPU_Segment宽度与GNN接口.md`。
