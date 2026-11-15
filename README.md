# CPU 密集交点优化（2026-11-16 增量版）

**本版本基于用户上传的工程直接修改，并用提供的 `examples/B02_transparent.png` 回归。** 主要新增：透明 BGRA Alpha 正确读取、交汇节点的图路径约束聚类、过滤交点内微小自环、可靠二度交点折叠、歧义节点降级及按段长限制裁切。所有 GNN 接口仍保持 CPU 纯计算，不自动绘图或写文件。

- 详细差异、参数与 B02 对比：[docs/B02_密集交点优化_CPU.md](docs/B02_密集交点优化_CPU.md)
- B02 基准：`OPENBLAS_NUM_THREADS=1 python bench_b02_cpu.py --repeats 3`
- 测试：`python -m pytest -q`（36 项通过）
- 含 Alpha 的输入：`cv2.imread(path, cv2.IMREAD_UNCHANGED)`，CLI 同样保留 Alpha。
- `vectorize(image, Config(), output_mode='skeleton'|'segments'|'full')` 均不做可视化或输出文件。只有显式 `export_result` / CLI 才写文件。

> 旧版 PDF `算法实现与使用手册.pdf` 为原算法流程手册；此轮的修改与 B02 实测以新文档 `docs/B02_密集交点优化_CPU.md` 和补充 PDF 为准。

---

# CPU 线稿骨架与 Segment 平均宽度（Python）

基于 Zhang et al. (2022), *Vectorizing Line Drawings of Arbitrary Thickness via Boundary-based Topology Reconstruction* (DOI: 10.1111/cgf.14485) 的**研究性工程复现**。此次版本针对 CPU 上被 GNN 调用的场景优化，**每个 Segment 只有一个平均宽度值**，且所有公共被调接口都不会生成图片、SVG、PDF、图表或写文件。

## 定义与设计取舍

- **Keypoint**：骨架的端点或交汇点；GNN 自己定义的角点也可作为关键点。
- **Segment (`G0`, `G1`, …)**：两个 Keypoint 之间一条有效的骨架段；闭合无端点环属于单独的闭环 Segment。
- **Stroke (`S0`, `S1`, …)**：多个 Segment 根据交汇处连续性组合的笔画；不同于单个 Segment。
- **线宽**：`mean_width_px`，单位像素，**每段一个浮点数**。不返回逐点 `w(s)`、左右法线距离、PCHIP 控制点，也不需要逐点宽度扫描。

核心 CPU 路径如下：

1. 骨架和拓扑：使用已有 Zhang–Suen + 论文启发的边界拓扑重建；只要骨架时传 `output_mode="skeleton"`，跳过语义 Stroke 汇总、宽度、边界 bitmap 和可视化工作。
2. 单段宽度：复用骨架重建所需的一次 EDT；对该 Segment 的骨架点去掉两端不稳定部分、最多取 96 个中间位置，使用 `2 * distance_to_background - 0.5` 近似宽度，并对宽度值做上下 10% 截尾平均。
3. **已拥有 GNN 图时更快**：直接使用 `line_label_map` (像素值=Line ID) 和 **一次** OpenCV 距离变换，通过两个 `np.bincount` 计算全部 Line 的平均宽度。无需重做骨架化、重建拓扑或逐 Line 射线扫描。

EDT 给出的平均值是**几何近似**：很短的 Segment、曲线中心偏移、交汇区和重叠线条仍会产生偏差，`quality` 只是采样数量的覆盖提示，不是绝对误差或真实边界的概率置信度。

## 安装

```bash
pip install -r requirements.txt
```

只计算线宽的 `width_profile.py` / `gnn_adapter.py` 不需要 PyTorch。运行用户的 GNN 适配副本需另外安装 `requirements-gnn.txt`。

## 推荐接口：已有 GNN 图，只取平均线宽（最快）

```python
from gnn_adapter import parameterize_gnn_line_graph

# graph 是 gnn_colorizer_v6_edge_ll.extract_ll_graph(...) 的返回字典
widths = parameterize_gnn_line_graph(graph)
ids = widths['line_ids']                       # (N_l,) 保留原 GNN Line ID
mean_width = widths['line_mean_width_px']      # (N_l,) float32，单位像素
quality = widths['line_width_quality']        # (N_l,) 0..1，采样数量指示
```

**纯函数式调用**，不触发可视化，不修改 `graph`，也不改已有 GNN 的 Line 5D / L-L Edge 2D 特征。`return_profiles=True` 只会附加轻量、每段恒宽的 `SegmentMeanWidth` 记录（而非宽度曲线）。如已准备 EDT，可传 `distance_map=...` 进一步避免重复计算。

`gnn_integration/gnn_common.py` 中也提供薄封装：

```python
from gnn_common import extract_line_segment_mean_widths
widths = extract_line_segment_mean_widths(graph)
```

**重要**：GNN 的 Line ID 基于它自己的 Keypoint 分割，不能认为 `L12` 必然等于本项目的 `G12`。此接口始终依据 GNN 的 Line 编号输出。

## 只要骨架（不需要宽度、可视化或 Stroke 结果）

```python
import cv2
from skeleton_api import extract_skeleton_mask
from vectorizer import Config

image = cv2.imread('examples/x_junction.png')
sk = extract_skeleton_mask(image, Config(compute_widths=False))  # bool H×W
```

如需 Keypoint、Segment + 每 Segment 一个宽度标量（默认不执行完整 Stroke 拓扑组装）：

```python
from skeleton_api import extract_skeleton_bundle
bundle = extract_skeleton_bundle(image)
for segment in bundle.segments:
    print(segment['id'],segment['start_node'],segment['end_node'],
          segment['mean_width_px'])
```

`vectorizer.vectorize(image, cfg, output_mode='skeleton')` 只返回重建骨架；`output_mode='segments'` 另外返回 G Segment 均宽与原始骨架图，但不执行 `assemble_strokes`；`output_mode='full'` 才计算完整语义 Stroke 拓扑。三者均不会自行写文件或绘图，只有显式 `export_result` 才会导出。

## 可选接入 GNN 骨架化后端

`gnn_integration/gnn_common.py`、`gnn_integration/gnn_colorizer_v6_edge_ll.py` 是按上传文件准备的**可选副本**。GNN 原始骨架默认不变：

```bash
# 默认 legacy 不调用新的边界骨架化
LINE_SKELETON_BACKEND=legacy python gnn_integration/gnn_colorizer_v6_edge_ll.py --help

# 可选 boundary：执行纯计算边界重建，保持 RGBA 形状及颜色
LINE_SKELETON_BACKEND=boundary python gnn_integration/gnn_colorizer_v6_edge_ll.py --help
```

切换骨架后端会改变图缓存和特征分布。Region 的 4-连通背景分区如有变化，会警告并回退原始骨架化，不能据此宣称与旧模型 checkpoint 等价。**上述开关不会自行计算宽度**；需要宽度时另行调用纯计算接口。

## 命令行导出是独立用途

仅主动调用 CLI/`export_result` 时才会执行图片/SVG/JSON 输出（不是 GNN 入口的一部分）：

```bash
python vectorizer.py examples/x_junction.png -o outputs/x --svg-mode centerline
# 如果显式选 --svg-mode original，SVG 使用每段恒定平均宽度，不再是变宽轮廓
```

其中 `outputs/x_widths.json` 每个 G 段只含一个 `mean_width_px` 等摘要字段；B 桥接图元不作为独立的宽度参数化 Segment。

## CPU 基准与测试

```bash
python benchmark_cpu.py --sizes 512 1024 2048 --lines 32 --repeats 3
python -m pytest -q
```

在本运行环境、32 条合成线段、默认 `line_label_map` 快速路径，单次完整 EDT + 所有 Segment 宽度均值的平均时间：**512² ≈7.4ms、1024² ≈13.2ms、2048² ≈59.1ms**（3 次测量；含第一次初始化时间）。这是**宽度后处理本身**的测试，不包含 `extract_ll_graph`、边界拓扑重建、GNN 训练或推理，不能用来代表整条流水线速度。

交汇拓扑算法仍采用论文启发的边界切线判据、双侧法线射线和必要的连接重建，**只取消了额外的逐点线宽参数化**，并没有不加区分地删掉拓扑重建所需的边界运算。`skeletonize` 在 CPU 上对墨线包围盒做细化，以避免反复遍历大面积空白背景。

完整字段契约与 GNN 使用注意事项见 [`docs/CPU_Segment宽度与GNN接口.md`](docs/CPU_Segment宽度与GNN接口.md)。论文复现局限见 [`docs/复现说明.md`](docs/复现说明.md)。
