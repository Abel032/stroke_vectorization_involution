# 密集交汇处拓扑重建：B02 CPU 优化说明

版本：2026-11-16 · 用户上传工程的增量修订（非论文官方实现）

## 1. 目的及输入

在既有 Boundary-based Topology Reconstruction 项目上解决：

1. 近距离却独立的交叉被欧氏距离 `junction_merge_distance` 错误合并；
2. 删除伪尖刺后，残留 2 度“交点”仍触发不必要的裁切/外推；
3. 多像素交点内部的 2～7 px 微小自环被识别为独立闭合 Stroke；
4. 薄线或短段的端点被相对长度过大的 `max_trim` 裁切；
5. 原始 B02 是 BGRA 透明底，直接按 BGR 变灰会把透明黑背景当作墨线；
6. GNN 调用必须保持 CPU 纯计算，不生成 PNG、SVG、JSON 或绘图对象。

B02 原图：2048×730×4（BGRA）。本次对比统一使用 `alpha>127`，以及 `min_component=4`。因此与用户此前在其他预处理条件下的 596 段 / 336 Stroke 统计不能直接作数值对比。

## 2. 修改内容与算法约束

### 2.1 从“欧氏距离”改为“骨架图短路径”合并交点

原代码只看两个 junction blob 的质心间距 ≤3.1px，这会把相距很近但没有物理连接的交点合在一起。现用 `cKDTree.query_pairs` 找候选，再验证两个 blob **沿原始骨架 8 邻接图可在不超过 3 条边的路径内到达**；中间像素必须为 2 度，不允许穿过第三个交点。合并后簇跨度不能超过 5px。过桥用的 2 度像素被收编进交点簇，不生成额外的微自环。`euclidean` 只作为回归/消融选项保留。

### 2.2 删除交点内部的伪闭环

仅对 `start=end=同一个真实 junction` 且总周长不超过 7px、所有点距交点锚点不超过 3px 的小循环忽略；**不删除**没有附着在交点上的独立闭环，也不会删除更长的附着闭环。这个阈值是工程启发式：极小的真实附着圆圈可能仍被误判，可用 `junction_microcycle_max_length=0` 关闭。

### 2.3 剪枝后折叠 2 度交点

如果某个原交点只剩两个有效 Segment，且两条离开节点的局部切线内积不超过 `degree2_outward_cos=0.35`，把两段的中心线样本直接沿交点拼起来，并且 **仅保留一个有效 G Segment**；另一个原 ID 记录为 `degree2_collapsed`，不记为伪尖刺删除。这避免该节点继续产生两条独立 extrapolate 桥接。

若两臂同向或者歧义太大，不强行拼接：标记为 `demoted_pseudojunctions`，原段端点与 ID 保留，但不再把该位置统计成 3 度以上的物理交点。此处属于**保守保留歧义**，并非已经恢复真实落笔路径。

### 2.4 限制裁切相对长度

原有上限：`min(max_trim, 2*EDT+3)`。

新增上限：`min(max_trim, 2*EDT+3, max(0.75, trim_fraction_limit*L))`，其中 `L` 为当前 Segment 弧长；默认 `trim_fraction_limit=0.24`。短段每端最多裁掉约 24%，防止两端裁切几乎相遇。仍使用原论文启发的边界切线一致性寻找可靠点。

### 2.5 BGRA Alpha 正确识别

当输入含透明像素时，直接根据 Alpha 构造前景掩码；透明处的 RGB 任意值不会影响结果。CLI 改用 `cv2.IMREAD_UNCHANGED` 保留通道。仅对不透明灰度/BGR/BGRA 图按亮度阈值或 Otsu 二值化。`gnn_adapter.skeletonize_lineart_rgba` 仍按既有 GNN 的 ink-alpha 规则工作，未改其默认 `legacy` 行为。

### 2.6 宽度仍只按 Segment 一个均值

**没有**恢复 PCHIP、逐点左右半径或变宽曲线。`estimate_segment_width` 复用同一个 EDT 数组，并在有效关键点段上生成一个 `mean_width_px`；GNN 有 `line_label_map` 时直接用批量 `np.bincount`，不需要重跑拓扑。宽度值是简单 EDT 近似，短斜线、交叠与端点处仍可能存在误差。

## 3. B02 同输入条件下的完整重测

复现方式：旧代码 = 此轮用户上传的 `boundary_topology_python.zip`；新代码 = 本版本；两者读取同一份 `alpha>127` 的黑白图，并设置 `min_component=4`，避免预处理差异混淆。

| 指标 | 原版 | 新版 |
|:---|---:|---:|
| 初始 Skeleton Segment | 625 | 522 |
| 分类后有效 Segment（含结构折叠） | 560 | 414 |
| 交汇节点数 | 221 | 162 |
| 其中误标为 2 度的交汇节点 | 34 | 0 |
| Semantic Stroke | 388 | 273 |
| Stroke 长度中位数（像素） | 10.48 | 16.80 |
| 桥接记录数 | 455 | 363 |
| 1 次端到端 `vectorize(..., output_mode='full')` | 9.90 s | 约 1.81 s |

B02 改版内部统计：`collapsed_degree2_junctions=42`、`demoted_pseudojunctions=13`；363 条桥接中 extrapolate=222、boundary=116、tangent=25。绝对数量下降不意味着所有交叉都正确连接。尤其 **外推仍占 61.2%**，说明边界无法提供可靠支持的地方依旧很多。

新版 CPU 基准（原图 BGRA 输入，不写任何文件，3 次测量的中位数；环境设置 `OPENBLAS_NUM_THREADS=1`）：

| 输出模式 | 中位耗时 | 附加工作 |
|:---|---:|:---|
| `skeleton` | 1.59 s | 仅返回骨架与基础元数据 |
| `segments` | 1.63 s | 再返回 414 个平均宽度标量 |
| `full` | 1.81 s | 再组装 273 条 Semantic Stroke 拓扑 |

`segments` 模式下，B02 的 `mean_width_px` 中位为 **1.50px**（Alpha 阈值 127）。这与此前用户报告的 2.30px 不是同一预处理条件，不应默认为宽度“变准确/变不准确”，可用更一致的标准图像/真值另行验证。

## 4. GNN 推荐接入

```python
# 只提骨架，调用链无文件输出或可视化
import cv2
from skeleton_api import extract_skeleton_mask, extract_skeleton_bundle
from vectorizer import Config

im = cv2.imread('examples/B02_transparent.png', cv2.IMREAD_UNCHANGED)
sk = extract_skeleton_mask(im, Config(compute_widths=False))  # bool[H,W]

# 按关键点段返回宽度标量（不组装 Semantic Stroke）
bundle = extract_skeleton_bundle(im)
for seg in bundle.segments:
    width = seg['mean_width_px']
```

若 GNN 已构建 Line 图且有 `line_label_map`：

```python
from gnn_adapter import parameterize_gnn_line_graph
widths = parameterize_gnn_line_graph(graph)
ids = widths['line_ids']                    # 保留 GNN 自身 Line ID
means = widths['line_mean_width_px']       # float32[N_line]
```

`G Segment` 与 `GNN Line` 的关键点切分可能不同，不应假定编号一一相同；此处仍**不修改**网络的 Line 5D 与 L-L Edge 2D 特征，也不进行任何 PNG/SVG 或 Matplotlib 处理。

**GNN Region 分区风险（已在 B02 核实）**：GNN 旧预处理以 `alpha>0` 定义墨线，与本次直接 B02 基准的 `alpha>127` 不同；在原始 B02 上，重建后的骨架不能保持原墨线的全部 4 连通背景 Region 一一对应，因此 `partition_guard=True` 将**回退 legacy 骨架**（或没有 fallback 时抛错）。此项保护必须保留：减少 Stroke 碎裂不能以静默改变 GNN Region 标签为代价。GNN 默认 legacy 不受影响，Segment 宽度边信息可以独立附加。若需要强制使用新骨架，必须显式关闭 guard，并重新生成缓存、审核 Region 标签及重新评估模型。

**输出掩码严格二值**：`rasterize()` 已将 `cv2.LINE_AA` 更换为 `cv2.LINE_8`，避免弱抗锯齿值在转成 bool 时扩张骨架、误连区域；但这不能保证 Region 分区一致，上述 guard 仍必要。

## 5. 参数与消融

| Config / CLI 参数 | 默认 | 含义 |
|:---|:---:|:---|
| `junction_merge_mode` / `--junction-merge-mode` | geodesic | 近点融合需图路径；euclidean=旧策略 |
| `junction_merge_distance` | 3.1 | 邻近候选筛选半径（像素） |
| `junction_merge_max_hops` | 3 | 合并允许的骨架图最短跳数 |
| `junction_blob_max_span` | 5.0 | 合并后节点簇最大跨度 |
| `junction_microcycle_max_length` | 7.0 | 删除附着微自环的最大周长 |
| `collapse_degree2` / `--no-degree2-collapse` | True | 是否合并可靠的两臂残余节点 |
| `degree2_outward_cos` | 0.35 | 两臂向外切线允许的最大余弦 |
| `demote_unresolved_pseudojunctions` | True | 将不能可靠合并的二度节点移出交点集合 |
| `trim_fraction_limit` | 0.24 | 单端最大裁切占整个段长的比例 |
| `alpha_threshold` | 127 | 透明线稿 alpha 墨线阈值 |

## 6. 测试与适用范围

```bash
pip install -r requirements.txt
python -m pytest -q              # 36/36
OPENBLAS_NUM_THREADS=1 python bench_b02_cpu.py --repeats 3
python vectorizer.py examples/B02_transparent.png -o outputs/B02 --no-widths
```

最后一条是**显式诊断导出命令**，会写图像/拓扑 JSON；与普通 GNN 调用完全分离。

对真实 B02 的局限：没有真实的每条 Stroke 标注，不能以“越少越好”为目标无限合并。高密度毛发、笔触穿越、双线贴近时仍可能误配；目前算法仍是研究型工程复现，而不是论文作者发布的完全等价实现。若要进一步提升，建议引入**交点附近的局部候选路径评分及全局一致性匹配**，并以人工标注的 junction ground truth 检测假阳性与漏连接，而非仅优化段数。
