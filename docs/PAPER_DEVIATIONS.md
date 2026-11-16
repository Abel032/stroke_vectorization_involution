# 与论文的差异清单（Zhang et al., CGF 2022, DOI 10.1111/cgf.14485）

本仓库是论文 *Vectorizing Line Drawings of Arbitrary Thickness via Boundary-based
Topology Reconstruction* 的工程复现。下面逐条列出**与论文不一致的地方**，并标注
当前状态：`已对齐` = 已按论文改动；`保留` = 有意保留的差异及原因；`扩展` = 论文
没有、本实现新增的工程启发式。

代码位置均指 `vectorizer.py`（另有 `boundary_rays.py`、`stroke_topology.py`）。

---

## 1. 已按论文修正（2026-11-16）

### 1.1 Eq.(1)/(4) 切向一致性：`|cos|` → 有符号 `cos`

论文 Eq.(1)/(4) 是有符号点积；旧实现取 `abs()`，会把"双向反向（180°）"的假尖刺
判成完全一致。

- 现在：两侧都有边界命中时用 `max(cos_left, cos_right)`（符号保留）；只有一侧命中
  时无法判断朝向，退回 `|cos|`。Eq.(4) 同样改为有符号，并把负相关按"无证据"截到 0。
- 位置：`initiated_batch()`、`boundary_initiated_batch()`、`initiated()`。
- 依据：真实线条的两条边界沿轮廓走向一正一反，所以 `max(两侧)` 对任意骨架走向都
  恒为 +1；只有"两侧都反向"（折回型尖刺）才会被判为不一致。
- 实测（B02 800² 裁剪）：79 个尖刺候选里 2 个此前被 `abs` 漏删。

### 1.2 Eq.(3) 边界引导完备性：按论文重写

旧实现：`short ∧ (不一致点 >25%) ∧ 与同交点兄弟段共享边界段`。其中
"多数点切向不一致"恰好排除了论文要点名的**长且切向一致的假尖刺**，属于判据写反。

- 现在：建立**逐边界点**反查表 `(contour, vertex) → {骨架段}`；候选采样点若左右
  两侧的边界点都被多于一条骨架段关联（候选自身算一条），占比 ≥50%，且候选够短，
  则判为假尖刺。不再要求"多数点不一致"。
- 位置：`classify_segments()`。
- 保留的两个非论文细节（已在代码注释标明）：
  1. **长度上界 `short`**：本实现的边界段按 1px 顶点 + 枢轴切分，粒度比论文粗，
     不加长度上界时长线段共用边界也会被误删（实测整图 76 个候选里 67 个命中）。
  2. 一条**保守补充规则**：`short ∧ 与同交点兄弟段共用边界段 ∧ 多数点不一致`
     （旧规则原样保留，仅作为补充），用来兜住 3–6px 的像素级毛刺。

### 1.3 边界分段：43° 转角 + NMS → "拟合偏差 >1px" 递归细分

论文 §4.1.2：拟合曲线，找出偏离栅格 >1 像素最严重的点作为枢轴，断开重拟合，
重复直到没有枢轴。

- 现在：`_boundary_pivots()` 按"相对平滑/拟合曲线的偏差 >1px"递归细分。
- 实现差异：偏差用**沿弧高斯平滑**后的曲线计算，而不是逐段 `splprep`。两者作用
  相同（去掉像素阶梯、转角被拉平），但平滑是 O(P) 的一维滤波；实测换成逐段
  splprep 会让整体慢 ~35%（3.8s vs 2.6s，800² 裁剪）。
- `Config.boundary_corner_deg`（43°）**已不再使用**，仅为 CLI 兼容保留。

### 1.4 边界切向：栅格弦差 → 枢轴之间平滑后取切向

论文要求从拟合后的边界曲线取切向（"直接取栅格梯度在拐角/拥挤处不精确"）。

- 现在：`_arc_tangents()` 在枢轴之间对边界弧做轻度平滑（端点/转角保持不动），
  再取对称差分切向。位置：`Boundaries.__init__`。
- 这一改动会让 Eq.(1)/(4) 的判定更稳，因此尖刺判定数量会随之变化（B02 裁剪：
  tangent 类 17 → 12，completeness 类 2 → 4，总数 19 → 18）。

### 1.5 盲外推：多臂延伸求交、取最远延伸交点（论文 §4.3.2）

论文：对断点拟合 2 次多项式 → 一直延伸到边界 → 延伸臂相互求交 → **取沿延伸方向最远的
交点**作为新的交点位置 → 回剪到该点。

- `_quadratic_ray()`：从 cut-off 沿 2 次多项式**盲延伸**（不朝任何目标点收弯），
  按 0.5px 步长前进，遇到 `edt<=0`（离开墨线）或 `max_bridge` 才停。
- `_arm_intersections()`：同一交点上的多条延伸臂两两求最近点（容差 1px，
  且交点须在交点中心 `max_bridge` 内），对每根臂记录**弧长最长**的那个交点。
- `reconstruct()`：有交点的臂回剪到交点，并把末点精确落到交点坐标 —— 多条臂共享
  同一点，拓扑因此仍然连通。
- **兜底保留**：没有交点的臂（例如 T 形交点只伸出一条臂）继续走原来的
  `_quadratic_extension()`（吸附到最近的已有中心线/交点中心），这条固定连通域的
  逻辑**没有删除**，只是降级为 fallback。
- 实测（B02 800² 裁剪）：199 条盲延伸臂中有 **33 条**通过交点回剪；输出与"全部
  走吸附兜底"相比折线不同（SHA 不同），但段数/笔画数/交点数/桥接数完全一致，
  说明只改变交点附近的几何形状，不改变拓扑规模。

与论文仍有差距：论文是在多条臂之间求**最远延伸交点**并可能形成新的交点簇，
本实现逐臂独立回剪，未做"交点簇合并 + 交点重编号"（等价效果由后续
`junction_merge_distance` 的合并兜住）。

---

## 2. 有意保留的差异

| 项 | 论文 | 本实现 | 原因 |
|---|---|---|---|
| spline 平滑参数 | "smoothing parameter 0.05"（§4.1.1） | `splprep(s = 0.05·N)` | 归一化后 RMS 残差 ≈0.22px，与"平滑但贴合像素"的图示一致；按字面 `s=0.05` 会几乎插值像素阶梯 |
| 重建后拓扑 | "两段骨架在拓扑上合并成一段"（§4.3.1） | 桥接是**独立 polyline**，仅在 `assemble_strokes` 里拼成 stroke 路径 | 保持 `source_graph['segments']` 的段标识稳定，便于下游（GNN）按段索引；连通性由端点重合保证 |
| 边界段定义 | 拟合偏差细分出的"矢量化边界段" | 1px 顶点 + 枢轴分段 + 沿弧平滑切向 | 不额外保存矢量边界段（省内存、少一次拟合） |
| 初始骨架 | Zhang–Suen | Zhang–Suen（`_zhang_suen`），若 OpenCV contrib 存在则用 `ximgproc.thinning` | 逐位一致；有更快的等价实现时优先 |

## 3. 论文没有的工程扩展（不算"错误"，但会影响结果）

- `junction_merge_mode='geodesic'`（≤2 跳、span≤5px）与 `junction_merge_distance`
- 交点内 2–7px 微自环过滤（`junction_microcycle_*`）+ `add_path` 里的 2×2 环保护
- `collapse_degree2`（二度伪交点折叠）与 `demote_unresolved_pseudojunctions`
- `trim_fraction_limit=0.24`（裁切量不超过段长比例）、`max_trim=22`
- 射线长度 `min(ray_limit, 2.8·EDT+4)`；`boundary_normal_dot_limit=0.34`、
  `boundary_query_k=48`
- `diagonal_policy='topology_safe'`（抑制 2×2 冗余对角边）
- `min_component=4` 小连通域清理、`alpha_threshold=127` 透明 PNG 支持
- `width_profile.py` 的逐段均宽（`w ≈ 2·EDT−0.5`）与 `stroke_topology.py` 的
  stroke 组装，属于本仓库为 GNN 接口新增的侧车，不在论文范围内

---

## 4. 复现与回归

```powershell
cd E:\ai\boundary_topology_python
python bench_speed.py            # 三种 output_mode 的耗时（crop800 / full）
python bench_fingerprint.py out.json   # 输出指纹（段/笔画折线 SHA、宽度、计数）
python -m pytest -q              # 需要 pytest；在无 pytest 环境用上面的脚本回归
```

`E:\ai\aiColorize\Debug\btp_perf\` 下另有本次核查用的脚本：`check_fix.py`（计数）、
`check_cost.py`（completeness 开关开销 + 重建中心线渲染）、`verify.py`（label_map
逐字节对比）。
