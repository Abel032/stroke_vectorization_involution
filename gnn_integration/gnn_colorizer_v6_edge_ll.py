"""
 GNN 动漫线稿上色系统 — v6 (Line-Centric L-L GNN)
 Anime Line Art Colorization with Line-Centric Line-Line GNN (v6)
============================================================================

【当前实现概览（2026-11-13 基底；2026-11-14 指标、骨架拓扑及图提取修订）】

  1. 图结构与主干
     - Line 为细图节点；Line-Line 边关联共享 Keypoint / 几何连续等关系，
       Region 是独立上下文节点并通过 Line→Region→Line 支路参与消息融合。
     - GNN_LAYERS 层 Line 主路使用 GATv2 风格的消息传递和残差连接；
       Corner Stroke、Continuous Stroke 组内聚合、Region 分支与主路并行加性融合。
     - DropEdge / Relation DropPath / NBA 按顶部开关控制；NBA 并非始终启用。

  2. 输入特征
     - Line 5D = relative_arc_length / straightness / FRT / rel_arc_len / boundary_share。
       线色为离散 Embedding；独立 line_pos/region_pos 提供相对位置上下文。
     - L-L Edge 2D = |tan_half| / len_ratio，关系多热编码另行传入。
     - Region 线色几何固定 MAX_LINE_COLORS×3D（当前 24D），
       逐像素聚合颜色边界长度与方向；另有 region_pos 2D 和 region_shape_feat 2D。
     - 跨对象比较的位置/长度参考 content_scale；Region 边界长度参考 sqrt(area)。

  3. 分类与推理
     - Triple Head：无序两侧颜色对 + 线条颜色的联合分类。
     - Side Head：共享权重分别预测几何 +n/-n 侧颜色；独立 Direction Head 已删除。
     - Triple×Side 联合后验提供 MRF 方向相关的 Unary 证据；
       MRF Pairwise 继续使用 Triple 无序色对证据；max-product BP 求 Region 标签。
     - 最终结果可包含原始线稿及色线回填；颜色调色板索引 0 为透明背景。

  4. 骨架与图提取（v66：无剪枝、触边保护、微小闭区就近填墨）
     - 透明/白底线稿统一为 alpha 墨线；不触边的 <=4px 封闭透明区
       从最近的原始墨线像素复制 RGBA 后删除；Region GT 平铺使用逐区域分组众数。
     - 路径沿骨架像素图连续追踪；分叉回溯不重复计入物理弧长。
     - 关键点与路径关系不再使用可能互相覆盖的 5x5 标签图；
       8 连通 Line 接触边使用 NumPy 分片批量构建。
     - 抗锯齿 alpha 不再单独占用线色嵌入类别，颜色按 RGB 身份建线色表。
     - 多像素交点簇共享亚像素几何锚点；交点各臂进行单侧 TLS 切向拟合，
       Edge 使用真实接续弯角，Region 两侧用多点法向投票。
     - 骨架化比较预处理图与骨架图的逐 Region 分区映射，并逐边恢复贴边主干连接；
       图缓存核对输入文件元数据与调色板，旧缓存必须重建。

  5. 评价与可视化（两套 Region 指标不可互换）
     - Graph RegionAcc：GNN 骨架图的 Region 节点，全部有效 Region 为分母；
       同时记录仅投票准确率和投票覆盖率。
     - 成图 RegionAcc：gnn_common 根据原线稿掩码 4 连通划区，比较两图 RGBA 众数。
     - PixelAcc / mIoU：原线稿墨线像素被排除，按调色板内 GT 像素统计；
       另有全部非线条像素严格 RGBA PixelAcc 及覆盖率，避免未知 GT 静默误判。
     - 终端、日志 JSON 与 HTML 明确标出指标来源；
       HTML 的线边际诊断均值不是 MRF 最终信念分布。

【代码导航】
  特征/几何 ................. line_local_frame / path_arc_length / compute_line_features /
                              compute_line_frt / edge_tan_half / _line_pca_direction
  图提取 .................... extract_ll_graph (Line/Keypoint/Region/L-L 边/两套分组)
    ├─ 路径排序 ............. order_paths_with_sknw (A2, 可选) / gnn_common.trace_skeleton_component
    ├─ 交点续接匹配 ......... match_continuations_mwm (A1, 可选) / 内置单对 legacy
    └─ 分组粗化 ............. _compute_group_coarse (scipy 连通分量) / _compute_stroke_coarse
  数据集 .................... LineartLLDataset / ll_collate_fn / build_dataset_bundle
  模型与分类 ................ LineGNNLayer / LineColorizerBackbone / ClassifierHead /
                              LineColorizerGNN / scatter_softmax
  解码与置信度 .............. predict_regions_from_logits / solve_region_colors_mrf /
                              _conf_from_probs / _joint_mrf_map_refine
  损失 ...................... _triple_loss / _joint_region_nll / _region_support_rank_loss /
                              _side_rank_svm / compute_loss / combine_loss_terms
  训练与验证 ................ train_ll_gnn / evaluate_model / train_multitask_pretrain
  成图评价 .................. evaluate_pixel_level / gnn_common.evaluate_coloring_images
  推理与 HTML ............... colorize_image_ll / visualize_ll_graph
  checkpoint 安全 ........... build_structure_meta / load_checkpoint_guarded /
                              save_checkpoint_atomic / load_checkpoint
  主入口 .................... main / _run_multitask_pipeline

【实验开关（全部为环境变量；不设置即 S0 / 官方基线行为）】
  名称                     默认        作用                                  采纳状态
  STROKE_BRANCH            both        笔画支路: both / corner_only / cont_only   既有
  STROKE_STAT_DIM          0           每统计量先压缩到 k 维 (0=原 193 维拼接)     S1-A 已 FAIL
  STROKE_BRANCH_MODE       baseline    支路末端: baseline / deep (感知机+MLP)      db1–db12 实验
  STROKE_LOWRANK_RANK      0           支路投影 SVD 低秩化 (A0 计划, 已搁置)       未采纳
  STROKE_MLP_HIDDEN        32          deep 模式拼接后 MLP 宽度
  STROKE_MLP_LAYERS        2           deep 模式带激活隐层数 (>=2)
  STROKE_STAT_HIDDEN       0           deep 模式每统计量的两层投影中间宽度
  STROKE_SHARE_MLP         0           Corner/Continuous 共用拼接后 MLP           db7/db12 实验
  STROKE_BRANCH_RESIDUAL   0           支路输出 + x 恒等短路                      db6 实验
  STROKE_BRANCH_DROPOUT    0           支路 MLP 内 dropout (训练期)                db7/db12 实验
  REGION_RANK_WEIGHT       0           Region support 排序损失权重 (db12 用 1.0)   db12 新基底
  REGION_RANK_MODE         logistic    auc(无超参) / logistic / hinge             db12 用 auc
  REGION_RANK_MARGIN       0.1         仅 hinge 模式使用
  LINE_PATH_BACKEND        legacy      legacy / sknw (A2: sknw 提供有序像素)      等价, 可选
  CONT_PAIR_BACKEND        legacy      legacy / mwm (A1: NetworkX 最大权匹配)     A1 已 FAIL
  BRANCH_SCALE_INIT        0.05        支路融合标量初值
  BRANCH_GATE_MODE         global      global / node (逐 Node 逐支路门控)

  说明: 非默认开关只用于实验复现; 正式基线固定为"全部默认"(S0) 或 db12 新基底
  (deep + share_mlp + dropout + rank=1.0/auc), 见 docs/A2_LIBRARY_REFACTOR.md 与
  docs/DB12_RANK_LOSS.md。任何开关都会进入图缓存 key 与 checkpoint 结构指纹。

============================================================================
"""

import os
import hashlib
import glob
import time
import warnings
import math
import json
import pickle
import random
import argparse
import sys
import re
from collections import Counter, deque
from typing import Dict, Any, Optional
from tqdm import tqdm

import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import label as sk_label
# matplotlib 延迟到 _plot_training_curves 内部再导入, 缩短启动导入时间

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from gnn_common import (
    set_seed, scatter_sum, scatter_mean, scatter_max,
    process_alpha,
    skeletonize_lineart, detect_keypoints,
    build_global_palettes,
    _VIS_HTML_TEMPLATE, _to_native, _mode_rgb_packed,
    prepare_lineart_for_graph, trace_skeleton_component, skeleton_contact_pairs,
    unify_target_by_regions, estimate_keypoint_branch_tangent,
    infer_line_plus_region_a, junction_keypoint_anchors,
    evaluate_coloring_images, region_accuracy_counts, pixel_scores_from_confusion,
)

# ==================== 0. 全局常量 ====================
INPUT_ROOT = "input_B"

# --- 训练超参 ---
EPOCHS = 800
LEARNING_RATE = 0.005   # 2图小样本过拟合: 0.01 太快记忆化, 减半减缓 (SWA 平均落在更稳区)
PATIENCE = 30
WEIGHT_DECAY = 0.03         # 数据极少(2图)→加强 L2 正则抑制过拟合 (原 0.03)
EARLY_STOP_SLOPE_WINDOW = 25   # 早停斜率观察窗口 (epoch 数)
EARLY_STOP_SLOPE_TOL = 1e-4    # 收敛阈值: 窗口内 total_loss 每 epoch 平均下降低于此值即停 (0=关闭)
SWA_WINDOW = 15       # SWA 参数平均的滚动窗口大小: deque 保留最近该数量的 epoch 快照,
                            # 训练结束对窗口内快照做等权平均; train_loss 最低点被挤出窗口时
                            # 单独补回, 保证"最低点附近"参与平均 (早停 patience == SWA_WINDOW)。
LANDSCAPE_NUM_DIRECTIONS = 3   # 损失景观方向数 (多方向取均值降低随机切片方差)

# --- 投票尺度消融开关 (默认 = 基线行为; 仅供消融实验) ---
#   UNARY_MASS_MODE: "clamp1" = 除以 clamp(mass, min=1.0) (基线) | "true" = 除以真实 mass
#   PAIRWISE_AGG   : "sum"    = 边内多条线直接求和 (基线)          | "inv_sqrt" = 1/sqrt(n_rs)
UNARY_MASS_MODE = os.environ.get("UNARY_MASS_MODE", "clamp1").strip().lower()
PAIRWISE_AGG = os.environ.get("PAIRWISE_AGG", "inv_sqrt").strip().lower()
# ==================== 投票/求解开关总表 ====================
# 全部可用环境变量控制; 不带前缀的默认值即"已验证配置"(见 DESIGN_NOTES §18-§25 的 4-seed 结果)。
#   UNARY_MODE            synergy(默认) | triple | side | adaptive | factorized | agree
#                         证据组合方式。G 类数据上 synergy 最优; J 类可试 adaptive。
#   UNARY_MASS_MODE       clamp1(默认) | true
#                         unary 分母是否 clamp 到 1。clamp1 等价"均匀先验补到 1 条线的证据量",
#                         改成 true 在 G/J 上都更差。
#   PAIRWISE_ENABLE       1(默认) | 0
#                         0 = 只用 unary。J 类(线密度 ~5/region)关掉更好, G 类会掉 RA。
#   PAIRWISE_AGG          inv_sqrt(默认) | sum | inv_n
#                         边内多条线的势聚合。inv_sqrt 是唯一"两数据集都不掉"的改动。
#   PAIRWISE_POT          triple(默认) | joint | hybrid
#                         边势来源。joint/hybrid 只在 J 上为正, G 上显著变差。
#   PAIRWISE_GATE         none(默认) | same_color | margin
#                         对"两端 unary 同色"的边做闸门。实测对 G 无效(阈值卡在 4.44 < 5.76)。
#   PAIRWISE_UNSEEN_PENALTY  10.0(默认)
#                         训练表外色对的有限退让。4 数据集消融显示 1.0 明显更差。
#   MRF_DUMP_PATH         (空) 设置后逐次转储 MRF 内部量, 供 tools/analyze_*.py 使用。
# ==========================================================
PAIRWISE_POT = os.environ.get("PAIRWISE_POT", "triple").strip().lower()
# PAIRWISE_ENABLE=0: 完全关闭边势 (等价 λ=0, 只保留 unary). 用于"仅 unary"配置。
PAIRWISE_ENABLE = os.environ.get("PAIRWISE_ENABLE", "1").strip() not in ("0", "false", "off")
# 真实推理允许训练集没见过的 Region 色对：不能把缺失的 Triple 标签当作“不可能”。
# 新色对的 Pairwise 代价是有限可控的，并仍可由 Region Unary 证据胜出；不是硬屏蔽。
#
# 默认值 10.0 = 旧行为的裁剪下界 (log((0+eps)/maxQ) 本来就会被 clip 到 -MRF_SOFT_M=-10)。
# 依据 (DESIGN_NOTES §24/§25, 4 seed 42-45 配对比较, 只改这一个值):
#   pen1 -> pen10:  G RA +0.60pp / mIoU +3.23pp;  B RA +2.62pp (t=2.48) / mIoU +5.69pp;
#                   V RA +0.70pp (t=2.97) / PixelAcc +6.68pp / mIoU +6.50pp;  J 逐 seed 完全相同
#                   (J 调色板仅 5 色, 无未见色对)。即"放宽惩罚以允许新组合"在 4/4 数据集上无正收益。
# 若某数据集确实需要鼓励未见组合, 可降到 3.0 (代价: B RA -2.6pp / G RA -0.6pp)。
PAIRWISE_UNSEEN_PENALTY = float(os.environ.get("PAIRWISE_UNSEEN_PENALTY", "10.0"))
if PAIRWISE_UNSEEN_PENALTY < 0:
    raise ValueError('PAIRWISE_UNSEEN_PENALTY 应为非负数')
# PAIRWISE_GATE: 对"两端 unary 首选色相同"的边做闸门, 不强行拆开它们
#   "none"(默认) | "same_color"(该 (c,c) 惩罚置 0) | "margin"(仅当双方 unary 边际都超过该惩罚时才置 0)
PAIRWISE_GATE = os.environ.get("PAIRWISE_GATE", "none").strip().lower()
# UNARY_MODE: "synergy"(默认, Triple×Side) | "triple"(仅 triple 边际) | "side"(仅 side 似然)
UNARY_MODE = os.environ.get("UNARY_MODE", "synergy").strip().lower()

MRF_MISMATCH_WEIGHT = 1.0         # pairwise 势强度 λ: ψ = λ·clip(log(Q/maxQ), -M, 0)。
                                  #   注意: 实际势**不含** w_i / conf_i 乘子 (该注释为旧版残留);
                                  #   边内多条线按 1/sqrt(n_rs) 归一化后再求和。
MRF_SOFT_PAIR = True             # E1: pairwise 用 log pair_probs 软势 (False=旧 argmax mismatch 硬惩罚)
MRF_SOFT_M = 10.0                # 软势 log 比率的裁剪下界 (避免低概率 pair 产生失控负值)

# --- Cosine 分类头 (triple / side 同构) ---
#   logits = s · (x̂ · ŵᵀ): 特征与类权重都做 L2 归一化, s 为**可学**温度。
#   基准: s 初值 10, 损失 = CE + label smoothing 0.1。
#   注意: s 可学 + CE 会把概率越推越尖 (实测单线 top1 ≈ 0.9997, 概率比 ~1e4)。
#   这一点被专门验证过 (§14): 主动抑制锐化 (改成 margin 目标 / 固定 s=1 / 加大 LS)
#   全部掉点, 说明"锐化"是投票链路的承重结构, 不是可剥离的副产物, 因此不压。
COS_SCALE_INIT = 10.0            # cosine 头温度初值 (可学参数; 基准值 10)

# 以下两个常量只服务于可选的 margin 实验模式, 默认 (ce) 不启用, 结论见 §14:
CS_TOPK = 5                      # 只约束最难的 top-5 竞争者 (= hard negative mining);
                                 #   logit-margin 口径的阈值默认取 ln(C), 见 _cs_topk_hinge
CS_MASS_DELTA = 0.2              # 质量 margin: p[y] − Σp[其它] ≥ 0.2, 即 p[y] ≥ 0.6

# --- Region 颜色直接监督 (核心: 训练目标对齐真正评价指标) ---
# 颜色本质是 region 的属性, 但主损失是 line-level triple 分类且按 line 平均:
#   大区域被很多线包围 → 在 line 损失里权重极高; 小区域(爪子/鼻子)只有 1~2 条边界线
#   → 几乎不被训练 → 小区域系统性被周围颜色吞掉。
# 此损失把每条线的无序色对边际聚合到 region 级颜色分布, 直接对 region 真值做 NLL,
# 且每个 region 等权 → 小区域与大区域同等重要。
# Label smoothing: 抑制"过自信地判错"。CE 会把预测推向 0/1, 在 2 张训练图下
#   置信度会顶到 1.0 (实测 consistency 中位数 = 1.0), 而置信度在本项目里直接当投票权重,
#   于是"自信的错线"票最重。加 smoothing 从训练源头把最高置信度压下来。
#   0 = 关闭。
LABEL_SMOOTHING = 0.1

# --- 支路融合标量初值 (改进清单第 2 项) ---
#   三条粗支路 (coarse / cont_coarse / region) 的软注入强度初值, 原值 0.05。
#   主消息支路隐式权重为 1, 支路初值过小可能在训练初期被压制/收敛到近零。
#   可用环境变量覆盖, 便于不改代码地跑同一项的不同取值:
#       $env:BRANCH_SCALE_INIT="0.3"; python _run1.py br3 42,43,44,45
BRANCH_SCALE_INIT = float(os.environ.get("BRANCH_SCALE_INIT", "0.05"))

# --- 支路增益的实现方式 (清单第 3 项) ---
#   "global": 每支路一个全局可学标量 (基准, 默认) —— 与基准提交逐位等价。
#   "node"  : 逐 Node 逐支路自适应门控 g_{i,b} = amp_b * sigmoid(W_b [h_i; m_{i,b}] + c_b)。
#   4-seed 实测 (seed 42-45, 均为 window=20 口径): B 0.8405 vs 0.8191 (+2.1pp, t≈1.2);
#   V 0.5246 vs 0.5348 (-1.0pp, t≈-1.4)。方向相反且均不显著, 故默认仍用 global。
#   (window=25 的基准是 B 0.8214 / V 0.5594, 见 §15.1 —— 两套口径不可直接相减。)
#   门控实现保留以便以后在别的结构上复测。切换: $env:BRANCH_GATE_MODE="node"
BRANCH_GATE_MODE = os.environ.get("BRANCH_GATE_MODE", "global").strip().lower()
# REGION_POOL_MODE: Region 桥接支路的 line->region 池化方式。
#   "incidence"(默认): 每条线对每个**不同**的接触 region 只贡献一次, 统一求均值 -> 与 A/B 编号无关。
#   "split"(旧行为, 保留复现): scatter_mean(A侧) + scatter_mean(B侧)。两侧各占 50% 权重,
#     内部线 (A==B) 计两次, 结果依赖 A/B 编号 (而 A/B 只是按面积排序的图提取约定), 幅度约 2.7x。
#   实测 (4 线反例): split 给 107.5, incidence 给 39.75 (统一均值真值), 且只交换一条线的 A/B
#   编号就让 split 从 107.5 变 138.75 (+29%), incidence 不变。
#   4-seed 效果 (G/H/I, 成图 RegionAcc): G -0.91pp (t=-0.90) / H +0.52pp (t=0.46) / I +0.62pp (t=0.35),
#   RA 全部落在噪声内; 但 H 的 PixelAcc +1.59pp (t=1.96)、mIoU +3.13pp (t=3.29) 显著为正。
#   采纳理由: A/B 编号不应携带语义, 故取结构正确的版本, 接受 G 在噪声内的单点下降。
REGION_POOL_MODE = os.environ.get("REGION_POOL_MODE", "incidence").strip().lower()
# 笔画池化统计量固定为 mean+max+std+log_size (= 4H+1 维)。曾试过去掉 std ("no_std"),
#   在 G 上 RA +2.42pp 但 PixelAcc -17pp (大区域系统性判错)、I 上三项全负, 收益不干净, 已删除该选项。
# STROKE_BRANCH: 启用哪些笔画池化支路。"both"(默认) | "corner_only" | "cont_only"
#   依据: 单成员组 (Corner 在 B01 上 355/382) 时池化退化为逐线变换, 支路可能冗余。
STROKE_BRANCH = os.environ.get("STROKE_BRANCH", "both").strip().lower()
# STROKE_STAT_DIM: >0 时, 每个统计量先各自线性压缩到该维度, 再与 x / log_size 拼接后综合。
#   16 (默认, 已采纳) = 每个统计量 48 -> 16 (+GELU), 再 Linear(97,48); 每分支每层 9,312 -> 6,960 参数。
#   0 = 旧结构: 三个统计量原样 193 维拼进一个 Linear(193,48)。
#   好处: 参数量下降 (5 层合计 -22,560, 全模型 -8.3%) 且多一层非线性; 三个统计量都保留。
#   4-seed (G/H/I): RA +1.66 / +0.05 / +2.03pp, PixelAcc +6.10 / -2.27 / +3.97pp, mIoU -2.76 / -2.72 / +3.18pp。
#   对照: 直接删掉 std ("no_std") 在 G 上 RA +2.42pp 但 PixelAcc -17pp (大区域系统性判错), 已否决并删除该选项。
STROKE_STAT_DIM = int(os.environ.get("STROKE_STAT_DIM", "0"))   # B0 基线=0; 采纳候选由环境变量指定
assert STROKE_STAT_DIM >= 0, STROKE_STAT_DIM
assert STROKE_BRANCH in ("both", "corner_only", "cont_only"), STROKE_BRANCH

# STROKE_LOWRANK_RANK (S1-LR): >0 时把每条 Stroke 支路的末端投影 Linear(193,48) 换成
#   Sequential(Linear(193,r,bias=False), Linear(r,48,bias=True), GELU)。
#   池化公式 / 统计量 / GELU 位置 / branch_norm / 缩放 / 残差 全部不变; 只做权重低秩近似。
#   默认 0 = 原结构 (B0), 因此不设置该环境变量时行为与 S0 完全一致。
#   权重由 tools/s1lr0_convert_svd.py 从 S0 checkpoint 做截断 SVD 生成 (不训练)。
STROKE_LOWRANK_RANK = int(os.environ.get("STROKE_LOWRANK_RANK", "0") or 0)
assert STROKE_LOWRANK_RANK >= 0, STROKE_LOWRANK_RANK

# STROKE_BRANCH_MODE = "deep" (2026-11-15 新架构试验):
#   支路先对 mean/max/std 各做 **一层感知机 (Linear + ReLU)** 并压到 STROKE_STAT_DIM 维,
#   与 x(48) / log_size(1) 拼接后, 再过 STROKE_MLP_LAYERS (>=2) 个带激活的隐层
#   (宽度 STROKE_MLP_HIDDEN), 最后 Linear(h,48) 投回主路维度, 再走原有 branch_norm /
#   缩放 / 残差相加。池化公式、组内统计、GELU 之外的其它结构、MRF、置信度均不变。
#   默认 "baseline" = S0 原结构 (单层 Linear(193,48)+GELU), 因此不设置时行为完全不变。
STROKE_BRANCH_MODE = os.environ.get("STROKE_BRANCH_MODE", "baseline").strip().lower()
assert STROKE_BRANCH_MODE in ("baseline", "deep"), STROKE_BRANCH_MODE
STROKE_MLP_HIDDEN = int(os.environ.get("STROKE_MLP_HIDDEN", "32") or 32)
STROKE_MLP_LAYERS = int(os.environ.get("STROKE_MLP_LAYERS", "2") or 2)
# STROKE_STAT_HIDDEN>0 时, 每个统计量的投影改成两层感知机:
#   Linear(48, STROKE_STAT_HIDDEN)+ReLU -> Linear(STROKE_STAT_HIDDEN, STROKE_STAT_DIM)+ReLU
#   (默认 0 = 单层 48->STROKE_STAT_DIM, 即 db1 的形态)
STROKE_STAT_HIDDEN = int(os.environ.get("STROKE_STAT_HIDDEN", "0") or 0)
# STROKE_BRANCH_RESIDUAL=1 (仅 deep 模式): 支路输出加上节点自身 x 的恒等短路,
#   让深支路只需学"相对主路的修正量", 降低小数据集(E2 张训练图)上的方差; 默认 0 关闭。
STROKE_BRANCH_RESIDUAL = int(os.environ.get("STROKE_BRANCH_RESIDUAL", "0") or 0)
# STROKE_SHARE_MLP=1 (仅 deep 模式): Corner / Continuous 两支路**共用**拼接后的 MLP
#   (统计量投影各自独立)。参数近乎减半, 同时等效于对支路做权重共享正则, 用于小样本数据集。
STROKE_SHARE_MLP = int(os.environ.get("STROKE_SHARE_MLP", "0") or 0)
# STROKE_BRANCH_DROPOUT>0: 支路 MLP 每个 ReLU 之后加 dropout (仅训练期生效)。
STROKE_BRANCH_DROPOUT = float(os.environ.get("STROKE_BRANCH_DROPOUT", "0") or 0.0)
if STROKE_BRANCH_MODE == "deep":
    assert STROKE_MLP_LAYERS >= 2, STROKE_MLP_LAYERS
    assert STROKE_MLP_HIDDEN > 0, STROKE_MLP_HIDDEN
    assert STROKE_STAT_HIDDEN >= 0, STROKE_STAT_HIDDEN

TRIPLE_LOSS_WEIGHT = 1.0          # 主 Triple CE 权重; 可消融, 默认保持原行为
# triple 主损失类型: "rank_svm" = Rank-SVM 多标签排序损失(hinge), "ce" = 交叉熵(原行为)。
#   排序损失要求真色对分数比每个负色对高出 RANK_SVM_MARGIN, 对"偏向同一高分色对"的平凡解压制更直接。
TRIPLE_LOSS_TYPE = "ce"           # triple 主损失: "ce" / "cs_mass" / "cs_topk" / "rank_svm" / "ce_plus_rank"
RANK_SVM_AUX_WEIGHT = 0.5         # ce_plus_rank 模式下排序项的权重。rank 内部已按负类数取平均
                                  #   (量级 ≈ margin), 调用处**不要**再除负类数 —— 早期版本重复
                                  #   除了一次 n_neg, 使该项被压 1/100 而实际退化为纯 CE (§16)。
RANK_SVM_MARGIN = 2.0             # hinge 阈值。logits 由余弦 × 温度10 得到 (范围 ±10), 取 2 ≈ 20% 量程
# 负类频率权重: 让常见类承担 margin 压力, 稀有色对几乎不参与排序竞争。
#   诊断依据 (DESIGN_NOTES 第 8 节): 均匀权重的排序损失会把大量稀有色对一起下压,
#   伤 input_B 的长尾 (101 类、每类样本 <1)。power=0 即原均匀行为。
RANK_SVM_FREQ_POWER = 0.0
RANK_SVM_FREQ_CLAMP = 5.0

# --- 核方法: triple 头从余弦(线性) 改为 RBF 核, 原型 = 原 cls_weight (NT, H) ---
#   s_t = -S·γ·‖feat - μ_t‖²  (log-RBF / 高斯判别; 直接 exp 会让 logits 量程过窄,
#   在置信度加权投票里被降权, 见 side 头的教训)
#   实测 NT 仅 101/85 (非上界 1053), 因此 N×NT×H ≈ 4.4M, 无需分块。
TRIPLE_CLASSIFIER = "cosine"
TRIPLE_RBF_SCALE = 40.0

# --- side 头的 Rank-SVM (多标签排序) ---
#   side 只有 C=26/40 个类, 长尾压力远小于 triple (B 有 101 个闭集类、每类样本 <1),
#   是排序损失更容易生效的地方; 而 triple 保持 CE 以保护 B 的长尾色对。
SIDE_LOSS_TYPE = "ce"             # "ce" / "cs_mass" / "cs_topk" / "rank_svm" / "ce_plus_rank"
SIDE_RANK_WEIGHT = 0.5
SIDE_RANK_MARGIN = 2.0

# --- 核方法: side 头的分类器从余弦(线性) 改为 RBF 核 ---
#   s_c = scale * exp(-γ ‖feat - μ_c‖²),  μ_c 为每个颜色的可学习原型 (side_weight),
#   γ = exp(log_γ) / S 为**可学习**宽度。这是标准的 RBF 核分类器 (kernel SVM 的 RBF 形式)。
#   注意: 若先把 feat / μ 归一化, RBF 会退化成余弦的单调函数 (排序完全相同), 因此这里用**未归一化**特征。
SIDE_CLASSIFIER = "cosine"        # "cosine" / "rbf"  (实测 rbf 与 log-rbf 均不优于 cosine)
SIDE_RBF_SCALE = 40.0             # log-RBF 的尺度: s_c = -S·γ·d², 使 logits 动态范围与余弦头相当
REGION_LOSS_WEIGHT = 1.0          # region 颜色 NLL 权重。注意它是**单个联合损失**
                                  #   (_joint_region_nll): 用乘性协同的联合后验, 梯度同时
                                  #   喂给 triple 与 side 两个头 (不再是两个独立损失)。

SIDE_LOSS_WEIGHT = 1.0            # Side-head 两侧颜色 CE 损失权重 (P1: 解耦 triple 的分类目标)

# --- Region 级 support 排序损失 (方案 A): 直接优化"高支持却判错" ---
#   同一张图内, 判错 region 的 support 必须低于判对 region (hinge, 只罚越界部分):
#       L_rank = mean_{wrong j, correct i} relu(support_j - support_i + margin)
#   support 与推理侧 decision_support 同口径: softmax(unary)[pred],
#   其中 unary 是该 region 的 conf 加权平均对数证据 (与 _joint_region_nll 共用)。
#   REGION_RANK_WEIGHT=0 时该项完全不参与 (默认, 与 S0 训练目标一致)。
REGION_RANK_WEIGHT = float(os.environ.get("REGION_RANK_WEIGHT", "0") or 0.0)
REGION_RANK_MARGIN = float(os.environ.get("REGION_RANK_MARGIN", "0.1") or 0.1)
# REGION_RANK_MODE: "logistic" (默认, 无超参: softplus(gap)/ln2, 并列代价=1, 权重可恒取 1)
#                   "hinge"    (旧版: relu(gap + margin), 需要 REGION_RANK_MARGIN)
REGION_RANK_MODE = os.environ.get("REGION_RANK_MODE", "logistic").strip().lower()
# 乘性协同的似然软化指数 β: side 似然取 q^β。β=1 不软化; β<1 降低"side 自信地给 0
#   掐死 triple 正确证据"的风险 (side 本身较弱, 软化有理由)。
SYNERGY_BETA = 1.0

SIDE_INFER_WEIGHT = 1.0        # 【开关】是否让 side 参与"乘性协同"的联合后验。
                               #   >0 = 启用 (side 作为 q_a(c0)·q_b(c1) 的似然因子乘进联合后验);
                               #   =0 = 关闭, 退化为 triple 的无序色对边际。
                               #   注意: 它**不再是**混合比例 —— 两路的相对强弱由概率乘积与
                               #   SYNERGY_BETA 决定, 没有人为的 1:1 之类的配比。
SIDE_SUPPORT_MODE = "saturating"  # 结构支持度权重函数: saturating | linear | log
SIDE_OTHER_MODE = "soft"      # 非主导侧处理: reward=两侧都奖励 | zero=非主导侧不奖励 | entropy=非主导侧鼓励不预测 | soft=双侧软加权(均值1)
# 注意力/回溯开关:
GAT_ABLATE = False   # True: softmax 前把注意力 logits 清零 → 均匀注意力(均值聚合)。
                     #   7-seed 实测关 GAT 无稳定收益 (均值 0.8109 vs base 0.8143), 故默认 False 保留 GAT。
NBA_ABLATE = False    # 默认 True: 关闭 NBA (上层最强影响邻居回溯屏蔽)。
                     #   GAT 学习注意力 + Dropout 已能自行调控邻居影响力, NBA 与之功能重叠, 关掉更简。

#   sigmoid 本身内置低置信回退, 不再需要单独的 conf 门控 (原 DIR_CONF_* 已删)。


# 主损失: 普通 one-hot CE (line-level triple 分类), 由 _triple_ce 计算, 训练与验证目标统一。

# 网格分辨率: 41 很精细但约 3.8× 更慢; 经验上 21 已足够平滑 (可视化足够), 可用 --landscape-steps 覆盖。
LANDSCAPE_STEPS = 21
LANDSCAPE_RANGE = 0.02
LANDSCAPE_SEED = 20260913

# --- 模型维度 ---
HIDDEN_DIM = 48          # GNN 隐藏层维度 (Line & Region)。曾扩到 64 无效果(数据瓶颈而非容量瓶颈),
                         # 回退 48 降参加速。head 数 3 配套 (head_dim = 48//3 = 16)。
LINE_ENC_DIM = 24        # Line 编码器中间维度
EDGE_ENC_DIM = 4        # L-L 边编码器中间维度
COLOR_EMB_DIM = 16       # 线色方向 + 区域线色几何 的共享嵌入维度。
                         # 原 8: region_color_compress(24→8) + cls_side_encoder(8→32) 串联构成
                         # rank≤8 的低秩瓶颈 (24 维输入先压到 8 再扩回 32)。→16 后 rank≤16 解除瓶颈;
                         # color_embed(8×16)/color_proj(16→48)/cls_side_encoder(16→32) 随之放宽。
MAX_LINE_COLORS = 8      # 线色嵌入表统一尺寸 (跨数据集共享 backbone 时, color_embed 与
                         # region_color_compress 的维度必须数据集无关, 故固定为常量而非 len(line_palette)。
                         # 各数据集的线色调色板在 main 内截断到该值, 保证 line_cidx < MAX_LINE_COLORS。

GNN_LAYERS = 5           # GNN 层数
# 方向连续笔画分组阈值 (见 _compute_stroke_coarse): 两线在交点附近局部切向的**有符号**余弦相似度。
#   续接判据 = 有符号点积 <= -此值 (反平行, 夹角 >= arccos(阈值))。
#   来自 "连续的笔画判断" (Rajan & Hammond 2008) 的 PCA 主方向点积 + stroke merging。
#   0.95 对应夹角 >= 161.8°: 连续笔画穿交点时两臂反平行(≈180°), 用有符号点积(而非|cos|)
#   可区分"反平行续接"与"分叉近同向"——abs() 会把小角度分叉也误并。
CONT_STROKE_THRESHOLD = 0.95

# --- 几何 backend 开关 (A1/A2 指令, 默认全部 legacy = A0 原行为) ---
# CONT_PAIR_BACKEND:
#   "legacy" = 每个关键点只取最反平行的一对 (原实现, A0 默认)
#   "mwm"    = 每个交点做**非完美最大权匹配** (NetworkX), 允许同一交点出现多对,
#              每条半分支(用 Line id 表示)最多参与一次, 允许部分/全部不匹配。
# LINE_PATH_BACKEND: "legacy" (手写 walk) | "sknw" (A2: sknw 提供有序像素, 未覆盖块回退 legacy)
CONT_PAIR_BACKEND = os.environ.get("CONT_PAIR_BACKEND", "legacy").strip().lower()
LINE_PATH_BACKEND = os.environ.get("LINE_PATH_BACKEND", "legacy").strip().lower()
assert CONT_PAIR_BACKEND in ("legacy", "mwm"), CONT_PAIR_BACKEND
assert LINE_PATH_BACKEND in ("legacy", "sknw"), LINE_PATH_BACKEND

# 异构图关键点特征: 关键点 crossing number (拓扑量, 天然 SE(2) 不变)。
#   cn=1 端点(自由端) | cn=2 笔画折角(corner) | cn>=3 交叉点(junction) | cn=0 孤立注入点。
#   保留原始 cn (而非压缩成 one-hot), 信息完整; 接入 GNN 时用 Embedding 即可。

# ============ Dropout 图景 (2026-09-06 定稿) ============
# 核心原则: 主干(线几何+颜色嵌入)是需保护的强信号, GNN 增量(residual)才是正则对象。
#   实测: 主干 dropout 有害(BASE_DROPOUT 0.1 微降/0.2 全崩, 已回退); 出口/神经元 dropout 打在增量上才有效。
#   这印证: 调 MSG_DROPOUT 能同时移动 PairAcc 与 ValLoss → 泛化瓶颈集中在 GNN 增量这 4 层的"窄容量窗口",
#   是 2 图小样本 + 4 层 GNN 容量过配的典型症状。正则落点务必在 residual, 勿动主干。
#
# 六类 dropout 全景 (按作用位置分层):
#   1. MSG_DROPOUT  (0.3)   — GNN 4 层内消息残差分支(out_proj 后)的神经元 dropout, 固定值, 各层一致。
#   2. ATTN_DROPOUT (0.1)   — GNN 注意力 dropout, 作用在 softmax 后的注意力系数 α 上(标准 GAT 做法), 固定值。
#   3. OUTPUT_DROPOUT(0.3)  — GNN 出口 residual 分支 dropout(gnn_dropout), 只打 GNN 增量、不碰主干 h_l_base。
#   4. DROPEDGE_RATE (0.35)  — DropEdge 结构级正则, 每次前向采样一个共享边 mask 复用全部层(删边≠删神经元), 防"背图结构"。实测 +5~7pp。
#   5. BASE_DROPOUT  (0.0)  — 主干 dropout, 已实测有害并回退为 0, 永不复用(主干=强信号, 不能关)。

ATTN_DROPOUT = 0.1         # 注意力 dropout: 作用在 softmax 归一化后的注意力系数 α 上 (标准 GAT 做法),
                            #   被置 0 的边其贡献真正归零 → 稀疏化聚焦边, 防注意力过拟合。固定值, 各层一致。
MSG_DROPOUT = 0.25          # 消息 dropout: 作用在 out_proj(aggr) 的残差增量上 (仅打 GNN 增量、不碰主干)。
                            #   与 ATTN_DROPOUT 分离 (位置/语义不同); 神经元 dropout 固定值即可, 无需随层递增。
OUTPUT_DROPOUT = 0.3     # GNN 出口残差分支的 dropout (作用于 residual=GNN增量, 非主干)。
DROPEDGE_RATE = 0.35      # DropEdge: 每次前向删 L-L 边的比例 (0=关闭)。防"背图结构"过拟合。
                         #   mask 每次前向采样一次、全部 GNN 层共享 (原始论文口径, 见 forward 循环)。
                         #   小样本(≤5图)下建议 0.3~0.5; 共享 mask 删 A/B 侧边并同步删 attr。
DROPEDGE_GROUP = True    # DropEdge 改为丢"粗化组之间的边"(inter-group), 组内边恒保留; False=旧生成森林随机删
                         #   0.3 实测有效(+5~7pp, 见 2026-09-06 日志), 作为默认开启。
                         #   注意: 此值是"删边"(DropEdge), 与上方 ATTN/MSG_DROPOUT(删神经元) 语义不同。
MC_DROPOUT_SAMPLES = 0   # MC Dropout 推理采样次数 (>1 = 开启: 推理期保持 dropout/DropEdge 开启, K 次前向平均 logits)。
                         #   权重共享的深度集成, 降低小样本下对 dropout mask 的敏感方差。0 = 关闭 (单次确定性推理)。
                         #   建议 10~20 (时间 ×K); 模型全 LayerNorm 无 BatchNorm, train/eval 切换安全。

# ==================== 无序 triple 目标: 对称分类器 (DeepSets 风格) ====================
# 方案B (最终): 把 triple 目标从有序 (a,b,c) 重构成无序对 (min(a,b), max(a,b), c)。
#   - (a,b,c) 与 (b,a,c) 合并为同一类 → 分类目标天然对称, 模型对"哪侧是大区域"完全无偏:
#     不存在可走的"槽位 a 恒为大区域 → 猜主色"捷径。
#   - 线两侧区域本就是无序对 (只关心"这条线两侧各是什么颜色"), 丢失 a/b 顺序是无损的。
#   - 解码/投票改用"对称双投"(两侧颜色证据对称投给 region_a/region_b), 由 region 一致性投票消歧,
#     最终 region 上色仍可解。
#   - 分类器对称化: 共享侧编码器处理两侧 region 特征, mean pooling 实现顺序不变。
# 实测 (input_B): 122 -> 99 unordered 类 (镜像合并 a!=b 对)。
CLASSIFIER_SIDE_DIM = 32  # 对称分类器: 共享侧编码器输出维度 (region_enc + color_emb → side_dim)

# ==================== 投票/求解: MRF 信念传播 ====================
# 历史上的 v2 硬级联与 v3 STE 联合软投票 (confirm_cascade_vote_marginals) 已删除,
# 其注释保留会误导读者。当前唯一的求解实现是 solve_region_colors_mrf:
#   - unary      = Σ w·log P / Σ w   (w = conf(乘性协同后验), 不用像素加权)
#   - pairwise   = triple 的无序色对势 (log(Q/maxQ) 裁剪)
#   - 求解       = max-product loopy BP, 20 轮, damping 0.5
#   - 训练/推理共用同一函数, 保证口径一致。

# 注: 旧的 VOTE_TEMP (不动点迭代软投票温度) 已随 v2/v3 级联投票一起失效 —— MRF 求解
#   不使用任何软投票温度, 故该常量与相关调参说明一并移除 (避免误以为它仍在生效)。

# --- Line / Edge 特征: 全部在局部坐标系中表达, SE(2) 不变 ---
LINE_FEAT_DIM = 5        # [relative_arc_length, straightness, frt, rel_arc_len, boundary_share]
                          # GNN 主干专注"几何拓扑 + 弧长"(尺度无关的比值/弯曲/密度/拓扑)。
                          # 绝对几何信息 (位置) 由独立分支 line_pos/region_pos 提供, 不混入 X_l。
                          # (g/r_l 已移除: g 依赖 content_scale 整图统计量, 图残缺时漂移; r_l 与 line_pos 冗余)
                          # local_pos_x / local_pos_y 已移除: 中点投影量对分类贡献有限, 删减无关噪声
                         # 尺度归一化用线自身长度 dir_norm 或 content_scale (禁用 img_diag / pos_scale);
                         # frt=局部密度(线长累加返回率): 邻居弦长均值/(均值+自身弦长), 值域 [0,1], 孤立线=0 (tier-1)
                         # rel_arc_len=log(弧长/邻居弦长均值), 同一邻接聚合算法, 反映线相对于局部邻域的尺度 (tier-1)
                         # endpoint_ratio 已移除: = -log(1+straightness) 仅是 straightness 的单调变换, 无独立信息.

# --- 位置嵌入 (独立分支) ---
# line_pos = (line 中点 − 骨架质心) / content_scale, region_pos = (region 质心 − 骨架质心) / content_scale。
#   2 维相对坐标 → 低维 MLP → 嵌入, 作为独立分支加性注入 (不参与 GNN 消息传递)。
#   - 平移/缩放不变 (减质心 + 除 content_scale); 旋转等变 (动漫固定正立, 方向信息是强先验)。
#   - 低维 (POS_HIDDEN), 非 AnT 式 256 维大编码器, 控参数量、防记忆空间位置。
POS_HIDDEN = 16

EDGE_ATTR_DIM = 2        # 无向边: [|tan_half|, len_ratio]
                         # |tan_half| = |tan(α/2)|, 正切半角夹角大小 (取绝对值, 无向, 目标无序)
                         # len_ratio = log(max(len) / min(len)), 对数弧长比 (天然无向)
                         # (已删除 color_agree / path_coherence: 二者分别经 color_embed 节点特征
                         #  与 corner/cont 组内边注入 GNN, 属重复信息)

# 主 L-L 边的关系多热编码: [轮廓连续, 笔画连续, 共享区域]。
# 一条边可以同时属于多个关系, 不把重叠拓扑语义压成单一 edge type。
EDGE_RELATION_DIM = 3
RELATION_DROPPATH_RATE = 0.15  # 关系路径级随机深度: 同一关系的所有边共享一次开关

# Region 线色方位特征的原点 = "软 pole" 的锐度 (见 extract_ll_graph 内同名小节):
#   w = exp(beta * (d/d_max - 1)), beta→0 退化为面积质心, beta→∞ 收敛到最大内切圆圆心。
#   实测 beta=16: 原点落在区域外的比例 0.00%, 内切深度中位数 8.0 (pole 9.0),
#                 形状连续变化时原点最大位移 2.5px (pole 100px) → 在内部且连续。
#   调大更贴近 pole (更深、更"居中"), 但形状微扰下位移变大。
REGION_ORIGIN_BETA = 16.0

# 方位分量取单位方向均值 (每像素等权): 两侧边界像素不抵消且 4/4 seed 提升;
#   其余特征 (X_l/edge) 仍严格 SE(2) 不变 (色线朝向本身是固定语义)
LC_GEO_DIR_EPS = 0.1        # 单位方向平滑阈值 (× 区域自身尺度), 避免核心处方向随机

# log1p 系特征的"分辨率尺度": sign(x)·log1p(|x| / FEAT_LOG_EPS)
#   原版 (eps=1) 在低端 (|x|<<1) 退化为 log1p(|x|)≈|x|, 小值全坍缩到 0 附近 → 丧失分辨率。
#   除以 eps 把低端搬回 log1p 的 O(1) 分辨率区间 (低端放大 1/eps 倍), 高端仍是对数压缩。
#   权衡: 每减小一个数量级 → 低端分辨率 +约0.55, 高端常量偏移 +2.30 (对称常量, 由 bias 吸收)
#     eps=1e-1 → 低端  8.4x, 高端量级  9.20
#     eps=1e-2 → 低端   36x, 高端量级 11.50   (默认)
#     eps=1e-3 → 低端   57x, 高端量级 13.81
FEAT_LOG_EPS = 1e-2

# ==================== 并行笔画粗支路 (CoRe-GNN / 图粗化) ====================
# 不破坏细图，额外输出 line→stroke 归属及笔画组邻接。
# 当前 GNN 支路执行组内消息/聚合广播，不执行 Stroke→Stroke 组间 GNN；
# 已生成的组间 edge_index 可供后续实验，不能将其描述成已生效的组间传播。
STROKE_COARSE_ENABLED = True   # 是否启用并行笔画粗支路
# =============================================================

# ==================== 贡献源消融开关 (定位过拟合来源) ====================
# 分类器由 Triple（对称双侧）与 Side（+n/-n 分开）两个头组成：
#   h_l      : Line 5D 几何/拓扑特征 + 线色嵌入，经过 GNN → hidden_dim
#   h_r      : Region 24D 线色几何压缩 + 位置编码 + 2D 形状先验 → COLOR_EMB_DIM
#              (名字用 h_r 而非 h_r_color: 它已不只是"线色几何", 而是整个区域上下文)
# 消融方式: 数据侧置零(而非删维度), 模型结构/维度不变, 只需递增 GRAPH_CACHE_VERSION。
# 两开关独立, 可组合出任意消融:
#   基线 E0     : 全 True
#   消 h_l      : ENABLE_LINE_STREAM=False (线几何 X_l 置0 + 线色嵌入 color_embed 禁用)
#   消 h_r      : ENABLE_RCOLOR_STREAM=False (region_line_color_geo 置0; 位置/形状仍会加进来)
# ================================================================================
ENABLE_LINE_STREAM = True    # h_l 流（Line X_l 5D + 线色 Embedding）
ENABLE_RCOLOR_STREAM = True  # h_r 流中的线色几何部分 (region_line_color_geo); 位置/形状不受此开关控制

GRAPH_CACHE_VERSION = "v66_boundary_nearest_ink_microfill_220121"  # 微小闭区最近线色合并 + 边界拓扑保护；新旧图缓存隔离
GRAPH_CACHE_BASE = GRAPH_CACHE_VERSION       # 不含消融后缀的 base 版本号 (main 里重算 suffix 时保留 schema 变更)

def dropedge_keep_connected_mask(edge_index, num_nodes, drop_rate):
    """DropEdge 但保护连通性: 保留原图(无向)的一棵生成森林, 仅对非树(冗余)边按
    drop_rate 随机删 → 删边后连通分量数不变, 既不产生孤立节点也不产生新子图。

    动机: 纯 Bernoulli 逐边删会随机把某节点的所有入射边删光 → 该节点在 GNN 中收不到
    任何消息(孤立节点); 也可能切断连通分量产生新子图(孤岛)。本函数先以并查集取原图
    的一棵生成森林并恒留(保证每个原本有边的节点至少留 1 条树边), 再对冗余边随机删。
    线稿图为高度环状结构, 冗余边占比高, 仍能实质性删边。

    随机化: 每次调用先随机打乱边序再跑 UF → 不同调用建出不同生成森林 (树边随机化),
    避免同一批树边永远不被丢弃。树边的反向边也一并保留, 保证 GNN 双向消息传递。

    Args:
        edge_index: (2, E) long, 定向边列(连通性按无向处理; 允许含双向/自环列)。
        num_nodes:  节点数(边索引取值上界, 自动按 max 索引放宽)。
        drop_rate:  非树边的删除概率 ∈ [0,1]。注意: 实际删边比例 = drop_rate × (非树边占比),
                    树状稀疏图下有效删边会自动减小(为保连通性不可避)。
    Returns:
        keep: (E,) bool tensor (与 edge_index 同 device), True=保留该边。
    """
    E = edge_index.size(1)
    device = edge_index.device
    if E == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    n = int(num_nodes)
    max_idx = int(edge_index.max().item())
    if max_idx + 1 > n:
        n = max_idx + 1

    # 随机打乱边序 → 每次建出不同的生成森林 (树边随机化)
    perm = torch.randperm(E, device='cpu')
    ei = edge_index.cpu().numpy()

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # 路径折叠
            x = parent[x]
        return x

    # Pass 1: 按随机顺序跑 UF 建生成森林, 记录无向树边集合
    keep_tree = np.zeros(E, dtype=bool)
    tree_undirected = set()
    for pi in range(E):
        i = int(perm[pi])
        a = find(int(ei[0, i]))
        b = find(int(ei[1, i]))
        if a != b:                  # 该边连接两个不同分量 → 树边, 恒留
            parent[a] = b
            keep_tree[i] = True
            tree_undirected.add((min(int(ei[0, i]), int(ei[1, i])),
                                  max(int(ei[0, i]), int(ei[1, i]))))

    # Pass 2: 树边的反向边也强制保留 (保证双向消息传递)
    for i in range(E):
        if not keep_tree[i]:
            u, v = int(ei[0, i]), int(ei[1, i])
            if (min(u, v), max(u, v)) in tree_undirected:
                keep_tree[i] = True

    keep_tree_t = torch.from_numpy(keep_tree).to(device=device)
    # 非树边以 (1-drop_rate) 概率保留; 树边恒留 (keep_tree 已为 True, OR 不改变)
    keep_nontree = torch.rand(E, device=device) >= drop_rate
    return keep_tree_t | keep_nontree


def dropedge_group_mask(edge_index, stroke_id, cont_stroke_id, drop_rate):
    """DropEdge 丢"粗化组之间的边"(软版): 每条 L-L 边 (u,v) 比较两端在两套粗化组
    (stroke_id 角点→笔画 / cont_stroke_id 方向连续→笔画) 的归属差异度 d ∈ {0,1,2}:
        d=0  两套组都相同 → 恒留;
        d=1  仅一套组不同 → 丢边概率 drop_rate/2;
        d=2  两套组都不同 → 丢边概率 drop_rate。
    即"两端归属组号码越不同, 越容易丢"。组内边(同组)保留, 模拟遮挡切断部件间连接而非随机删边。

    Args:
        edge_index:     (2, E) long, 定向边列。
        stroke_id:      (N,) long 或 None, 角点→笔画粗化组, 负值=无组(不参与差异)。
        cont_stroke_id: (N,) long 或 None, 方向连续→笔画粗化组。
        drop_rate:      最大丢边概率 ∈ [0,1]。
    Returns:
        keep: (E,) bool tensor, True=保留该边。
    """
    E = edge_index.size(1)
    device = edge_index.device
    if E == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    # 双向边结构: [forward (E/2), reverse (E/2)], 同一条无向边的两个方向各占一个位置。
    half = E // 2
    src = edge_index[0][:half]
    dst = edge_index[1][:half]
    diff_deg = torch.zeros(half, device=device, dtype=torch.float32)
    n_groupings = 0
    for gid in (stroke_id, cont_stroke_id):
        if gid is None:
            continue
        n_groupings += 1
        valid = gid >= 0
        gs = gid[src]
        gd = gid[dst]
        diff_deg += (valid[src] & valid[dst] & (gs != gd)).to(torch.float32)
    if n_groupings == 0 or diff_deg.max().item() <= 0.0:
        return torch.ones(E, dtype=torch.bool, device=device)
    p_drop = (diff_deg / float(n_groupings)) * drop_rate   # d/n ∈ [0,1], 越不同越容易丢
    # 双向对称丢弃: 同一条无向边的 forward/reverse 共享同一个随机掩码 (避免单向丢边导致消息不对称)
    keep_half = torch.rand(half, device=device) >= p_drop
    return torch.cat([keep_half, keep_half], dim=0)

# 消融开关状态编码进缓存版本: 任一开关变化都会改变缓存 key, 触发重建, 避免串数据。
# 两位: L(线几何流) C(区域上下文流), 1=保留 0=消融。例 "11"=全开, "01"=消 h_l。
def _ablation_cache_suffix():
    # 几何 backend 必须进入缓存 key: 换 backend 就是换图, 不能复用旧缓存。
    return (f"abl{int(ENABLE_LINE_STREAM)}{int(ENABLE_RCOLOR_STREAM)}"
            f"_cp{CONT_PAIR_BACKEND}_lp{LINE_PATH_BACKEND}"
            f"_sk{os.environ.get('LINE_SKELETON_BACKEND','legacy').lower()}")

GRAPH_CACHE_VERSION = GRAPH_CACHE_VERSION + "_" + _ablation_cache_suffix()

# 置信度热度图渐变色阶 (取代旧固定阈值三段 >=0.6绿 / >=0.4黄 / <0.4红)。
# conf∈(0,1] 沿 红(低确信)→黄(borderline)→绿(高确信) 分段线性连续插值, 无离散跳变;
# 锚点 0.4=黄 与旧 borderline 阈值对齐, 保证 consistency 去饱和校准语义不变;
# conf<=0 (无投票/无效区域) 仍走灰色 (180,180,180), 不入渐变。
CONF_GRAD_T   = (0.1, 0.7, 1.0)
CONF_GRAD_RGB = np.array(
    [[200,   0,   0],   # 0.0 红: 低确信/疑似错误
     [200, 200,   0],   # 0.4 黄: borderline
     [  0, 200,   0]],  # 1.0 绿: 高确信
    dtype=np.float32,
)


_RANDOM_SEED = random.randint(0, 999999)
set_seed(_RANDOM_SEED)


# ==================== 1. 特征计算与几何工具 ====================
def _signed_log1p(x: float, eps: float = FEAT_LOG_EPS) -> float:
    """保号对数压缩 + 低端放大: sign(x)·log1p(|x|/eps).

    原版为 sign(x)·log1p(|x|) (等价于 eps=1), 动机是压制高端大值: 局部位置特征
    以"线长"为单位, 短线远离质心时可达数十, 直接送入 MLP 会压制 LayerNorm 的
    其它维度。但代价是低端被压缩——|x|<<1 时 log1p(|x|)≈|x|, 小值全挤在 0 附近,
    网络无法区分"极小"与"稍小"。

    修复: 压缩前先按 eps 重标定 (等价于把 |x| 放大 1/eps 后再压缩):
      |x| >> eps → log|x| + log(1/eps)   高端仍是对数压缩 (保住 LayerNorm 防护)
      |x| << eps → |x|/eps               低端被放大 1/eps, 不再坍缩到 0
      x = 0      → 0                     连续 (因 log1p(0)=0), 且保持奇函数对称

    注: 低端陡峭必然在对数区累积 +log(1/eps) 的常量偏移, 这是数学上的对价;
        该偏移对称, 由网络 bias 吸收。eps 越小 → 低端分辨率越高、高端偏移越大。

    严格单调 ⇒ 不变量之上的单调变换仍是不变量, 因此不破坏 SE(2) 不变性。
    """
    return math.copysign(math.log1p(abs(x) / (FEAT_LOG_EPS)), x)


def line_local_frame(kp1, kp2, px_sorted, py_sorted):
    """线的局部坐标系 + 有效弦长.

    Returns:
        (ux, uy, nx_v, ny_v, chord)
        u     = 单位方向向量 (kp1→kp2, 方向无语义: 目标无序, 不再强制"大区域在左")
        n     = u 逆时针旋转 90° 的法向量
        chord = 有效端点距离 (像素)

    端点退化保护 (kp1 == kp2, 即只挂在一个关键点上的悬垂路径 / 孤立闭合圆):
      ① 退回排序路径的首尾连线 (悬垂路径的真实走向)
      ② 再退回路径包围盒对角线 (真闭合环首尾相邻, 只有包围盒仍有跨度)
      ③ 仍退化 (单点路径) → 取 +X 方向, chord = 1
    三重退化分支全部只依赖图像自身几何, 因此保持 SE(2) 不变性。
    """
    raw_dx = kp2[0] - kp1[0]
    raw_dy = kp2[1] - kp1[1]
    chord = math.sqrt(raw_dx ** 2 + raw_dy ** 2)
    is_self_loop = chord < 1e-3   # kp1 == kp2 (自环边 / 孤立闭合圆)
    if is_self_loop:
        # 自环边: 首尾连线不可靠 (闭合环贪心追踪一圈后首尾相邻, 得到伪小 chord),
        # 直接跳到包围盒对角线 (闭合环的真实跨度在包围盒, 而非首尾连线)。
        raw_dx = float(px_sorted.max() - px_sorted.min())
        raw_dy = float(py_sorted.max() - py_sorted.min())
        chord = math.sqrt(raw_dx ** 2 + raw_dy ** 2)
    elif chord < 1e-3:
        # 非自环但端点退化 (悬垂路径): 退回排序路径的首尾连线
        raw_dx = float(px_sorted[-1] - px_sorted[0])
        raw_dy = float(py_sorted[-1] - py_sorted[0])
        chord = math.sqrt(raw_dx ** 2 + raw_dy ** 2)
    if chord < 1e-3:
        # 仍退化 (单点路径): 包围盒对角线
        raw_dx = float(px_sorted.max() - px_sorted.min())
        raw_dy = float(py_sorted.max() - py_sorted.min())
        chord = math.sqrt(raw_dx ** 2 + raw_dy ** 2)
    if chord < 1e-3:
        raw_dx, raw_dy, chord = 1.0, 0.0, 1.0
    ux, uy = raw_dx / chord, raw_dy / chord
    return ux, uy, -uy, ux, chord


def path_arc_length(px_sorted, py_sorted, distinct_edges=False):
    """路径几何弧长 (真实路径长度, 像素). 退化时回退为点数.

    连续路径可能包含分叉回溯；distinct_edges=True 使重复无向边只贡献一次弧长。
    仍保留 step<=2.0 容错保护，禁止错误的非邻接跳跃污染长度。
    """
    px = np.asarray(px_sorted, dtype=np.float64)
    py = np.asarray(py_sorted, dtype=np.float64)
    if len(px) <= 1:
        return float(len(px_sorted))
    dx = np.diff(px)
    dy = np.diff(py)
    step = np.sqrt(dx * dx + dy * dy)
    # 只累加 8 邻域连续步长; 跨段跳跃 (step > 2.0) 剔除 (不贡献弧长)
    if distinct_edges:
        # 连续路径可回溯；重复经过同一无向像素边不应重复计入物理弧长。
        x0,y0=px[:-1].astype(np.int64),py[:-1].astype(np.int64)
        x1,y1=px[1:].astype(np.int64),py[1:].astype(np.int64)
        valid=step<=2.0
        if np.any(valid):
            stride=int(max(px.max(),py.max())+2)
            a=y0[valid]*stride+x0[valid];b=y1[valid]*stride+x1[valid]
            lo=np.minimum(a,b);hi=np.maximum(a,b)
            edges=np.stack([lo,hi],axis=1)
            _,idx=np.unique(edges,axis=0,return_index=True)
            arc=float(np.sum(step[valid][idx]))
        else:
            arc=0.0
    else:
        arc = float(np.sum(step[step <= 2.0]))
    if arc < 1e-6:
        arc = float(len(px_sorted))
    return arc


def compute_line_features(kp1, kp2, px_sorted, py_sorted, mid_x, mid_y,
                          origin_x, origin_y, content_scale, arc_len=None):
    """2D Line 特征 (纯函数, 便于 SE(2) 不变性单测).

    返回 [relative_arc_length, straightness]

    全部在"线的局部坐标系"中表达, 并以线自身长度或内容尺度 content_scale 归一化,
    因此对旋转、平移、缩放均不变 (路径点数按弧长代理量随尺度线性变化)。
    不使用图像对角线 (画幅量) 与 pos_scale (95 分位, 离群敏感)。

    endpoint_ratio 已移除: = -log(1+straightness), 与 straightness 单调冗余。
    local_pos_x / local_pos_y 已移除: 中点相对骨架质心的投影对区域着色信息有限。
    local_bend 已移除: 法向弯曲量仅有 straightness 在法向的分量, 无独立信息;
      tanh 有界化取值压到 [-1,1], 削弱了区分度。

    长度/曲线特征的对数压缩 (2026-11-02 修订):
      - relative_arc_length: 原 math.log(x + 1e-6) 在低端 (x<<1) 坍缩到 0,
        且 +1e-6 是像素量纲的绝对常数, 破坏尺度不变性。改用 _signed_log1p
        (eps=FEAT_LOG_EPS) 低端放大 1/eps 倍、高端对数压缩, 与设计意图一致。
      - straightness: 原为裸比值 arc_len/chord, 无界 (闭合环/高弯曲 chord→0 时
        可爆到数十上百), 首层 Linear 被极端值主导, 正常区间 (1~10) 分辨力被稀释。
        改为 log(arc_len/chord): 直线=0、弯曲>0, 单调保号, 压平长尾。
    """
    ux, uy, nx_v, ny_v, chord = line_local_frame(kp1, kp2, px_sorted, py_sorted)
    # 除法用"取下界"而非 "chord + 1e-6": 后者是像素量纲的绝对常数, 缩放时会引入相对误差
    chord_safe = max(chord, 1e-6)
    # --- 计算几何弧长（真实路径长度） ---
    if arc_len is None:
        arc_len = path_arc_length(px_sorted, py_sorted)

    # [0] 相对弧长：弧长 / 内容尺度 (低端放大 + 对数压缩)
    #     content_scale 已在调用方保证 >= 1.0, 无需 +1e-6 兜底。
    relative_arc_length = _signed_log1p(arc_len / content_scale)

    # [1] 直线度 = log(弧长 / 弦长) (对数压缩, 直线=0, 弯曲>0)
    straightness = math.log(arc_len / max(chord, 1e-6))

    return [relative_arc_length, straightness]


def compute_avg_neighbor_chord(edge_index_ll, chords):
    """计算每条线的邻居弦长均值（使用与 compute_line_frt 相同的邻接聚合算法）。

    返回:
        np.ndarray, shape (N,), dtype float64: 每条线的平均邻居弦长。
        孤立线（度为 0）返回 0.0。
    """
    N = len(chords)
    if N == 0:
        return np.zeros((0,), dtype=np.float64)
    chords = np.maximum(np.asarray(chords, dtype=np.float64), 1e-6)

    # 无向邻接 (去重; edge_index_ll 已含双向, 这里兜底去重)
    adj = [[] for _ in range(N)]
    ei = np.asarray(edge_index_ll, dtype=np.int64)
    if ei.size:
        for a, b in ei:
            a = int(a); b = int(b)
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)

    avg_neighbor = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if not adj[i]:
            avg_neighbor[i] = 0.0
            continue
        avg_neighbor[i] = float(np.mean(chords[np.asarray(adj[i], dtype=np.int64)]))

    return avg_neighbor


def compute_line_frt(edge_index_ll, chords):
    """局部密度 (线长累加返回率) 线特征: 1D, SE(2) 不变, 可预计算.

    设计目的: 帮助模型识别*局部结构* (主干 vs 末梢 vs 孤立), 且对跨帧变化鲁棒.
    度数 (degree) 在跨帧时不稳定 (节点可能被拆分/重连), 故不采用 π∝deg 的随机游走
    FRT; 改用*线长累加返回率*: 每条边的返回长度 (从该边返回所需的代价 ≈ 邻居弦长)
    相加后除以节点的度, 得到「平均每条边的返回量」——即局部密度 (连接强度, 而非
    连接总数). 除以度不会抹掉密度: 真密集时邻居弦长总和本身就大, 均值仍相对高.

        avg(i)  = ( Σ_{j∈N(i)} chord_j ) / degree(i)   # 每条边返回长度相加 ÷ 度
        frt(i)  = avg(i) / (avg(i) + chord_i)           # 自身弦长归一, 值域 (0,1)

    - 孤立线(度0): 除零 → frt=0 (明确端点, 天然为 0)
    - 邻居平均弦长 ≫ 自身 → frt → 1 (自身是局部"末梢/细节")
    - 邻居平均弦长 ≪ 自身 → frt → 0 (自身是局部"主干")
    - 由尺度(弦长)决定; 节点拆分时邻居弦长变化温和 → 跨帧鲁棒
    刚性变换下弦长不变; 均匀缩放下弦长同比例 → 比值不变 → SE(2) 不变.
    """
    N = len(chords)
    if N == 0:
        return np.zeros((0,), dtype=np.float64)
    chords_arr = np.maximum(np.asarray(chords, dtype=np.float64), 1e-6)

    avg_neighbor = compute_avg_neighbor_chord(edge_index_ll, chords)

    frt = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if avg_neighbor[i] == 0.0:
            frt[i] = 0.0
        else:
            frt[i] = avg_neighbor[i] / (avg_neighbor[i] + chords_arr[i])

    return frt


def edge_tan_half(src_dir, dst_dir):
    """正切半角有向夹角 tan(α/2) = cross/(1+dot), 连续无跳变, SE(2) 不变.

    反向边 (src↔dst 交换) 时符号取反.

    Args:
        src_dir: 源线方向向量 (feat_dir)
        dst_dir: 目标线方向向量 (feat_dir)

    Returns:
        float: tan(α/2), 裁剪到 [-10, 10]。源线/目标线方向退化时返回 0。
    """
    sx, sy = src_dir
    dx, dy = dst_dir
    ns = math.hypot(sx, sy)
    nd = math.hypot(dx, dy)
    if ns < 1e-3 or nd < 1e-3:
        return 0.0
    dot = (sx * dx + sy * dy) / (ns * nd)
    cross = (sx * dy - sy * dx) / (ns * nd)
    tan_half = cross / (1.0 + dot + 1e-6)
    return max(-10.0, min(10.0, tan_half))


def _line_pca_direction(px, py):
    """笔画 PCA 主方向 (来自 "连续的笔画判断" 论文方法).
    
    对线的全部像素坐标做 PCA，取第一主成分方向作为该线的"全局走向"
    —— 与端点方向不同，PCA 方向对局部曲率鲁棒：弯曲线路的 PCA 方向代表
    其"整体流向"而非"首尾捷径"。
    
    论文核心公式: DP_i = P_history · P_future_i
    即比较"历史 PCA 方向"与各候选"未来 PCA 方向"的点积，
    选最大者作为笔画连续方向。
    
    这里对每条线独立计算 PCA 主方向，随后在边特征中求 |cos| 相似度，
    实现 SE(2) 不变的"笔画全局一致性"信号。
    
    Returns:
        (dx, dy) 单位向量，或 None（路径 < 4 像素）。
    """
    N = len(px)
    if N < 4:
        return None
    pts = np.column_stack([np.asarray(px, dtype=np.float64),
                           np.asarray(py, dtype=np.float64)])
    pts_c = pts - pts.mean(axis=0)
    cov = pts_c.T @ pts_c / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]   # 最大特征值 → 第一主成分
    norm = float(np.linalg.norm(pc1))
    if norm < 1e-8:
        return None
    return tuple(pc1 / norm)


# 8 邻域顺时针环偏移 (全文件统一, 用于关键点环遍历与交叉数计算)
RING_8_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]


def _ring_crossing_number(skeleton_bin: np.ndarray, x: int, y: int) -> int:
    """关键点在 8 邻域环上的交叉数 (crossing number).

    0=孤立点, 1=端点, 2=骨架上的普通点, >=3=分叉点。
    纯函数, 与环遍历共用 RING_8_OFFSETS。
    """
    h, w = skeleton_bin.shape
    ring = np.zeros(8, dtype=np.uint8)
    for ri, (dy, dx) in enumerate(RING_8_OFFSETS):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skeleton_bin[ny, nx]:
            ring[ri] = 1
    ring_s = np.append(ring, ring[0]).astype(np.int16)
    return int(np.sum(np.abs(np.diff(ring_s))) // 2)


def _to_np(x):
    """图字典中的字段既可能是 torch.Tensor 也可能是 numpy/数组, 统一取成 np.ndarray。"""
    return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)


# ==================== 2. L-L 图提取 (extract_ll_graph) ====================
def _compute_group_coarse(pairs, edge_index_ll, N_l):
    """通用分组粗化: 把 pairs 里相连的线归为同一组, 折叠出组级邻接。

    细图 (原 line 图) 完整保留, 组级图仅作长程消息的并行支路。

    连通分量用 **scipy.sparse.csgraph.connected_components** 计算 (替代手写并查集),
    并把组 id 按"节点首次出现顺序"重映射 —— 与旧实现逐位一致 (下游 scatter 依赖该 ids)。

    Args:
        pairs        : 相连线对集合 (最终索引, 已过滤/重映射)
        edge_index_ll: (2, E) 最终双向 L-L 边张量 (已过滤/重映射)
        N_l          : 最终线数

    返回 (group_id, edge_index_group):
      group_id         : (N_l,) long, 每条线所属组 (紧凑 0..num_groups-1)
      edge_index_group : (2, E_g) long, 组级双向边
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if N_l <= 0:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((2, 0), dtype=torch.long)
    if pairs:
        rows = np.fromiter((a for a, _ in pairs), dtype=np.int64, count=len(pairs))
        cols = np.fromiter((b for _, b in pairs), dtype=np.int64, count=len(pairs))
        adj = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                         shape=(N_l, N_l)).tocsr()
        _n, raw = connected_components(adj, directed=False)
    else:
        raw = np.arange(N_l, dtype=np.int64)
    # 按"首次出现"重映射, 复刻旧并查集的 id 顺序 (保证下游数值逐位一致)
    _, first_idx = np.unique(raw, return_index=True)
    remap = np.empty(int(raw.max()) + 1, dtype=np.int64)
    remap[np.asarray(raw)[np.sort(first_idx)]] = np.arange(len(first_idx), dtype=np.int64)
    group_id = remap[raw]

    # 从最终 L-L 边折叠出组级邻接 (双向边各折叠一次, 去重)
    group_edges = set()
    ei = edge_index_ll.numpy() if torch.is_tensor(edge_index_ll) else np.asarray(edge_index_ll)
    for e in range(ei.shape[1]):
        a, b = int(ei[0, e]), int(ei[1, e])
        ga, gb = int(group_id[a]), int(group_id[b])
        if ga == gb:
            continue
        group_edges.add((min(ga, gb), max(ga, gb)))

    if group_edges:
        ei_g = np.zeros((2, 2 * len(group_edges)), dtype=np.int64)
        for i, (ga, gb) in enumerate(sorted(group_edges)):
            ei_g[0, 2 * i] = ga
            ei_g[1, 2 * i] = gb
            ei_g[0, 2 * i + 1] = gb
            ei_g[1, 2 * i + 1] = ga
        edge_index_group = torch.tensor(ei_g, dtype=torch.long)
    else:
        edge_index_group = torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(group_id, dtype=torch.long), edge_index_group


def _pairs_to_edge_index(pairs):
    """把无向线对集合转成 (2, 2E) 双向有向边张量 (组内链式消息传递用)。

    corner_pairs / cont_pairs 是 stroke 内部的线-线连接对, 转成双向边后每个节点
    可沿链聚合前驱+后继。
    """
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    n = len(pairs)
    ei = np.zeros((2, 2 * n), dtype=np.int64)
    for i, (a, b) in enumerate(sorted(pairs)):
        ei[0, 2 * i] = a
        ei[1, 2 * i] = b
        ei[0, 2 * i + 1] = b
        ei[1, 2 * i + 1] = a
    return torch.tensor(ei, dtype=torch.long)


# 最近一次 extract_ll_graph 的交点续接诊断 (A0/A1 审计用, 纯诊断, 不参与数值)
_GEOM_DIAG_LAST = {}


def match_continuations_mwm(lines_at_kp, kp_idx, path_dir_map, line_info,
                            threshold=CONT_STROKE_THRESHOLD):
    """A1: 单个交点上的**非完美最大权匹配** (NetworkX), 纯函数、可单测。

    与 legacy 的唯一区别: legacy 在该交点只保留"最反平行的一对";
    这里允许同一交点多对续接, 但每条半分支(以 Line id 表示)最多用一次, 也允许不匹配。

    判据完全沿用原硬阈值: 有符号余弦 `cos <= -threshold`, 边权 `w = -cos - threshold > 0`。
    `maxcardinality=False` (弱匹配不许抢占), 不额外引入曲率/颜色/Region/置信度惩罚。

    Returns
    -------
    (pairs, diag)
        pairs : set of (min_id, max_id) 规范化线对
        diag  : dict, 逐交点诊断 (候选数/匹配数/余弦/是否用了 PCA 回退/被跳过的非法向量)
    """
    import networkx as nx

    lines = sorted({int(x) for x in lines_at_kp})
    diag = {'kp': int(kp_idx), 'n_lines': len(lines), 'n_candidates': 0,
            'n_matched': 0, 'n_zero_weight': 0, 'n_pca_fallback': 0,
            'n_skipped_vector': 0, 'pairs': [], 'cos': []}
    if len(lines) < 2:
        return set(), diag

    dirs = {}
    for lid in lines:
        v = path_dir_map.get((lid, kp_idx))
        if v is None:
            v = line_info[lid].get('pca_dir')
            if v is not None:
                diag['n_pca_fallback'] += 1
        dirs[lid] = v

    g = nx.Graph()
    g.add_nodes_from(lines)
    for i, a in enumerate(lines):
        va = dirs.get(a)
        if va is None:
            diag['n_skipped_vector'] += 1
            continue
        na = math.hypot(va[0], va[1])
        if not (math.isfinite(na) and na > 1e-3):
            diag['n_skipped_vector'] += 1
            continue
        for b in lines[i + 1:]:
            vb = dirs.get(b)
            if vb is None:
                continue
            nb = math.hypot(vb[0], vb[1])
            if not (math.isfinite(nb) and nb > 1e-3):
                continue
            cosine = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)
            if not math.isfinite(cosine):
                continue
            cosine = max(-1.0, min(1.0, float(cosine)))
            if cosine <= -threshold:
                diag['n_candidates'] += 1
                weight = -cosine - threshold
                diag['cos'].append((a, b, round(cosine, 6)))
                if weight > 0.0:
                    g.add_edge(a, b, weight=weight)
                else:
                    # cos 恰好等于 -threshold: 保留旧版边界语义的计数, 但不让零权重占位
                    diag['n_zero_weight'] += 1

    pairs = {tuple(sorted((int(a), int(b))))
             for a, b in nx.max_weight_matching(g, maxcardinality=False, weight='weight')}
    diag['n_matched'] = len(pairs)
    diag['pairs'] = sorted(pairs)
    return pairs, diag


def _compute_stroke_coarse(corner_pairs, cont_pairs, edge_index_ll, N_l):
    """计算两套笔画分组粗化结构 (CoRe-GNN 并行粗支路, 不破坏细图)。

    分组 1 (角点→笔画): corner_pairs = 角点(crossing==2)相连的线对 → 同一条笔画。
    分组 2 (方向连续→笔画): cont_pairs = 交点附近局部切向一致(path_coherence≥阈值)的线对
      → 同一条连续笔画 (几何分组, 与颜色无关, 能跨交点合并, 见 Rajan & Hammond 2008)。

    ring 支路已移除: 其"区域成员"语义与 region 支路冗余, 且并查集在线桥接下会退化为
    骨架连通分量。环序相邻关系仍保留在 edge_relation_mask 第 0 维 (纯几何边特征)。

    Args:
        corner_pairs   : 角点相连的线对集合 (最终索引)
        cont_pairs     : 方向连续线对集合 (交点局部切向一致, 与颜色无关, 最终索引)
        edge_index_ll  : (2, E) 最终双向 L-L 边张量
        N_l            : 最终线数

    返回 (stroke_id, edge_index_stroke, cont_stroke_id, edge_index_cont_stroke,
          edge_index_intra_stroke, edge_index_intra_cont)
      edge_index_intra_* : (2, 2E) 组内双向边 (corner/cont 线对转边, 见 _pairs_to_edge_index)。
    """
    stroke_id, edge_index_stroke = _compute_group_coarse(corner_pairs, edge_index_ll, N_l)
    cont_stroke_id, edge_index_cont_stroke = _compute_group_coarse(cont_pairs, edge_index_ll, N_l)
    edge_index_intra_stroke = _pairs_to_edge_index(corner_pairs)
    edge_index_intra_cont = _pairs_to_edge_index(cont_pairs)
    return (stroke_id, edge_index_stroke, cont_stroke_id, edge_index_cont_stroke,
            edge_index_intra_stroke, edge_index_intra_cont)


def order_paths_with_sknw(path_labels, path_mask):
    """A2: 用 **sknw** 为 legacy 的每个路径连通块提供有序像素序列。

    与 A0 的关系: **分段完全沿用 legacy** (关键点 3x3 清除 + 8 连通 label()),
    因此 Line 数量/像素集合逐位不变; 只是把"手写贪心 walk + BFS 回溯"
    换成 sknw 的官方实现 (骨架 → 多重图 → 边的 pts 有序像素)。

    返回 (order_map, diag):
      order_map : {label: (px_sorted, py_sorted)}; 只包含能被 sknw 单边完整覆盖的块;
      diag      : 统计 (覆盖块数 / 未覆盖块数 / 多边拆分的块数)。

    未覆盖或多边拆分的块**不写入** order_map, 下游自动回退 legacy walk —— 保证不静默改变几何。
    """
    import sknw

    diag = {'n_blocks': int(path_labels.max()), 'covered': 0, 'multi_edge': 0,
            'uncovered': 0, 'sknw_edges': 0}
    order_map = {}
    mask = np.asarray(path_mask, dtype=np.uint8)
    if not np.any(mask):
        diag['uncovered'] = diag['n_blocks']
        return order_map, diag
    graph = sknw.build_sknw(mask, multi=True, iso=True, ring=True, full=False)
    per_label = {}
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        diag['sknw_edges'] += 1
        pts = np.asarray(data['pts'], dtype=np.int64)
        if pts.size == 0:
            continue
        px = pts[:, 1]
        py = pts[:, 0]
        labs = path_labels[py, px]
        vals, counts = np.unique(labs[labs > 0], return_counts=True)
        if vals.size == 0:
            continue
        lab = int(vals[np.argmax(counts)])
        # 只接受"整条边都属于同一块"的情形, 否则交给 legacy 回退
        if not np.all(labs == lab):
            continue
        per_label.setdefault(lab, []).append((len(px), px, py))
    for lab, items in per_label.items():
        if len(items) > 1:
            diag['multi_edge'] += 1
            continue
        _n, px, py = items[0]
        order_map[int(lab)] = (px.astype(np.int32), py.astype(np.int32))
        diag['covered'] += 1
    diag['uncovered'] = diag['n_blocks'] - diag['covered'] - diag['multi_edge']
    return order_map, diag


def extract_ll_graph(lineart_rgb: np.ndarray, target_rgb: np.ndarray,
                     line_palette: Dict, color_palette: Dict,
                     debug_prefix: Optional[str] = None,
                     ablation=None) -> Optional[Dict[str, Any]]:
    # ablation: 可选 dict, 显式覆盖三大消融开关 { 'line': bool, 'region': bool, 'rcolor': bool }。
    #   默认 None = 用模块全局开关。多进程 spawn 下 worker 进程不继承主进程对全局的修改,
    #   故 worker 必须通过此参数显式传入开关值 (否则消融在 worker 里失效)。
    if ablation is None:
        ablation = {}
    _abl_line = ablation.get('line', ENABLE_LINE_STREAM)
    _abl_rcolor = ablation.get('rcolor', ENABLE_RCOLOR_STREAM)
    _abl_stroke_coarse = ablation.get('stroke_coarse', STROKE_COARSE_ENABLED)
    # 交点续接诊断 (A0/A1 审计用): 逐交点候选/匹配/回退统计, 不参与任何数值。
    _geom_diag = {'backend': CONT_PAIR_BACKEND, 'path_backend': LINE_PATH_BACKEND,
                  'n_junctions': 0, 'n_candidate_pairs': 0, 'n_matched_pairs': 0,
                  'n_multi_pair_junctions': 0, 'n_unmatched_halfbranches': 0,
                  'n_zero_weight': 0, 'n_pca_fallback': 0, 'junctions': []}
    """
    提取 L-L 边中心图 (节点/边特征均为局部坐标系 SE(2) 不变量):
      - 节点: Line (5D 局部坐标系几何特征: [relative_arc_length, straightness, frt, rel_arc_len, boundary_share] + color_index 整数)
      - 边: L-L (2D 属性: [tan_half, len_ratio]),
             每条有向边携带所属线 (src) 的两侧 Region
      - 边分类特征: Region 上下文 (5D × 2)
      - Region 分配: 所属线在环上的 CW/CCW 直接邻居 (非弧段扫描)

    局部坐标系约定 (全文件统一):
        u = (kp2 - kp1) / |kp2 - kp1|          # 方向无语义 (目标无序, 不再强制"大区域在左")
        n = (-u_y, u_x)                        # u 逆时针旋转 90°
        任意向量 v 的局部坐标: (v·u, v·n)
    """
    # 统一线稿前处理：白底标准化 + 小封闭透明区沿最近线色填补，
    # 保留触边 Region；需在 target 分区、骨架化及标签提取之前完成。
    if np.shape(lineart_rgb) != np.shape(target_rgb):
        raise ValueError(f'线稿与 GT 的 RGBA 形状不一致: {np.shape(lineart_rgb)} vs {np.shape(target_rgb)}')
    lineart_rgb = prepare_lineart_for_graph(lineart_rgb)
    if len(line_palette) > MAX_LINE_COLORS:
        # API 直接调用时与 main 相同：仅用最前面的 MAX_LINE_COLORS 个主线色。
        line_palette = {k: v for k, v in line_palette.items() if k < MAX_LINE_COLORS}
    h, w = lineart_rgb.shape[:2]

    # --- 上色图预处理: 线稿透明像素提取连通分量 → 对上色图相应区域去杂 ---
    # 按 Region ID 分组计算众数：从 O(R×H×W) 的重复掩码扫描变成一次像素分组。
    unified_target, region_markers, num_pre_regions = unify_target_by_regions(
        lineart_rgb, target_rgb, min_fraction=0.15)
    target_rgb = unified_target
    white_mask = lineart_rgb[..., 3] == 0
    cross_struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    # --- 骨架化：保留所有短枝，不做剪枝；小孔已合并入邻近线条，
    #     避免断开导致 Region 合并。骨架拓扑变化后需重新生成图缓存。 ---
    skeleton_rgb = skeletonize_lineart(lineart_rgb)
    skeleton_bin = skeleton_rgb[..., 3] > 0

    keypoints = detect_keypoints(skeleton_bin)
    if not keypoints:
        return None

    # --- 清理紧挨交点的伪端点 ---
    junctions_set = set()
    endpoints_list = []
    corners_list = []
    for kp in keypoints:
        cn = _ring_crossing_number(skeleton_bin, kp[0], kp[1])
        if cn >= 3:
            junctions_set.add(kp)
        elif cn == 1:
            endpoints_list.append(kp)
        else:
            corners_list.append(kp)

    cleaned_endpoints = []
    for ep in endpoints_list:
        ex, ey = ep
        near_junction = False
        for junc in junctions_set:
            jx, jy = junc
            if abs(ex - jx) <= 1 and abs(ey - jy) <= 1:
                near_junction = True
                break
        if not near_junction:
            cleaned_endpoints.append(ep)

    keypoints = list(junctions_set) + cleaned_endpoints + corners_list

    # +++ 确保每个骨架连通域至少有一个关键点 (防止孤立闭合圆被漏采样) +++
    # 孤立完美闭合圆既无端点也无分叉点, 若角点检测也未命中, 其 keypoints 为空,
    # 整条闭合圆路径会因 connected_kps 为空被丢弃。此处为其注入一个任意关键点,
    # 使其成为自环边 (kp1 == kp2) 保留在图中。
    # 8连通: 骨架线条是 1 像素宽的, 对角线连接的骨架像素必须视为同一连通域
    skel_label_result = label(skeleton_bin, structure=np.ones((3, 3), dtype=np.uint8))
    skel_labels = skel_label_result[0]
    skel_num = int(skel_label_result[1])
    kp_mask = np.zeros_like(skeleton_bin, dtype=bool)
    if len(keypoints) > 0:
        kps_arr = np.array(keypoints)
        kp_mask[kps_arr[:, 1], kps_arr[:, 0]] = True
    n_injected_kps = 0
    for ci in range(1, skel_num + 1):
        comp_mask = (skel_labels == ci)
        if not np.any(comp_mask & kp_mask):
            ys_c, xs_c = np.where(comp_mask)
            new_kp = (int(xs_c[0]), int(ys_c[0]))
            keypoints.append(new_kp)
            kp_mask[new_kp[1], new_kp[0]] = True
            n_injected_kps += 1
    if n_injected_kps > 0:
        print(f"[图结构] 为 {n_injected_kps} 个孤立骨架连通域注入关键点 (孤立闭合圆保护)")
    # +++ 孤立连通域保护结束 +++

    N_p = len(keypoints)
    if N_p == 0:
        return None

    # --- 骨架质心 (origin) + 内容尺度 content_scale + pos_scale (仅调试用) ---
    skel_ys_all, skel_xs_all = np.where(skeleton_bin)
    if len(skel_ys_all) > 0:
        origin_x = np.mean(skel_xs_all)
        origin_y = np.mean(skel_ys_all)
        skel_dists = np.sqrt((skel_xs_all - origin_x) ** 2 +
                             (skel_ys_all - origin_y) ** 2)
        pos_scale = float(np.percentile(skel_dists, 95))
    else:
        # 无骨架像素: 图随后会返回 None, 此处只为保证后续计算不崩溃,
        # 取 1.0 而非任何画幅量 (min(w,h)/4 之类) —— 避免画幅尺度渗回特征
        origin_x, origin_y = w / 2.0, h / 2.0
        pos_scale = 1.0
    pos_scale = max(pos_scale, 1.0)

    # content_scale = 骨架点相对质心 RMS 展布 sqrt(mean(d^2)), 用于跨线/跨区域/跨图可比的位置/长度归一化,
    #   取代画幅敏感的 img_diag; 用矩量(连续)而非 pos_scale 分位数(跳变)。
    #   注: 取二阶矩(对离群骨架更敏感) 而非一阶矩; 若实测离群拉偏可改 float(np.mean(skel_dists)) 但需重训。
    if len(skel_ys_all) > 0:
        content_scale = float(np.sqrt(np.mean(skel_dists ** 2)))
    else:
        content_scale = pos_scale
    content_scale = max(content_scale, 1.0)

    kp_to_idx = {pt: i for i, pt in enumerate(keypoints)}

    # --- 异构图关键点节点: crossing number (拓扑量, SE(2) 不变) ---
    # 关键点是图的一等节点, 其拓扑类型 (endpoint/corner/junction) 由原始 cn 表达。
    kp_cn_arr = np.empty(N_p, dtype=np.int64)
    for i, kp in enumerate(keypoints):
        kp_cn_arr[i] = _ring_crossing_number(skeleton_bin, kp[0], kp[1])

    # 多像素 junction 簇只共享局部切向的亚像素锚点，不合并 KP 或改变 CN 标签。
    kp_geometry_anchors,kp_junction_cluster_sizes=junction_keypoint_anchors(
        keypoints,kp_cn_arr,(h,w))

    # --- 颜色对照表 ---
    if line_palette and len(line_palette) > 0:
        line_rgb_to_id = {v: k for k, v in line_palette.items()}
        line_rgb_array = np.array(list(line_palette.values()), dtype=np.float32)
        num_line_colors = len(line_palette)
    else:
        line_rgb_to_id = {}
        line_rgb_array = np.zeros((2, 4), dtype=np.float32)
        num_line_colors = 2

    markers_result = label(~skeleton_bin, structure=cross_struct)
    markers: np.ndarray = markers_result[0]  # type: ignore[assignment]
    num_regions: int = int(markers_result[1])  # type: ignore[arg-type]

    # --- Region 面积 + 面积质心 (向量化) ---
    # 面积质心用于 region 局部参考系的"符号稳定": 它是 region 自身的几何属性,
    # 与骨架遍历顺序无关 (骨架遍历顺序会随像素扫描变化, 直接用作基会导致手性随机翻转)
    markers_flat = markers.ravel()
    areas = np.bincount(markers_flat, minlength=num_regions + 1)  # index 0 = 背景

    ys_grid, xs_grid = np.mgrid[0:h, 0:w]
    sum_x = np.bincount(markers_flat, weights=xs_grid.ravel().astype(np.float64),
                        minlength=num_regions + 1)
    sum_y = np.bincount(markers_flat, weights=ys_grid.ravel().astype(np.float64),
                        minlength=num_regions + 1)

    region_areas = {}
    region_centroids = {}
    for r_id in range(1, num_regions + 1):
        region_areas[r_id] = int(areas[r_id])
        if areas[r_id] > 0:
            region_centroids[r_id] = (sum_x[r_id] / areas[r_id],
                                      sum_y[r_id] / areas[r_id])
        else:
            region_centroids[r_id] = (w / 2.0, h / 2.0)

    # Region 线色方位特征原点用"软 pole"(深度加权质心, beta=16): 既在区域内(0.00% 落外)又连续(最大位移 2.5px),
    #   避免面积质心落入非凸区域外(10.7%) 与最大内切圆圆心 argmax 跳变(100px)。
    #   仅服务于 Region 线色方位特征。
    dist_map = distance_transform_edt(markers > 0)
    dist_flat = dist_map.ravel()
    if num_regions > 0:
        d_max_r = ndimage.maximum(dist_map, markers,
                                  index=np.arange(1, num_regions + 1))
    else:
        d_max_r = np.zeros(0, dtype=np.float64)
    d_max_flat = np.zeros(num_regions + 1, dtype=np.float64)
    d_max_flat[1:num_regions + 1] = d_max_r
    # -1 再取 exp: w ∈ [e^-beta, 1], 避免 e^16 量级的大数 (归一化后与不减等价)
    w_flat = np.exp(REGION_ORIGIN_BETA
                    * (dist_flat / np.maximum(d_max_flat[markers_flat], 1e-9) - 1.0))
    w_flat[markers_flat == 0] = 0.0
    xs_f = xs_grid.ravel().astype(np.float64)
    ys_f = ys_grid.ravel().astype(np.float64)
    sw = np.bincount(markers_flat, weights=w_flat, minlength=num_regions + 1)
    swx = np.bincount(markers_flat, weights=w_flat * xs_f, minlength=num_regions + 1)
    swy = np.bincount(markers_flat, weights=w_flat * ys_f, minlength=num_regions + 1)

    region_lc_origin = {}
    for r_id in range(1, num_regions + 1):
        if sw[r_id] > 1e-12:
            region_lc_origin[r_id] = (float(swx[r_id] / sw[r_id]),
                                      float(swy[r_id] / sw[r_id]))
        else:
            region_lc_origin[r_id] = region_centroids[r_id]   # 退化: 空区域

    # --- Line Label: 清除关键点 3×3 + 8连通标记 (分段口径不变; sknw 只提供有序像素) ---
    _sknw_order = None
    path_mask = skeleton_bin.copy()
    for kx, ky in keypoints:
        y_lo = max(0, ky - 1)
        y_hi = min(h, ky + 2)
        x_lo = max(0, kx - 1)
        x_hi = min(w, kx + 2)
        path_mask[y_lo:y_hi, x_lo:x_hi] = False
    path_result = label(path_mask, structure=np.ones((3, 3)))
    path_labels: np.ndarray = path_result[0]  # type: ignore[assignment]
    num_paths: int = int(path_result[1])  # type: ignore[arg-type]
    if LINE_PATH_BACKEND == "sknw":
        # A2: 分段沿用 legacy (Line 数量/像素集合不变), 但像素**顺序**改由 sknw 的
        #     官方骨架→多重图实现给出; sknw 无法单边覆盖的块自动回退 legacy walk。
        _sknw_order, _sknw_diag = order_paths_with_sknw(path_labels, path_mask)
        _geom_diag['sknw'] = _sknw_diag
        if debug_prefix is not None:
            print(f"  [v6 sknw 排序] 块 {_sknw_diag['n_blocks']} → 覆盖 {_sknw_diag['covered']}"
                  f", 多边拆分回退 {_sknw_diag['multi_edge']}, 未覆盖回退 {_sknw_diag['uncovered']}")

    X_l = []
    chords = []          # 每条线有效弦长 (自身尺度, 供局部密度 frt 归一)
    arc_lens = []        # 每条线几何弧长 (供 rel_arc_len = log(arc_len/avg_neighbor_chord))
    color_indices = []
    point_to_lines = {i: [] for i in range(N_p)}
    line_idx = 0
    line_info = {}
    path_to_line = {}
    path_dir_map = {}

    # --- 关键点-路径关联: 每个关键点独立检查其 5×5 触达的路径 ---
    # 旧版把关键点标签直接写入 kp_label_map，近距离关键点会互相覆盖。
    path_to_kps = {}
    for kp_idx, (kx,ky) in enumerate(keypoints):
        ylo,yhi=max(0,ky-2),min(h,ky+3)
        xlo,xhi=max(0,kx-2),min(w,kx+3)
        for lab in np.unique(path_labels[ylo:yhi,xlo:xhi]):
            if lab > 0:
                path_to_kps.setdefault(int(lab),set()).add(kp_idx)

    # 向量化路径像素枚举: 一次性取出所有路径像素并按 path_labels 分组,
    # 替代原先 O(num_paths × H×W) 的逐路径 np.where(path_labels == i) 全图扫描 (预处理主热点之一).
    # 稳定排序保持各路径内像素的原始(row-major)顺序, 与 np.where(path_labels == i) 完全一致.
    _py_all, _px_all = np.where(path_labels > 0)
    _lbl_all = path_labels[_py_all, _px_all]
    _order = np.argsort(_lbl_all, kind="stable")
    _py_all = _py_all[_order]
    _px_all = _px_all[_order]
    _lbl_all = _lbl_all[_order]
    _uniq, _split_idx = np.unique(_lbl_all, return_index=True)
    if _split_idx.size == 0:
        # 极短线稿在关键点 3x3 清理后可能完全没有可用路径；应优雅跳过。
        return None
    _segs = [slice(_split_idx[k], _split_idx[k + 1]) for k in range(len(_split_idx) - 1)]
    _segs.append(slice(_split_idx[-1], len(_lbl_all)))
    for _lab, _seg in zip(_uniq.tolist(), _segs):
        py = _py_all[_seg]  # np.where 返回 (行y, 列x)
        px = _px_all[_seg]

        # 不再使用覆盖式 kp_label_map；保留每个有效关键点的关联信息。
        connected_kps = sorted(path_to_kps.get(int(_lab), ()))
        if len(connected_kps) == 0:
            continue
        if len(connected_kps) > 2:
            # path_labels 的一个连通块有时几何上靠近 3 个以上关键点：
            # 一条 Line 只能有两个端点。保留空间跨度最大的两个候选作为端点，
            # 其余靠近中段/平行笔画的关键点不能凭 5x5 接触直接建立虚假连接。
            connected_kps = list(max(
                ((a,b) for ai,a in enumerate(connected_kps) for b in connected_kps[ai+1:]),
                key=lambda p: ((keypoints[p[0]][0]-keypoints[p[1]][0])**2
                               +(keypoints[p[0]][1]-keypoints[p[1]][1])**2,
                               -p[0],-p[1])))
        idx1 = connected_kps[0]
        idx2 = connected_kps[1] if len(connected_kps) > 1 else connected_kps[0]
        kp1, kp2 = keypoints[idx1], keypoints[idx2]

        # --- 沿像素邻接图连续追踪 ---
        # 不再用 unvisited.pop() 强制跳到空间不相邻的另一段。
        # 分叉处如需经过已访问节点以到达其他末梢，允许真实邻接的回溯。
        if _sknw_order is not None and int(_lab) in _sknw_order:
            px_sorted, py_sorted = _sknw_order[int(_lab)]   # sknw 给的 8 连通有序序列
            _n_backtrack = 0
        else:
            px_sorted, py_sorted, _n_backtrack = trace_skeleton_component(px,py)
        # 对齐关键点方向: 确保序列靠近 kp1 开始 (自环 kp1==kp2 时为中立即无影响)
        dist_start_to_kp1 = abs(int(px_sorted[0]) - kp1[0]) + abs(int(py_sorted[0]) - kp1[1])
        dist_end_to_kp1 = abs(int(px_sorted[-1]) - kp1[0]) + abs(int(py_sorted[-1]) - kp1[1])
        if dist_start_to_kp1 > dist_end_to_kp1:
            px_sorted = px_sorted[::-1].copy()
            py_sorted = py_sorted[::-1].copy()
        # 回溯是为了使路径连续，物理中点应基于去重后的骨架像素。
        _first_pixel_idx = np.sort(np.unique(
            py_sorted.astype(np.int64)*w+px_sorted,return_index=True)[1])
        mid_idx = int(_first_pixel_idx[len(_first_pixel_idx)//2])
        mid_x = float(px_sorted[mid_idx])
        mid_y = float(py_sorted[mid_idx])

        # --- 交点附近单分支切向（弧长窗口 TLS，避开交点混合与回溯） ---
        tangent_confidence = {}
        for ckp_idx in connected_kps:
            v, conf = estimate_keypoint_branch_tangent(
                px_sorted, py_sorted, kp_geometry_anchors[ckp_idx], radius=12.0)
            path_dir_map[(line_idx, ckp_idx)] = v
            tangent_confidence[ckp_idx] = conf

        # --- 3D Line 特征 (局部坐标系 SE(2) 不变量) + color_index (整数) ---
        # 局部坐标系 / 弦长: compute_line_features(2D) + compute_line_frt(1D)
        _ux, _uy, _nx_v, _ny_v, norm_de = line_local_frame(kp1, kp2, px_sorted, py_sorted)
        raw_dx, raw_dy = _ux * norm_de, _uy * norm_de   # 有效方向向量 (供边特征复用同一坐标系)
        arc_len = path_arc_length(px_sorted,py_sorted,distinct_edges=True)
        line_feat = compute_line_features(
            kp1, kp2, px_sorted, py_sorted, mid_x, mid_y,
            origin_x, origin_y, content_scale, arc_len=arc_len)

        # 众数线色: 向量化 np.unique(packed-RGB) 众数 —— 等价 Counter.most_common(1)[0][0]
        # (含平局取"先出现"约定, 与 B3 主色填充同款)。原逐像素 tuple+Counter 是预处理主热点(L799-801)。
        path_colors = skeleton_rgb[py, px]
        pc = path_colors.reshape(-1, 4).astype(np.int64)
        packed = (pc[:, 0] << 24) | (pc[:, 1] << 16) | (pc[:, 2] << 8) | pc[:, 3]
        vals, counts = np.unique(packed, return_counts=True)
        max_c = int(counts.max())
        cand = vals[counts == max_c]
        if cand.size == 1:
            mode_packed = int(cand[0])
        else:
            # 平局: 取光栅序中最早出现的颜色 (等价 Counter 先插入键)
            first_idx = np.array([int(np.argmax(packed == v)) for v in cand])
            mode_packed = int(cand[int(np.argmin(first_idx))])
        # 与 build_global_palettes 的线色口径一致：线色由 RGB 决定，
        # Alpha 只表示笔触的抗锯齿覆盖率，不产生新的 line_color ID。
        mode_color = ((mode_packed >> 24) & 0xFF,
                      (mode_packed >> 16) & 0xFF,
                      (mode_packed >> 8) & 0xFF,
                      255)
        if mode_color in line_rgb_to_id:
            line_cidx = line_rgb_to_id[mode_color]
        else:
            c_np = np.array(mode_color, dtype=np.float32)
            dists = np.sum((line_rgb_array - c_np) ** 2, axis=1)
            line_cidx = int(np.argmin(dists))
        # line_cidx: 离散线色类别索引 (0 ~ num_line_colors-1), 不归一化
        # 颜色由模型通过 nn.Embedding 学习语义表示

        # 计算几何弧长 (复用 path_arc_length, 与线特征口径一致: 剔除跨段跳跃)
        # [relative_arc_length, straightness]
        #   + frt (局部密度, 在 edge_index_ll 定稿后图级追加, 见下方)
        X_l.append(line_feat)
        color_indices.append(line_cidx)
        chords.append(norm_de)
        arc_lens.append(arc_len)

        for ckp_idx in connected_kps:
            point_to_lines[ckp_idx].append(line_idx)

        s_idx_t = max(0, mid_idx - 3)
        e_idx_t = min(len(px_sorted) - 1, mid_idx + 3)
        local_dir_x = int(px_sorted[e_idx_t] - px_sorted[s_idx_t])
        local_dir_y = int(py_sorted[e_idx_t] - py_sorted[s_idx_t])
        if local_dir_x == 0 and local_dir_y == 0:
            local_dir_x = kp2[0] - kp1[0]
            local_dir_y = kp2[1] - kp1[1]

        line_info[line_idx] = {
            'mid': (mid_x, mid_y),
            'local_dir': (local_dir_x, local_dir_y),
            'feat_dir': (raw_dx, raw_dy),
            'feat_len': norm_de,
            'kp1': kp1, 'kp2': kp2,
            'connected_kps': tuple(connected_kps),
            'tangent_confidence': tangent_confidence,
            'tangent_dirs': {ki: path_dir_map[(line_idx, ki)] for ki in connected_kps},
            'bfs_order': (px_sorted, py_sorted),
            'pixel_count': len(px),          # 像素点数（用于权重等）
            'path_len_raw': arc_len,         # 真实几何弧长
            'color_index': line_cidx,
            'pca_dir': _line_pca_direction(px, py),  # PCA 全局方向
        }
        path_to_line[_lab] = line_idx
        line_idx += 1


    N_l = line_idx
    if N_l == 0:
        return None

    X_l = torch.tensor(X_l, dtype=torch.float32)              # (N_l, 2) 前2D (rel_arc_length, straightness), FRT+rel_arc_len+boundary_share 后追加
    color_index_t = torch.tensor(color_indices, dtype=torch.long)

    # ==========================================================
    # --- 构建 line_label_map: 骨架像素 → line_idx ---
    # ==========================================================
    ring_cw = RING_8_OFFSETS
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    line_label_map = np.full((h, w), -1, dtype=np.int32)
    path_ys, path_xs = np.where(path_labels > 0)
    for ppy, ppx, pl in zip(path_ys, path_xs, path_labels[path_ys, path_xs]):
        if int(pl) in path_to_line:
            line_label_map[ppy, ppx] = path_to_line[int(pl)]

    # 填充关键点 3×3 邻域
    for kp_idx, lines_at_kp in point_to_lines.items():
        if not lines_at_kp:
            continue  # 孤立关键点 (无有效连接线), 邻域保持未标记
        kx, ky = keypoints[kp_idx]
        for dy_r, dx_r in ring_cw:
            ny, nx = ky + dy_r, kx + dx_r
            if (0 <= ny < h and 0 <= nx < w
                    and skeleton_bin[ny, nx] and line_label_map[ny, nx] < 0):
                best_l, best_dot = -1, -1.0
                for l_idx in lines_at_kp:
                    v = path_dir_map.get((l_idx, kp_idx))
                    if v is None:
                        continue
                    vx, vy = v
                    vnorm = math.hypot(vx, vy)
                    if vnorm < 1e-6:
                        continue
                    rnorm = math.hypot(dx_r, dy_r)
                    dot = (vx * dx_r + vy * dy_r) / (vnorm * rnorm)
                    if dot > best_dot:
                        best_dot = dot
                        best_l = l_idx
                if best_l >= 0:
                    line_label_map[ny, nx] = best_l
                else:
                    # 若点积匹配失败（通常因路径过短导致方向向量缺失），
                    # 直接分配该关键点的第一条邻接线，保证骨架连续性和环扫描完整性。
                    line_label_map[ny, nx] = lines_at_kp[0]

    # ==========================================================
    # --- 构建 L-L 边 + Region 分配 (核心新逻辑) ---
    # ==========================================================

    # 预计算 Region 形状特征
    all_areas = [region_areas[rid] for rid in region_areas]
    area_95 = float(np.percentile(all_areas, 95)) if all_areas else 1.0
    area_95 = max(area_95, 1.0)
    # 面积归一化参考 = 同图区域面积的中位数 (同类节点集合自身的统计量)。
    # 旧版用"面积占画幅比例 (area/(h*w)) 的 95 分位", 是画幅量 → 已弃用
    area_median = float(np.median(all_areas)) if all_areas else 1.0
    area_median = max(area_median, 1.0)

    # --- Per-line 两侧 Region: 沿路径多点 8 邻域投票 (大/小无方向) ---
    # 单一真相源: 线的两侧 region 只在此处计算一次, corner 建边与训练目标 (triple_label)
    #   均引用此处结果, 消除旧"中点法线采样(line_info.region_big/small) vs 投票"两套算法不一致。
    # 语义无方向: 按面积排序, line_region_a = 大区域, line_region_b = 小区域 (均 0-indexed)。
    line_region_a_list = []
    line_region_b_list = []
    for l_id in range(N_l):
        bfs_px, bfs_py = line_info[l_id]['bfs_order']
        # 同一物理像素即使回溯时出现多次，在 Region 侧边投票中只计一次。
        _first_ix=np.sort(np.unique(
            bfs_py.astype(np.int64)*w+bfs_px,return_index=True)[1])
        bfs_px,bfs_py=bfs_px[_first_ix],bfs_py[_first_ix]
        n_bfs = len(bfs_px)
        # 使用几何弧长决定采样点数，至少 5 个，每 3 像素取一点
        arc_len = line_info[l_id]['path_len_raw']
        num_samples = max(5, int(arc_len / 3.0))
        # 沿弧长均匀采样（转换为像素索引比例）
        sample_fracs = np.linspace(0.0, 1.0, num_samples, endpoint=True)
        region_votes = {}
        for frac in sample_fracs:
            si = int(frac * (n_bfs - 1))
            si = min(si, n_bfs - 1)
            sx, sy = int(bfs_px[si]), int(bfs_py[si])
            # 检查 8 邻域（同原）
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        rid = markers[ny, nx]
                        if rid > 0:
                            region_votes[rid] = region_votes.get(rid, 0) + 1

        # 按投票数排序, 取 top-2, 再按面积排 (大/小无方向)
        sorted_r = sorted(region_votes.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_r) >= 2:
            r1, r2 = sorted_r[0][0], sorted_r[1][0]
            if region_areas.get(r1, 0) < region_areas.get(r2, 0):
                r1, r2 = r2, r1
            line_region_a_list.append(r1 - 1)
            line_region_b_list.append(r2 - 1)
        elif len(sorted_r) == 1:
            line_region_a_list.append(sorted_r[0][0] - 1)
            line_region_b_list.append(sorted_r[0][0] - 1)
        else:
            line_region_a_list.append(0)
            line_region_b_list.append(0)

    # --- 角点笔画分组 (corner group): 被交点(cn>=3)隔断、仅经角点(cn==2)相连的线归为同一笔画 ---
    # 即"联通的 Node 按交点切分": 直接对每个角点关键点取其相连的所有线两两 union
    # (角点处应恰有 2 条线), 不依赖环序相邻; 交点(cn>=3)是分隔符, 不参与 union。
    corner_pairs = set()
    for kp_idx, lines_at_kp in point_to_lines.items():
        if int(kp_cn_arr[kp_idx]) != 2:
            continue
        _ls = sorted(set(lines_at_kp))
        for _i in range(len(_ls)):
            for _j in range(_i + 1, len(_ls)):
                if _ls[_i] != _ls[_j]:
                    corner_pairs.add((min(_ls[_i], _ls[_j]), max(_ls[_i], _ls[_j])))

    # --- 关键点环遍历: 同时构建 L-L 边 + 分配两侧 Region ---
    edge_data = {}  # (l_a, l_b) → {regions, keypoints, ...}
    cont_pairs = set()    # 方向连续线对 (交点局部切向一致, 与颜色无关; 供"连续笔画"分组)
    ring_pairs = set()    # 环序相邻且共享 region 的线对 (edge_relation_mask 第0维 + MRF pairwise transition 统计)
    keypoint_line_order = {}  # kp_idx -> 环序线列表 (关键点节点保存相接线的线序, 过滤前索引)

    for kp_idx, lines_at_kp in point_to_lines.items():
        if len(lines_at_kp) < 2:
            continue
        kx, ky = keypoints[kp_idx]

        # CW 扫描环, 记录每个位置的类型
        ring_entries = []  # (type, val, ring_pos)
        for ri, (dy_r, dx_r) in enumerate(ring_cw):
            ny, nx = ky + dy_r, kx + dx_r
            if 0 <= ny < h and 0 <= nx < w:
                ll_val = line_label_map[ny, nx]
                rid = markers[ny, nx]
                if ll_val >= 0 and ll_val in lines_at_kp:
                    ring_entries.append(('L', ll_val, ri))
                elif rid > 0:
                    ring_entries.append(('R', rid, ri))
                else:
                    ring_entries.append(None)
            else:
                ring_entries.append(None)

        # 提取唯一的 line 环序 (用于建 L-L 边)
        ring_lines_unique = []
        for entry in ring_entries:
            if entry is not None and entry[0] == 'L':
                if not ring_lines_unique or ring_lines_unique[-1] != entry[1]:
                    ring_lines_unique.append(entry[1])
        if len(ring_lines_unique) > 1 and ring_lines_unique[0] == ring_lines_unique[-1]:
            ring_lines_unique.pop()

        if len(ring_lines_unique) < 2:
            continue

        # 关键点节点保存相接线的环序 (异构图: 关键点是一等节点, 记录其相接线顺序)。
        keypoint_line_order[kp_idx] = list(ring_lines_unique)

        # 方向连续分组 (独立于 L-L 边): 共享任意关键点的两线, 方向一致则合并。
        # 不依赖 L-L 边的环序相邻 —— 直接对 lines_at_kp (该关键点相接的所有线) 两两配对。
        # (Rajan & Hammond 2008 的 stroke-tracing: 到达交点(歧义点)后做"硬选择"——在多个
        #   stroke future 中选与历史方向最一致的一个继续, 而非所有过阈值的都 union)。
        # 故每个关键点只选"最反平行"的那一对合并, 其余候选放弃。
        # 关键: 用有符号点积(射线夹角), 不是 |cos|(直线夹角)。
        #   path_dir_map 的切向都"远离关键点"定向; 连续笔画穿过交点时两臂反平行 → 点积≈-1;
        #   分叉(fork)的两臂近同向 → 点积≈+1。abs() 会把两者都折成≈1 而误并分叉,
        #   故续接判据 = 有符号点积 <= -阈值 (反平行)。每个交点最多只选出最反平行的一组。
        _lines_at = sorted(set(lines_at_kp))
        _geom_diag['n_junctions'] += 1
        if CONT_PAIR_BACKEND == "mwm":
            # A1: 非完美最大权匹配 (允许同交点成对多对; 每条半分支最多一次)
            _pairs_a1, _diag_a1 = match_continuations_mwm(
                _lines_at, kp_idx, path_dir_map, line_info,
                threshold=CONT_STROKE_THRESHOLD)
            cont_pairs.update(_pairs_a1)
            _geom_diag['n_candidate_pairs'] += _diag_a1['n_candidates']
            _geom_diag['n_matched_pairs'] += _diag_a1['n_matched']
            _geom_diag['n_zero_weight'] += _diag_a1['n_zero_weight']
            _geom_diag['n_pca_fallback'] += _diag_a1['n_pca_fallback']
            if _diag_a1['n_matched'] > 1:
                _geom_diag['n_multi_pair_junctions'] += 1
            _geom_diag['n_unmatched_halfbranches'] += (
                _diag_a1['n_lines'] - 2 * _diag_a1['n_matched'])
            _geom_diag['junctions'].append(_diag_a1)
        else:
            _best_coh = 1.0   # 取最负(最反平行)的点积
            _best_pair = None
            for _i in range(len(_lines_at)):
                for _j in range(_i + 1, len(_lines_at)):
                    _a, _b = _lines_at[_i], _lines_at[_j]
                    # 方向连续判断用"交点附近的局部切向" (path_dir_map), 而非整条线的全局 PCA 方向。
                    # 弯曲线路的整条 PCA 主方向 ≠ 交点处切线方向, 会在交点处把两条局部走向
                    # 不同的线错误合并 (全局 PCA 对局部曲率鲁棒 → 反而丢失了交点处斜率信息)。
                    _pa = path_dir_map.get((_a, kp_idx))
                    _pb = path_dir_map.get((_b, kp_idx))
                    if _pa is None:
                        _pa = line_info[_a].get('pca_dir')
                    if _pb is None:
                        _pb = line_info[_b].get('pca_dir')
                    if _pa is None or _pb is None:
                        continue
                    _na = math.hypot(_pa[0], _pa[1])
                    _nb = math.hypot(_pb[0], _pb[1])
                    if _na > 1e-3 and _nb > 1e-3:
                        _coh = (_pa[0] * _pb[0] + _pa[1] * _pb[1]) / (_na * _nb)   # 有符号 cos
                        if _coh < _best_coh:
                            _best_coh = _coh
                            _best_pair = (min(_a, _b), max(_a, _b))
            if _best_pair is not None and _best_coh <= -CONT_STROKE_THRESHOLD:
                cont_pairs.add(_best_pair)
                _geom_diag['n_matched_pairs'] += 1
                _geom_diag['n_candidate_pairs'] += 1
                _geom_diag['junctions'].append(
                    {'kp': int(kp_idx), 'n_lines': len(_lines_at),
                     'n_candidates': 1, 'n_matched': 1, 'pairs': [list(_best_pair)],
                     'cos': [[_best_pair[0], _best_pair[1], round(_best_coh, 6)]]})
            else:
                _geom_diag['n_unmatched_halfbranches'] += len(_lines_at)

        # 角点 (corner, cn==2): 该关键点只切分一条笔画 → 恰好 2 条线,
        # 且这两"线"是同一条笔画的两半, 必须无条件相连 (不能受 region 配置约束)。
        # 交点 (junction, cn>=3): >=3 条线, 每条线是两个 wedge (区域) 之间的边界。
        # 注意: 必须用拓扑 crossing number (kp_cn_arr) 而非 len(lines_at_kp) 判定。
        #   point_to_lines 经 5×5 关键点膨胀匹配, 交点(cn>=3)的某条线常被漏匹配
        #   (len==2 却实为交点), 若按 len 判定会把两条不同的线误并成同一笔画 →
        #   墨色不一致的 corner 边 (实测 140 个 cn==3 交点被误判, 而真角点 147 个仅 1 个不一致)。
        is_corner = (int(kp_cn_arr[kp_idx]) == 2)

        # 对每对相邻 line (环序), 建 L-L 边并分配 Region
        for i in range(len(ring_lines_unique)):
            l_a = ring_lines_unique[i]
            l_b = ring_lines_unique[(i + 1) % len(ring_lines_unique)]
            pair = (min(l_a, l_b), max(l_a, l_b))
            # ring_pairs 不在此无条件添加: 环序相邻的线对只有在真正"共享同一有效 region"
            # (即它们是同一区域的两条相邻边界, 属于同一轮廓环)时才标记为环序相邻。
            # ring_pairs 用于 edge_relation_mask 第0维 (纯几何关系) 与 MRF pairwise transition
            # 统计, 不再用于粗分组支路 (ring 支路已移除, 见 _compute_stroke_coarse)。
            # corner_pairs 已在上方按"角点(cn==2)相连的线"独立构建, 不在此重复添加。

            # 找到 l_a 和 l_b 在 ring_entries 中的位置
            a_positions = [j for j, e in enumerate(ring_entries)
                          if e is not None and e[0] == 'L' and e[1] == l_a]
            b_positions = [j for j, e in enumerate(ring_entries)
                          if e is not None and e[0] == 'L' and e[1] == l_b]
            if not a_positions or not b_positions:
                continue

            # 选择在环序中相邻的 a, b 位置对
            best_a_pos, best_b_pos = None, None
            best_gap = float('inf')
            for ap in a_positions:
                for bp in b_positions:
                    # CW 距离: a → b
                    gap_ab = (bp - ap) % len(ring_entries)
                    if gap_ab == 0:
                        gap_ab = len(ring_entries)
                    # CW 距离: b → a
                    gap_ba = (ap - bp) % len(ring_entries)
                    if gap_ba == 0:
                        gap_ba = len(ring_entries)
                    min_gap = min(gap_ab, gap_ba)
                    if min_gap < best_gap:
                        best_gap = min_gap
                        if gap_ab <= gap_ba:
                            best_a_pos, best_b_pos = ap, bp
                        else:
                            best_a_pos, best_b_pos = bp, ap

            if best_a_pos is None or best_b_pos is None:
                continue

            n = len(ring_entries)
            a_pos = best_a_pos

            # 找 l_a 自身的两侧 region (直接邻居, 非弧段扫描)
            # CW 方向 (+1, +2, ...): l_a 顺时针侧的 region
            region_cw = None
            for step in range(1, n):
                idx = (a_pos + step) % n
                entry = ring_entries[idx]
                if entry is not None and entry[0] == 'L':
                    break  # 遇到另一条线, 中间无 region
                if entry is not None and entry[0] == 'R':
                    region_cw = entry[1]
                    break

            # CCW 方向 (-1, -2, ...): l_a 逆时针侧的 region
            region_ccw = None
            for step in range(1, n):
                idx = (a_pos - step) % n
                entry = ring_entries[idx]
                if entry is not None and entry[0] == 'L':
                    break
                if entry is not None and entry[0] == 'R':
                    region_ccw = entry[1]
                    break

            # 分配两侧 Region:
            #  - 交点: 两侧必须是两个不同 region (两个楔形), 否则跳过
            #  - 角点: 两"线"是同一条笔画的两半, 两侧 region 直接取投票结果 (与训练目标一致),
            #   大/小无方向; 退化(两侧同区=内部线)时退回环扫描。
            if is_corner:
                region_a = line_region_a_list[l_a] + 1  # 0-indexed → 1-indexed
                region_b = line_region_b_list[l_a] + 1
                if region_a == region_b:
                    # 投票两侧同区(内部线) → 退回环扫描; 仍无效则跳过
                    if region_cw is None or region_ccw is None:
                        continue
                    region_a, region_b = region_cw, region_ccw
            else:
                if region_cw is None or region_ccw is None:
                    continue
                if region_cw == region_ccw:
                    continue
                region_a = region_cw
                region_b = region_ccw

            # --- 邻接拓扑: 从 la 和 lb 两侧各扫描, 确定互相邻接关系 ---
            # la CW 方向首线 (= la 的 CW 区域另一侧的线)
            first_cw_from_la = None
            for step in range(1, n):
                e = ring_entries[(best_a_pos + step) % n]
                if e is not None and e[0] == 'L':
                    first_cw_from_la = e[1]
                    break
            # la CCW 方向首线
            first_ccw_from_la = None
            for step in range(1, n):
                e = ring_entries[(best_a_pos - step) % n]
                if e is not None and e[0] == 'L':
                    first_ccw_from_la = e[1]
                    break
            # lb CW 方向首线
            first_cw_from_lb = None
            for step in range(1, n):
                e = ring_entries[(best_b_pos + step) % n]
                if e is not None and e[0] == 'L':
                    first_cw_from_lb = e[1]
                    break
            # lb CCW 方向首线
            first_ccw_from_lb = None
            for step in range(1, n):
                e = ring_entries[(best_b_pos - step) % n]
                if e is not None and e[0] == 'L':
                    first_ccw_from_lb = e[1]
                    break

            # la 的 CW region: la 和 first_cw_from_la 共享 → lb 邻接 iff first_cw_from_la == lb
            adj_la_cw = (first_cw_from_la == l_b)
            # la 的 CCW region: la 和 first_ccw_from_la 共享
            adj_la_ccw = (first_ccw_from_la == l_b)
            # lb 的 CW region: lb 和 first_cw_from_lb 共享 → la 邻接 iff first_cw_from_lb == la
            adj_lb_cw = (first_cw_from_lb == l_a)
            adj_lb_ccw = (first_ccw_from_lb == l_a)

            # ring_pairs: 仅"环序直接相邻 + 共享同一有效 region"的线对 (edge_relation_mask
            # 第0维的几何关系 + MRF pairwise transition 统计)。中间无 region 的紧贴线对
            # 或环首尾隔着多个 R/None 的闭合对, 不共享有效 region, 不算环序相邻。
            if (adj_la_cw and region_cw is not None) or (adj_la_ccw and region_ccw is not None):
                ring_pairs.add(pair)

            # lb 的两侧 region
            lb_region_cw = None
            for step in range(1, n):
                e = ring_entries[(best_b_pos + step) % n]
                if e is not None and e[0] == 'R':
                    lb_region_cw = e[1]
                    break
            lb_region_ccw = None
            for step in range(1, n):
                e = ring_entries[(best_b_pos - step) % n]
                if e is not None and e[0] == 'R':
                    lb_region_ccw = e[1]
                    break
            # lb 的面积排序: 大区域 = lb_r_a, 小区域 = lb_r_b
            lb_r_a, lb_r_b = lb_region_cw, lb_region_ccw
            if lb_r_a is not None and lb_r_b is not None:
                if region_areas.get(lb_r_a, 0) < region_areas.get(lb_r_b, 0):
                    lb_r_a, lb_r_b = lb_r_b, lb_r_a
            lb_cw_is_a = (lb_region_cw == lb_r_a) if lb_r_a is not None else True

            # 合并到 edge_data (同一条边可能在多个关键点出现)
            if pair not in edge_data:
                edge_data[pair] = {
                    'region_a': region_a,
                    'region_b': region_b,
                    'keypoints': [kp_idx],
                    'l_a_raw': l_a,
                    'l_b_raw': l_b,
                    'adj_la_cw': adj_la_cw, 'adj_la_ccw': adj_la_ccw,
                    'adj_lb_cw': adj_lb_cw, 'adj_lb_ccw': adj_lb_ccw,
                    'lb_cw_is_a': lb_cw_is_a,
                }
            else:
                # 多个关键点: 保留已有的 region 分配 (第一个关键点通常最可靠)
                edge_data[pair]['keypoints'].append(kp_idx)

    # 主路放宽: 共享关键点即连接 (不再要求环序相邻), 独立于环序扫描单独跑一轮。
    #   原逻辑放在主循环底部、被 `ring_lines_unique < 2` 的 continue 跳过, 导致环序扫描
    #   失败(只识别到 1 条线, 常见于角点两臂之一的邻域像素被错标)的关键点漏掉 L-L 边。
    #   移到循环外后, 对每个关键点的所有相接线两两补边, 保证"共享关键点即连接"不因
    #   环序扫描失败而漏边。补充边无共享 region 语义 (region 填 -1, 不进 R-R 邻接),
    #   edge_attr 仍为纯几何 [|tan_half|, len_ratio]。
    for kp_idx, lines_at_kp in point_to_lines.items():
        if len(lines_at_kp) < 2:
            continue
        _lines_sorted = sorted(set(lines_at_kp))
        for _i in range(len(_lines_sorted)):
            for _j in range(_i + 1, len(_lines_sorted)):
                _a, _b = _lines_sorted[_i], _lines_sorted[_j]
                _pair = (min(_a, _b), max(_a, _b))
                if _pair in edge_data:
                    continue  # 已建边 (环序相邻, 已有 region 语义)
                edge_data[_pair] = {
                    'region_a': -1,   # 标记: 无共享 region (非环序相邻的全连接补充边)
                    'region_b': -1,
                    'keypoints': [kp_idx],
                    'l_a_raw': _a,
                    'l_b_raw': _b,
                }

    # 基于像素邻接的补边：NumPy 分片并行扫描 4 个独立方向，代替每像素 8 邻居循环。
    for _a,_b in sorted(skeleton_contact_pairs(line_label_map,skeleton_bin)):
        if (_a,_b) not in edge_data:
            edge_data[(_a,_b)] = {
                'region_a':-1,'region_b':-1,'keypoints':[],
                'l_a_raw':_a,'l_b_raw':_b,
            }

    # --- Region 邻接 (R-R 边) + Region 特征 ---
    # R-R 邻接: 共享 L-L 边的两侧 region 互为邻居
    rr_pairs = set()
    for edata in edge_data.values():
        r1, r2 = edata['region_a'], edata['region_b']
        if r1 != r2:
            pair = (min(r1, r2) - 1, max(r1, r2) - 1)  # 0-indexed
            rr_pairs.add(pair)

    edge_index_rr_list = []
    for r1, r2 in rr_pairs:
        edge_index_rr_list.append([r1, r2])
        edge_index_rr_list.append([r2, r1])

    if edge_index_rr_list:
        edge_index_rr_t = torch.tensor(edge_index_rr_list, dtype=torch.long).t()
    else:
        edge_index_rr_t = torch.zeros((2, 0), dtype=torch.long)

    # --- 构建边张量 ---
    edge_index_ll = []
    edge_attr_ll = []
    edge_relation_mask = []
    edge_region_a = []
    edge_region_b = []
    edge_label_a = []
    edge_label_b = []
        # 颜色 target
    color_to_id = {v: k for k, v in color_palette.items()}
    clean_target = target_rgb.copy()
    same_color = target_rgb[..., 3] == 0   # 透明 = 未上色(背景)
    clean_target[same_color] = [0, 0, 0, 0]

    # 预计算每个 region 的 target label
    region_labels = {}
    # 向量化: 一次性取出所有 region 像素并按 region id 分组, 替代逐 region 全图 np.where
    #   (原 O(num_regions × H×W) → O(H×W)); 组内像素保持光栅序 (stable argsort),
    #   众数平局取"光栅序最早出现色"与原逐 region 循环完全等价 (已 verify_region_preprocess PASS).
    _m_flat = markers.ravel()
    # 计算 region 众数时剔除线稿笔触像素 (原始线稿不透明处), 只统计区域内部的上色像素,
    # 避免粗笔触边缘的线色(黑)污染众数, 把 region 误判成线色。
    _line_mask = (lineart_rgb[..., 3] > 0).ravel()
    _pos = (_m_flat > 0) & (~_line_mask)
    _rids = _m_flat[_pos]
    _cols = clean_target.reshape(-1, 4)[_pos]
    _packed = (np.asarray(_cols[:, 0], dtype=np.int64) << 24) | (np.asarray(_cols[:, 1], dtype=np.int64) << 16) | (np.asarray(_cols[:, 2], dtype=np.int64) << 8) | np.asarray(_cols[:, 3], dtype=np.int64)
    _order = np.argsort(_rids, kind='stable')
    _rids_s = _rids[_order]
    _packed_s = _packed[_order]
    # _rids_s 已按 stable argsort 排序。用差分切分得到"每个 region 首个像素"的起点, 在结果上
    # 与 np.unique(_rids_s, return_index=True) 完全等价; 但避免对已排序数组再做一次 mergesort
    # argsort —— 那会额外分配一个与 _rids_s 等大的 int64 置换数组 (5.4M 像素 ≈ 43 MiB), 在
    # 像素级大图 / 进程累计内存高时触发 _ArrayMemoryError (实测 numpy '_ArrayMemoryError').
    # 差分只需 1/8 尺寸的布尔临时数组 + 变长起点索引, 峰值分配显著更低。
    # 差分切分: _rids_s 非空时前置 0; 空时与 np.unique 一致返回空
    if _rids_s.size:
        _start = np.concatenate(([0], np.flatnonzero(_rids_s[1:] != _rids_s[:-1]) + 1))
    else:
        _start = np.zeros(0, dtype=np.intp)
    _uniq_r = _rids_s[_start]
    _ends = np.append(_start[1:], len(_rids_s))
    for _k in range(len(_uniq_r)):
        _r = int(_uniq_r[_k])
        _s = int(_start[_k]); _e = int(_ends[_k])
        _seg = _packed_s[_s:_e]
        _vals, _counts = np.unique(_seg, return_counts=True)
        _max_c = int(_counts.max())
        _cand = _vals[_counts == _max_c]
        if _cand.size == 1:
            _mcp = int(_cand[0])
        else:
            _first_idx = np.array([int(np.argmax(_seg == _v)) for _v in _cand])
            _mcp = int(_cand[int(np.argmin(_first_idx))])
        _mc = ((_mcp >> 24) & 0xFF, (_mcp >> 16) & 0xFF, (_mcp >> 8) & 0xFF, _mcp & 0xFF)
        region_labels[_r] = color_to_id.get(_mc, 0)
    # 鲁棒: 理论上 label 后每个 region 都有像素, 但兜底缺失 region
    for _r_id in range(1, num_regions + 1):
        if _r_id not in region_labels:
            region_labels[_r_id] = 0

    # --- 色线连通域 (按线色分别做 8 连通分量) ---
    color_px_sum = {}
    for l_id in range(N_l):
        ci = int(line_info[l_id]['color_index'])
        color_px_sum[ci] = color_px_sum.get(ci, 0) + int(line_info[l_id]['pixel_count'])
    main_line_color = max(color_px_sum, key=color_px_sum.get) if color_px_sum else -1

    # 按线色分组 (跳过主线色)
    lines_by_color = {}
    for l_id in range(N_l):
        ci = int(line_info[l_id]['color_index'])
        if ci == main_line_color:
            continue
        lines_by_color.setdefault(ci, []).append(l_id)

    palette_rgb_arr = np.array(list(color_palette.values()), dtype=np.float32)
    palette_id_arr = np.array(list(color_palette.keys()), dtype=np.int64)

    # 调色板 packed 查找表 (与原 _comp_target_label 判据完全等价)
    _pal_packed = set()
    _pal_id_of = {}
    for _t, _idv in color_to_id.items():
        _pr, _pg, _pb, _pa = int(_t[0]), int(_t[1]), int(_t[2]), int(_t[3])
        _pk = (_pr << 24) | (_pg << 16) | (_pb << 8) | _pa
        _pal_packed.add(_pk)
        _pal_id_of[_pk] = int(_idv)

    def _comp_target_label_packed(seg):
        """等价原 _comp_target_label: 在 packed RGB 段中找多数调色板色;
        无调色板色则取众数最近调色板色。平票取光栅序最先出现者 (Counter 稳定序一致)。"""
        _vals, _cnts = np.unique(seg, return_counts=True)
        _in_pal = np.array([int(v) in _pal_packed for v in _vals])
        if _in_pal.any():
            _max_c = int(_cnts[_in_pal].max())
            _cands = _vals[_in_pal & (_cnts == _max_c)]
            _first = np.array([int(np.argmax(seg == v)) for v in _cands])
            _win = int(_cands[int(np.argmin(_first))])
            return _pal_id_of[_win]
        if palette_rgb_arr.size == 0:
            return 0
        _max_c = int(_cnts.max())
        _cands = _vals[_cnts == _max_c]
        _first = np.array([int(np.argmax(seg == v)) for v in _cands])
        _mode = int(_cands[int(np.argmin(_first))])
        _arr = np.array([(_mode >> 24) & 0xFF, (_mode >> 16) & 0xFF, (_mode >> 8) & 0xFF, _mode & 0xFF], dtype=np.float32)
        _d = np.sum((palette_rgb_arr - _arr) ** 2, axis=1)
        return int(palette_id_arr[int(np.argmin(_d))])

    line_comp_map = np.full((h, w), -1, dtype=np.int64)
    comp_target = {}
    global_comp_next = 0
    for ci, l_ids in lines_by_color.items():
        # 收集该色全部 line 像素坐标 (一次性 concat, 避免逐条 append 到全图 mask)
        _px_parts, _py_parts = [], []
        for l_id in l_ids:
            bfs_px, bfs_py = line_info[l_id]['bfs_order']
            _px_parts.append(bfs_px)
            _py_parts.append(bfs_py)
        if not _px_parts:
            continue
        _px = np.concatenate(_px_parts).astype(np.int64)
        _py = np.concatenate(_py_parts).astype(np.int64)
        mask_c = np.zeros((h, w), dtype=np.uint8)
        mask_c[_py, _px] = 1
        lbl = sk_label(mask_c, connectivity=2)  # 8 连通
        # 一次性取出所有像素并按 label 稳定分组 (保留光栅序, 平票 tie-break 等价)
        _ys, _xs = np.nonzero(mask_c)
        _rgb = target_rgb[_ys, _xs]
        _packed = (np.asarray(_rgb[:, 0], dtype=np.int64) << 24) | \
                  (np.asarray(_rgb[:, 1], dtype=np.int64) << 16) | \
                  (np.asarray(_rgb[:, 2], dtype=np.int64) << 8) | \
                  np.asarray(_rgb[:, 3], dtype=np.int64)
        _lk = lbl[_ys, _xs]
        _order = np.argsort(_lk, kind='stable')
        _lk_s = _lk[_order]; _pk_s = _packed[_order]
        _ys_s = _ys[_order]; _xs_s = _xs[_order]
        _uniq, _start = np.unique(_lk_s, return_index=True)
        _ends = np.append(_start[1:], len(_lk_s))
        for _k in range(len(_uniq)):
            _s = int(_start[_k]); _e = int(_ends[_k])
            # 线色标签 = 该线在彩图中"被涂上的颜色", 不是 region 填充色。
            #   只在**不透明**像素上取众数: 透明 = 这条线没有被重涂, 不应参与投票
            #   (否则一条被部分上色的线会因为"透明占多数"被标成未涂)。
            #   整段全透明时退回主线色语义 (label 0 = 保留线稿原色, 不重涂)。
            _seg = _pk_s[_s:_e]
            _opaque_seg = _seg[(_seg & np.int64(0xFF)) > 0]
            comp_target[global_comp_next] = (
                _comp_target_label_packed(_opaque_seg) if _opaque_seg.size else 0)
            line_comp_map[_ys_s[_s:_e], _xs_s[_s:_e]] = global_comp_next
            global_comp_next += 1
    num_line_comps = global_comp_next

    # 每条线归属连通域 + target label (主线色 → 非色线, 丢弃)
    line_comp_id_list = []
    line_color_label_list = []
    is_color_line_list = []
    for l_id in range(N_l):
        ci = int(line_info[l_id]['color_index'])
        if ci == main_line_color:
            line_comp_id_list.append(-1)
            line_color_label_list.append(0)
            is_color_line_list.append(False)
            continue
        bfs_px, bfs_py = line_info[l_id]['bfs_order']
        pix_comps = line_comp_map[bfs_py, bfs_px]
        pix_comps = pix_comps[pix_comps >= 0]
        if pix_comps.size == 0:
            line_comp_id_list.append(-1)
            line_color_label_list.append(0)
            is_color_line_list.append(False)
            continue
        vals, cnts = np.unique(pix_comps, return_counts=True)
        g = int(vals[int(np.argmax(cnts))])
        line_comp_id_list.append(g)
        line_color_label_list.append(comp_target[g])
        is_color_line_list.append(True)

    def _tan_half(l_src: int, l_dst: int):
        """共享交点优先用分支切向计算接续弯角；无共享交点则退回全局角度。

        两个切向均从 KP 向外，先反转 source 形成“入射 → 出射”；
        直线续接 tan(angle/2)=0、90° 转弯约 1。保持原 Edge 维度不变。
        """
        ia,ib=line_info[l_src],line_info[l_dst]
        common_kps=set(ia['connected_kps']) & set(ib['connected_kps'])
        options=[]
        for ki in common_kps:
            va=path_dir_map.get((l_src,ki));vb=path_dir_map.get((l_dst,ki))
            ca=ia['tangent_confidence'].get(ki,0)
            cb=ib['tangent_confidence'].get(ki,0)
            if va is None or vb is None or min(ca,cb)<0.15:
                continue
            curvature=abs(edge_tan_half((-va[0],-va[1]),vb))
            options.append((min(ca,cb),-curvature,curvature))
        if options:
            return max(options)[2]
        return abs(edge_tan_half(ia['feat_dir'],ib['feat_dir']))

    for (l_a, l_b), edata in edge_data.items():
        r_a = edata['region_a']
        r_b = edata['region_b']

        # 固定方向: region_a 总是所属线两侧中面积更大的区域, region_b 是更小的。
        # 补充全连接边 (非环序相邻) region_a/b = -1 (无共享 region), 跳过面积排序。
        if r_a >= 0 and r_b >= 0 and region_areas.get(r_a, 0) < region_areas.get(r_b, 0):
            r_a, r_b = r_b, r_a

        # 边属性: 2D (1D 无向正切半角 + 1D 弧长比), 均对 a↔b 对称
        # 1) 局部转角 |tan(α/2)|: 共享交点使用一入一出的切向，其他边回退全局方向
        tan_half = abs(_tan_half(l_a, l_b))

        # 2) 弧长比 (尺度不变, 沿用原计算)
        len_a = line_info[l_a]['path_len_raw']
        len_b = line_info[l_b]['path_len_raw']
        len_ratio = math.log(max(len_a, len_b) / (min(len_a, len_b) + 1e-6) + 1e-6)

        # (color_agree 经 color_embed 节点特征、path_coherence 经 cont 组内边已注入, 不再重复)

        # edge_feat (2D): [|tan_half|, len_ratio]
        edge_feat = [tan_half, len_ratio]

        # Region 标签 (补充边无共享 region → label 0 / region 索引 -1)
        if r_a >= 0 and r_b >= 0:
            label_a = region_labels.get(r_a, 0)
            label_b = region_labels.get(r_b, 0)
            edge_region_a.append(r_a - 1)  # 大区域侧, 0-indexed
            edge_region_b.append(r_b - 1)  # 小区域侧
        else:
            label_a = 0
            label_b = 0
            edge_region_a.append(-1)
            edge_region_b.append(-1)

        # 无向边: 每条 L-L 边只存一次 (消息传递前在 backbone.forward 内展开为双向)
        edge_index_ll.append([l_a, l_b])
        edge_attr_ll.append(edge_feat)
        pair_key = (min(l_a, l_b), max(l_a, l_b))
        edge_relation_mask.append([
            float(pair_key in ring_pairs),
            float(pair_key in corner_pairs or pair_key in cont_pairs),
            float(r_a >= 0 and r_b >= 0 and r_a != r_b),
        ])
        edge_label_a.append(label_a)
        edge_label_b.append(label_b)

    if not edge_index_ll:
        return None

    edge_index_ll_t = torch.tensor(edge_index_ll, dtype=torch.long).t()
    edge_attr_ll_t = torch.tensor(edge_attr_ll, dtype=torch.float32)
    edge_relation_mask_t = torch.tensor(edge_relation_mask, dtype=torch.float32)
    edge_region_a_t = torch.tensor(edge_region_a, dtype=torch.long)
    edge_region_b_t = torch.tensor(edge_region_b, dtype=torch.long)
    edge_label_a_t = torch.tensor(edge_label_a, dtype=torch.long)
    edge_label_b_t = torch.tensor(edge_label_b, dtype=torch.long)

    # --- 局部密度 (线长累加返回率) 1D 线特征: 邻居弦长均值 / (均值 + 自身弦长), 孤立线=0 ---
    # 须在 edge_index_ll 定稿后计算 (图级二阶 pass); 复用现有邻接, 不新增提取.
    # 见 compute_line_frt: frt = mean(chord_neighbor) / (mean(chord_neighbor) + chord_i), SE(2) 不变.
    frt_arr = compute_line_frt(edge_index_ll,
                               np.asarray(chords, dtype=np.float64))
    # --- rel_arc_len 线特征: 保号 log1p 低端放大 (与 _signed_log1p 同口径) ---
    # 原 np.log(x + 1e-6) 在低端 (x<<1) 坍缩到 0, 改用 /FEAT_LOG_EPS 低端放大。
    avg_nbr_chord = compute_avg_neighbor_chord(edge_index_ll,
                                                np.asarray(chords, dtype=np.float64))
    arc_arr = np.maximum(np.asarray(arc_lens, dtype=np.float64), 1e-6)
    ratio = arc_arr / np.maximum(avg_nbr_chord, 1e-6)
    rel_arc_len_arr = np.sign(ratio) * np.log1p(np.abs(ratio) / FEAT_LOG_EPS)
    X_l = torch.cat([
        X_l,
        torch.tensor(frt_arr, dtype=X_l.dtype).unsqueeze(1),
        torch.tensor(rel_arc_len_arr, dtype=X_l.dtype).unsqueeze(1),
    ], dim=1)

    # --- 全图 target (region → color, 用于投票推理) ---
    target_y = torch.tensor([region_labels.get(r_id, 0)
                             for r_id in range(1, num_regions + 1)],
                            dtype=torch.long)

    if debug_prefix is not None:
        n_edges = len(edge_data)
        n_unconnected = sum(1 for li in range(N_l)
                           if li not in set(e[0] for e in edge_data) | set(e[1] for e in edge_data))
        print(f"  [v6 L-L图] Lines={N_l}, L-L边={n_edges}, "
              f"Regions={num_regions}, 未连接线={n_unconnected}")

    # ==========================================================
    # --- 两侧区域相同的Line (内部线) 保留, 不删除 ---
    # ==========================================================
    # 改进: 原先 line_region_a == line_region_b 的线 (只邻接一个区域) 会被整条删除。
    # 现在保留它们作为图节点 —— 仍参与消息传递, 并在 region 投票中为两侧(同一)region
    # 提供颜色证据 (允许投票)。仅关闭过滤 (keep 全 True), 下方原删除/索引重映射分支
    # 不再执行, 各 line 张量/索引保持不变。
    keep_mask_np = np.ones(N_l, dtype=bool)
    if debug_prefix is not None:
        _n_same = int(sum(1 for i in range(N_l)
                          if line_region_a_list[i] == line_region_b_list[i]))
        if _n_same > 0:
            print(f"  [v6] 保留 {_n_same} 条两侧同 region 的Line (参与投票), 共 {N_l} 条")
    if not keep_mask_np.all():
        keep_indices_np = np.where(keep_mask_np)[0]
        n_removed = N_l - len(keep_indices_np)

        # 旧索引 → 新索引映射
        old_to_new = np.full(N_l, -1, dtype=np.int64)
        old_to_new[keep_indices_np] = np.arange(len(keep_indices_np))

        # 重映射 corner_pairs 到新索引 (供笔画粗化: 过滤后索引已变)
        if _abl_stroke_coarse:
            _remapped = set()
            for (a, b) in corner_pairs:
                na, nb = int(old_to_new[a]), int(old_to_new[b])
                if na >= 0 and nb >= 0 and na != nb:
                    _remapped.add((min(na, nb), max(na, nb)))
            corner_pairs = _remapped
            _remapped_c = set()
            for (a, b) in cont_pairs:
                na, nb = int(old_to_new[a]), int(old_to_new[b])
                if na >= 0 and nb >= 0 and na != nb:
                    _remapped_c.add((min(na, nb), max(na, nb)))
            cont_pairs = _remapped_c
            _remapped_r = set()
            for (a, b) in ring_pairs:
                na, nb = int(old_to_new[a]), int(old_to_new[b])
                if na >= 0 and nb >= 0 and na != nb:
                    _remapped_r.add((min(na, nb), max(na, nb)))
            ring_pairs = _remapped_r

        # 保存旧的line_info
        old_line_info = line_info

        # 1. 重建线级张量
        keep_t = torch.from_numpy(keep_indices_np.copy()).long()
        X_l = X_l[keep_t]
        color_index_t = color_index_t[keep_t]

        # 2. 重建 line_info
        line_info = {new_i: old_line_info[int(old_i)]
                     for new_i, old_i in enumerate(keep_indices_np)}

        # 3. 重建 point_to_lines
        for kp in point_to_lines:
            point_to_lines[kp] = [int(old_to_new[l]) for l in point_to_lines[kp]
                                  if old_to_new[l] >= 0]

        # 3b. 重建 keypoint_line_order (关键点相接线环序, 同步重映射线索引)
        for kp in keypoint_line_order:
            keypoint_line_order[kp] = [int(old_to_new[l]) for l in keypoint_line_order[kp]
                                       if old_to_new[l] >= 0]

        # 4. 重建 line_label_map (像素 → 新line_idx)
        line_label_map = np.full((h, w), -1, dtype=np.int32)
        for new_i, old_i in enumerate(keep_indices_np):
            bfs_px, bfs_py = old_line_info[int(old_i)]['bfs_order']
            line_label_map[bfs_py, bfs_px] = new_i
        # 关键点邻域重新填充 (使用旧索引查询path_dir_map)
        for kp_idx, lines_at_kp in point_to_lines.items():
            if not lines_at_kp:
                continue
            kx, ky = keypoints[kp_idx]
            for dy_r, dx_r in ring_cw:
                ny, nx = ky + dy_r, kx + dx_r
                if (0 <= ny < h and 0 <= nx < w
                        and skeleton_bin[ny, nx] and line_label_map[ny, nx] < 0):
                    best_l, best_dot = -1, -1.0
                    for l_idx_new in lines_at_kp:
                        l_idx_old = int(keep_indices_np[l_idx_new])
                        v = path_dir_map.get((l_idx_old, kp_idx))
                        if v is None:
                            continue
                        vx, vy = v
                        vnorm = math.hypot(vx, vy)
                        if vnorm < 1e-6:
                            continue
                        rnorm = math.hypot(dx_r, dy_r)
                        dot = (vx * dx_r + vy * dy_r) / (vnorm * rnorm)
                        if dot > best_dot:
                            best_dot = dot
                            best_l = l_idx_new
                    if best_l >= 0:
                        line_label_map[ny, nx] = best_l
                    else:
                        line_label_map[ny, nx] = lines_at_kp[0]

        # 5. 过滤L-L边张量 (重映射索引)
        if edge_index_ll_t.size(1) > 0:
            ei_np = edge_index_ll_t.numpy()
            e_keep = keep_mask_np[ei_np[0]] & keep_mask_np[ei_np[1]]
            e_keep_idx = np.where(e_keep)[0]
            edge_index_ll_t = torch.from_numpy(old_to_new[ei_np[:, e_keep]].copy()).long()
            e_keep_t = torch.from_numpy(e_keep_idx.copy()).long()
            edge_attr_ll_t = edge_attr_ll_t[e_keep_t]
            edge_relation_mask_t = edge_relation_mask_t[e_keep_t]
            edge_region_a_t = edge_region_a_t[e_keep_t]
            edge_region_b_t = edge_region_b_t[e_keep_t]
            edge_label_a_t = edge_label_a_t[e_keep_t]
            edge_label_b_t = edge_label_b_t[e_keep_t]

        # 7. 过滤line级列表
        line_region_a_list = [line_region_a_list[int(i)] for i in keep_indices_np]
        line_region_b_list = [line_region_b_list[int(i)] for i in keep_indices_np]
        line_comp_id_list = [line_comp_id_list[int(i)] for i in keep_indices_np]
        line_color_label_list = [line_color_label_list[int(i)] for i in keep_indices_np]
        is_color_line_list = [is_color_line_list[int(i)] for i in keep_indices_np]

        # 8. 更新N_l
        N_l = len(keep_indices_np)

        if N_l == 0 or edge_index_ll_t.size(1) == 0:
            return None

        if debug_prefix is not None:
            print(f"  [v6] 删除 {n_removed} 条两侧区域相同的Line, 剩余 {N_l} 条")

    # 过滤/重映射之后用同一工具补全实际 8 连通接触边，避免重复的 Python 像素循环。
    _existing_ll=set(zip(edge_index_ll_t[0].tolist(),edge_index_ll_t[1].tolist()))
    _extra_pairs=skeleton_contact_pairs(line_label_map,skeleton_bin)-_existing_ll
    if _extra_pairs:
        _extra_list = sorted(_extra_pairs)
        _extra_ei = []
        _extra_attr = []
        _extra_rmask = []
        for _la, _lb in _extra_list:
            _tan = abs(edge_tan_half(line_info[_la]['feat_dir'],
                                     line_info[_lb]['feat_dir']))
            _la_len = line_info[_la]['path_len_raw']
            _lb_len = line_info[_lb]['path_len_raw']
            _len_ratio = math.log(max(_la_len, _lb_len) / (min(_la_len, _lb_len) + 1e-6) + 1e-6)
            _extra_ei.append([_la, _lb])
            _extra_attr.append([_tan, _len_ratio])
            _extra_rmask.append([0.0, 0.0, 0.0])  # 8连通接触补边: 无环序/笔画/region语义
        _extra_ei_t = torch.tensor(_extra_ei, dtype=torch.long).t()
        _extra_attr_t = torch.tensor(_extra_attr, dtype=torch.float32)
        _extra_rmask_t = torch.tensor(_extra_rmask, dtype=torch.float32)
        edge_index_ll_t = torch.cat([edge_index_ll_t, _extra_ei_t], dim=1)
        edge_attr_ll_t = torch.cat([edge_attr_ll_t, _extra_attr_t], dim=0)
        edge_relation_mask_t = torch.cat([edge_relation_mask_t, _extra_rmask_t], dim=0)
        edge_region_a_t = torch.cat([edge_region_a_t,
                                     torch.full((len(_extra_list),), -1, dtype=torch.long)], dim=0)
        edge_region_b_t = torch.cat([edge_region_b_t,
                                     torch.full((len(_extra_list),), -1, dtype=torch.long)], dim=0)
        edge_label_a_t = torch.cat([edge_label_a_t,
                                    torch.zeros(len(_extra_list), dtype=torch.long)], dim=0)
        edge_label_b_t = torch.cat([edge_label_b_t,
                                    torch.zeros(len(_extra_list), dtype=torch.long)], dim=0)

    # --- Line 第5维特征 boundary_share: 图结构走环, 分母只用有共享边的环周长 ---
    # 对于每条线两侧的 region pair (rA, rB):
    #   1) 从 L-L 图 edge_index_ll_t 提取每个 region 的线环路邻接 (contour_adj)
    #   2) BFS 仅走包含共享边的那个环 —— 有洞时外环/内环不会混合, 避免被洞内环稀释
    #   3) boundary_share = (shared_px / loop_rA_px + shared_px / loop_rB_px) / 2, ∈ [0, 1]
    # ≈0 → 两区域仅微小邻接; ≈1 → 共享边即为整个环

    # Step 1: region → 边界线集合
    lines_of_region = {}
    for l_id in range(N_l):
        for r in (line_region_a_list[l_id], line_region_b_list[l_id]):
            if r not in lines_of_region:
                lines_of_region[r] = set()
            lines_of_region[r].add(l_id)

    # Step 2: 从 edge_index_ll_t (双向有向边) 提取无向唯一边, 为每对邻接线查找公共 region
    ei_np = edge_index_ll_t.numpy()
    contour_adj = {r: {} for r in lines_of_region}   # {region: {line: [邻接线...]}}
    _seen_ll = set()
    for e in range(ei_np.shape[1]):
        la, lb = int(ei_np[0, e]), int(ei_np[1, e])
        if la == lb:
            continue
        upair = (la, lb) if la < lb else (lb, la)
        if upair in _seen_ll:
            continue
        _seen_ll.add(upair)
        common = ({line_region_a_list[la], line_region_b_list[la]}
                  & {line_region_a_list[lb], line_region_b_list[lb]})
        for r in common:
            contour_adj[r].setdefault(la, []).append(lb)
            contour_adj[r].setdefault(lb, []).append(la)

    # Step 3: 预计算每条线的 boundary_share (按共享边所在环的周长)
    line_px = np.array([line_info[l]['pixel_count'] for l in range(N_l)], dtype=np.float64)
    boundary_share_by_line = {}
    pair_to_lines = {}
    for l_id in range(N_l):
        key = (min(line_region_a_list[l_id], line_region_b_list[l_id]),
               max(line_region_a_list[l_id], line_region_b_list[l_id]))
        pair_to_lines.setdefault(key, []).append(l_id)

    for (ra, rb), all_shared in pair_to_lines.items():
        if ra == rb:
            continue
        shared_set = set(all_shared)
        handled = set()

        # 多个环路分别计算 (防两个 region 通过不同环多次邻接的罕见情形)
        for start_l in all_shared:
            if start_l in handled:
                continue

            # BFS ra: 找包含 start_l 的环
            visited_ra = set()
            stack = [start_l]
            while stack:
                cur = stack.pop()
                if cur in visited_ra:
                    continue
                visited_ra.add(cur)
                for nbr in contour_adj.get(ra, {}).get(cur, ()):
                    if nbr not in visited_ra:
                        stack.append(nbr)
            loop_ra_px = max(float(np.sum(line_px[np.asarray(sorted(visited_ra), dtype=np.int64)])), 1.0)

            # 该环内的共享线 (≠ pair 的全部共享线, 只取在当前环中的)
            seg_shared = [l for l in visited_ra if l in shared_set]
            seg_shared_px = float(np.sum(line_px[np.asarray(seg_shared, dtype=np.int64)]))

            # BFS rb: 找包含 start_l 的环
            visited_rb = set()
            stack = [start_l]
            while stack:
                cur = stack.pop()
                if cur in visited_rb:
                    continue
                visited_rb.add(cur)
                for nbr in contour_adj.get(rb, {}).get(cur, ()):
                    if nbr not in visited_rb:
                        stack.append(nbr)
            loop_rb_px = max(float(np.sum(line_px[np.asarray(sorted(visited_rb), dtype=np.int64)])), 1.0)

            bs = float(min((seg_shared_px / loop_ra_px
                            + seg_shared_px / loop_rb_px) / 2.0, 1.0))
            for l in seg_shared:
                boundary_share_by_line[l] = bs
            handled.update(seg_shared)

    # Step 4: 每线查表组装
    boundary_share_list = [boundary_share_by_line.get(l_id, 0.0) for l_id in range(N_l)]
    X_l = torch.cat([
        X_l,
        torch.tensor(boundary_share_list, dtype=X_l.dtype).unsqueeze(1),
    ], dim=1)

    # --- Region 线色几何特征 (逐像素, 原点=软 pole, 尺度=sqrt(area)) ---
    # 逐像素(非线中点): 颜色长度=该色边界像素数/sqrt(area)(长线权重更大); 方位=单位向量均值(两侧不抵消)。
    # 三要素: 原点=软 pole(平移不变) | 尺度=sqrt(area)(缩放不变) | 轴=图像 x/y 直角轴(旋转等变)。
    # 尺度须用 sqrt(area) 而非 content_scale: 后者压到 ~0.06 使方位信号消失 (p50 0.026 vs 0.41)。完全向量化。
    nc = MAX_LINE_COLORS   # 固定为常量 (而非 len(line_palette)): region_line_color_geo 维度须数据集无关,
                           # 使共享 backbone 的 region_color_compress 输入维度恒定。超出部分的线色桶恒为 0。
    N_r = int(num_regions)
    region_line_color_geo = np.zeros((N_r, nc * 3), dtype=np.float64)
    if N_r > 0 and N_l > 0:
        pix_x, pix_y, pix_reg, pix_col = [], [], [], []
        for l_id in range(N_l):
            bfs_px, bfs_py = line_info[l_id]['bfs_order']
            if len(bfs_px) == 0:
                continue
            # 路径为保证连通可能回溯：每个物理线像素仅计一次颜色几何权重。
            _uniq_ix=np.sort(np.unique(bfs_py.astype(np.int64)*w+bfs_px,return_index=True)[1])
            bfs_px,bfs_py=bfs_px[_uniq_ix],bfs_py[_uniq_ix]
            ci = int(color_index_t[l_id])
            # 线是两个 region 的公共边界 → 每个像素对两侧 region 各贡献一次。
            # 用 set 去重: 两侧同 region 的线 (保留未删) 只贡献一次, 不会重复计入同一 region。
            for r in {line_region_a_list[l_id], line_region_b_list[l_id]}:
                if not (0 <= r < N_r):
                    continue
                pix_x.append(bfs_px)
                pix_y.append(bfs_py)
                pix_reg.append(np.full(len(bfs_px), r, dtype=np.int64))
                pix_col.append(np.full(len(bfs_px), ci, dtype=np.int64))
        if pix_x:
            ax = np.concatenate(pix_x).astype(np.float64)
            ay = np.concatenate(pix_y).astype(np.float64)
            ar = np.concatenate(pix_reg)
            ac = np.concatenate(pix_col)
            # 软 pole 原点: 1-based region_lc_origin → 0-based 查找表
            cent = np.zeros((N_r, 2), dtype=np.float64)
            for r_id in range(1, N_r + 1):
                cent[r_id - 1] = region_lc_origin.get(r_id, (origin_x, origin_y))
            ox = ax - cent[ar, 0]        # 原始像素偏移 → 区域核心
            oy = ay - cent[ar, 1]
            # (region, color) 展平成一维桶 → bincount 一次性聚合
            flat = ar * nc + ac
            cnt = np.bincount(flat, minlength=N_r * nc).astype(np.float64)
            # 每个桶按其所属 region 的自身尺度归一化 (第 2 档)
            rscale = np.sqrt(np.maximum(np.array(
                [region_areas.get(r + 1, 1.0) for r in range(N_r)],
                dtype=np.float64), 1.0))
            per_bucket_scale = np.repeat(rscale, nc)
            # [块 1] 颜色长度 = 该颜色的边界像素数 / 区域自身尺度
            region_line_color_geo[:, :nc] = (cnt / per_bucket_scale).reshape(N_r, nc)

            # [块 2][3] 方位 = 单位方向均值: 每个像素等权, 方向不受"离核心多远"影响
            #   → 区域两侧的边界像素不会正负抵消
            #   平滑单位化 (eps = 区域尺度的 DIR_EPS_FRAC): 恰好落在核心上的像素其方向
            #   本无定义, 未平滑时会被 1/norm 放大成随机方向
            #   单位方向本身是 O(1) 量, 故不再除以区域尺度
            eps_pix = np.maximum(0.5, LC_GEO_DIR_EPS * rscale[ar]) 
            nrm = np.sqrt(ox * ox + oy * oy + eps_pix * eps_pix)
            sx = np.bincount(flat, weights=ox / nrm, minlength=N_r * nc)
            sy = np.bincount(flat, weights=oy / nrm, minlength=N_r * nc)
            safe = np.maximum(cnt, 1.0)
            region_line_color_geo[:, nc:2 * nc] = (sx / safe).reshape(N_r, nc)
            region_line_color_geo[:, 2 * nc:] = (sy / safe).reshape(N_r, nc)

    region_line_color_geo_t = torch.tensor(region_line_color_geo, dtype=torch.float32)

    line_region_a_t = torch.tensor(line_region_a_list, dtype=torch.long)
    line_region_b_t = torch.tensor(line_region_b_list, dtype=torch.long)
    line_label_a_t = torch.tensor([region_labels.get(r + 1, 0) for r in line_region_a_list], dtype=torch.long)
    line_label_b_t = torch.tensor([region_labels.get(r + 1, 0) for r in line_region_b_list], dtype=torch.long)

    # --- 多点局部法向 Side canonicalization（含置信度诊断） ---
    # 方向的符号仍由 feat_dir(kp1→kp2) 决定，不更改 Side 分类器定义。
    # 多点投票只用于决定 A/B 与 +/-n 对应关系；不改变 Region 标签或网络维度。
    plus_is_a=[]
    side_orientation_confidence=[]
    for l_id in range(N_l):
        li=line_info[l_id]
        raw_dx,raw_dy=li['feat_dir']
        x_path,y_path=li['bfs_order']
        is_a,confidence=infer_line_plus_region_a(
            x_path,y_path,(raw_dx,raw_dy),
            line_region_a_list[l_id],line_region_b_list[l_id],markers)
        if confidence==0.0:
            # 与旧版兼容的单点 fallback，明确降低置信度；防短线没有双侧样本。
            mid_x,mid_y=li['mid']
            nx,ny=-raw_dy,raw_dx
            norm=math.hypot(nx,ny)
            if norm>1e-8:
                px=int(round(mid_x+2*nx/norm))
                py=int(round(mid_y+2*ny/norm))
                rid=int(markers[py,px]) if 0<=px<w and 0<=py<h else 0
                if rid==int(line_region_b_list[l_id])+1 and line_region_a_list[l_id]!=line_region_b_list[l_id]:
                    is_a=False
        plus_is_a.append(is_a)
        side_orientation_confidence.append(confidence)
        li['side_orientation_confidence']=confidence
    plus_is_a_t=torch.tensor(plus_is_a,dtype=torch.bool)
    side_confidence_t=torch.tensor(side_orientation_confidence,dtype=torch.float32)
    # --- 三大贡献源消融 (数据侧置零, 维度/结构不变) ---
    # h_l 线几何流：X_l 当前为 5D 几何/拓扑特征。
    #   消 h_l 时置零全部 5D；线色嵌入(color_index)由 color_embed 单独控制。
    if not _abl_line:
        X_l = torch.zeros_like(X_l)
    # h_r 区域线色几何流
    if not _abl_rcolor:
        region_line_color_geo_t = torch.zeros_like(region_line_color_geo_t)

    # --- 笔画级粗化结构 (CoRe-GNN 并行支路): 在最终过滤后的细图上计算 ---
    stroke_id_t = None
    edge_index_stroke_t = None
    edge_index_intra_stroke_t = None
    cont_stroke_id_t = None
    edge_index_cont_stroke_t = None
    edge_index_intra_cont_t = None
    edge_index_intra_ring_t = None   # 环序相邻线对 (仅 MRF pairwise transition 统计用, 不进粗支路)
    if _abl_stroke_coarse:
        (stroke_id_t, edge_index_stroke_t,
         cont_stroke_id_t, edge_index_cont_stroke_t,
         edge_index_intra_stroke_t, edge_index_intra_cont_t) = _compute_stroke_coarse(
            corner_pairs, cont_pairs, edge_index_ll_t, N_l)
        edge_index_intra_ring_t = _pairs_to_edge_index(ring_pairs)
        if debug_prefix is not None:
            num_strokes = int(stroke_id_t.max().item()) + 1 if stroke_id_t.numel() else 0
            num_cont = int(cont_stroke_id_t.max().item()) + 1 if cont_stroke_id_t.numel() else 0
            print(f"  [v6 笔画粗化] 细图 {N_l} 线 → {num_strokes} 笔画(角点), "
                  f"{num_cont} 连续笔画(方向), 笔画边 {edge_index_stroke_t.size(1)} 条")

    # --- 异构图 Line-Keypoint 二部图边 (显式关键点结构, 暂不参与训练) ---
    # 每条最终 line 两端各连一个关键点 (kp1/kp2 坐标 → 索引)。闭合环退化端点可能不在
    # keypoints 里, 用 get 安全跳过。L-L 边 = 此二部图的 Line-KP-Line 两跳 (见边构建)。
    line_kp_edges = []
    for l_idx in range(N_l):
        li = line_info[l_idx]
        for kp in (li.get('kp1'), li.get('kp2')):
            kp_i = kp_to_idx.get(kp)
            if kp_i is not None:
                line_kp_edges.append([l_idx, kp_i])
    if line_kp_edges:
        line_kp_edge_t = torch.tensor(line_kp_edges, dtype=torch.long).t()
    else:
        line_kp_edge_t = torch.zeros((2, 0), dtype=torch.long)

    # --- 位置嵌入 (E4): line 中点 / region 质心 相对骨架质心的相对坐标 ---
    # line_pos = (mid − origin)/content_scale; region_pos = (centroid − origin)/content_scale。
    # 平移/缩放不变, 旋转等变 (方向是动漫正立的强先验)。
    line_pos_list = []
    for l_idx in range(N_l):
        mid_x, mid_y = line_info[l_idx]['mid']
        line_pos_list.append([(mid_x - origin_x) / content_scale,
                              (mid_y - origin_y) / content_scale])
    line_pos_t = torch.tensor(line_pos_list, dtype=torch.float32)          # (N_l, 2)

    region_pos_list = []
    for r_id in range(1, N_r + 1):
        cx, cy = region_centroids.get(r_id, (origin_x, origin_y))
        region_pos_list.append([(cx - origin_x) / content_scale,
                                (cy - origin_y) / content_scale])
    region_pos_t = torch.tensor(region_pos_list, dtype=torch.float32)      # (N_r, 2)

    _perim = np.zeros(N_r + 1, dtype=np.float64)
    # 向量化统计每个 region 的边界像素数 (一次 O(H*W) 扫描)。
    #   原实现为 per-region: `_m = (markers == _r)` 全图布尔比较 + `binary_erosion(_m)` 全图腐蚀,
    #   复杂度 O(N_r × H×W)。region 数达 300~400 时需数分钟到数十分钟, 在 pixel-level 评估里
    #   表现为"卡死"(实测 417 region 时被中断在 binary_erosion)。
    #   与原实现严格等价: 4-连通腐蚀被移除的像素, 正是"4-邻域中存在不同 mark 或越界"的像素;
    #   binary_erosion 默认 border_value=0, 因此图像边缘的 region 像素同样算边界。
    if markers.size and N_r > 0:
        _diff = np.zeros(markers.shape, dtype=bool)
        _diff[:, :-1] |= (markers[:, :-1] != markers[:, 1:])
        _diff[:, 1:] |= (markers[:, 1:] != markers[:, :-1])
        _diff[:-1, :] |= (markers[:-1, :] != markers[1:, :])
        _diff[1:, :] |= (markers[1:, :] != markers[:-1, :])
        _diff[0, :] = True
        _diff[-1, :] = True
        _diff[:, 0] = True
        _diff[:, -1] = True
        _lab = markers[_diff]
        _lab = _lab[(_lab >= 1) & (_lab <= N_r)]
        if _lab.size:
            _perim[:] = np.bincount(_lab.astype(np.int64), minlength=N_r + 1)[:N_r + 1]
    _shape = np.zeros((N_r, 2), dtype=np.float32)
    for _r in range(1, N_r + 1):
        _a = float(max(int(region_areas.get(_r, 0)), 1))
        _p = float(max(int(_perim[_r]), 1))
        _shape[_r - 1, 0] = float(np.log(_a / area_median))
        _shape[_r - 1, 1] = float(min(4.0 * np.pi * _a / (_p * _p), 1.0))
    region_shape_feat_t = torch.tensor(_shape, dtype=torch.float32)

    _geom_diag['backend'] = CONT_PAIR_BACKEND
    _geom_diag['path_backend'] = LINE_PATH_BACKEND
    _geom_diag['n_cont_pairs'] = len(cont_pairs)
    _GEOM_DIAG_LAST.clear()
    _GEOM_DIAG_LAST.update(_geom_diag)

    return {
        "X_l": X_l,                          # (N_l, 5) Line 几何/拓扑特征
        "color_index": color_index_t,          # (N_l,) 线色类别索引 (long)
        "region_line_color_geo": region_line_color_geo_t,  # (N_r, nc*3) 逐像素线色几何特征
        "edge_index_ll": edge_index_ll_t,     # (2, E) 无向
        "edge_attr_ll": edge_attr_ll_t,       # (E, 2)
        "edge_relation_mask": edge_relation_mask_t,  # (E, 3): ring/stroke/region
        "edge_region_a": edge_region_a_t,     # (E,) 所属线的大区域侧, 0-indexed
        "edge_region_b": edge_region_b_t,     # (E,) 所属线的小区域侧, 0-indexed
        "edge_label_a": edge_label_a_t,       # (E,)
        "edge_label_b": edge_label_b_t,       # (E,)
        "line_region_a": line_region_a_t,     # (N_l,) per-line 大区域侧, 0-indexed
        "line_region_b": line_region_b_t,     # (N_l,) per-line 小区域侧, 0-indexed
        "line_label_a": line_label_a_t,       # (N_l,) per-line 大区域侧 label
        "line_label_b": line_label_b_t,       # (N_l,) per-line 小区域侧 label
        "line_plus_region_a": plus_is_a_t,    # (N_l,) bool: region_a 是否在 +n 几何法向侧
        "line_plus_region_a_confidence": side_confidence_t,  # (N_l,) 多点法向证据一致性, 仅诊断
        "line_comp_id": torch.tensor(line_comp_id_list, dtype=torch.long),   # (N_l,) 色线连通域全局id (-1=主线)
        "line_color_label": torch.tensor(line_color_label_list, dtype=torch.long),  # (N_l,) 色线 target 填色
        "is_color_line": torch.tensor(is_color_line_list, dtype=torch.bool), # (N_l,) 是否色线
        "num_line_comps": num_line_comps,     # 色线连通域总数
        "main_line_color": int(main_line_color),  # 主线色索引 (骨架像素最多者)
        "line_pixel_count": torch.tensor([line_info[l]['pixel_count'] for l in range(N_l)], dtype=torch.float32),
        "edge_index_rr": edge_index_rr_t,     # (2, R-R边数) 双向
        "line_pos": line_pos_t,               # (N_l, 2) line 中点相对骨架质心坐标
        "region_pos": region_pos_t,           # (N_r, 2) region 质心相对骨架质心坐标
        "region_shape_feat": region_shape_feat_t,
        "y": target_y,                        # (N_r,) 全图 region labels
        "markers": markers,
        "num_regions": num_regions,
        "keypoints": keypoints,
        "pos_scale": pos_scale,               # 仅调试/可视化用: 不参与任何特征归一化
        "area_95": area_95,
        "area_median": area_median,
        "origin": (origin_x, origin_y),
        "skeleton_rgb": skeleton_rgb,
        "skeleton_bin": skeleton_bin,
        "lineart_rgb": lineart_rgb,
        "unified_target": unified_target,     # 上色图按原始线稿边界去杂后的平铺色 (RGBA), 仅用于导出预处理彩图
        "line_label_map": line_label_map,     # 像素 → line_idx
        "line_info": line_info,
        "N_l": N_l,
        "keypoint_cn": torch.tensor(kp_cn_arr, dtype=torch.long),
        "keypoint_geom_pos": torch.tensor(kp_geometry_anchors,dtype=torch.float32),
        "keypoint_junction_cluster_size": torch.tensor(kp_junction_cluster_sizes,dtype=torch.long),           # (N_p,) crossing number
        "line_kp_edge_index": line_kp_edge_t,   # (2, E_lk) Line-Keypoint 二部图边
        "keypoint_line_order": keypoint_line_order,  # {kp_idx: [line_idx...]} 关键点相接线环序
        "stroke_id": stroke_id_t,             # (N_l,) 笔画归属 (紧凑索引), 仅 STROKE_COARSE_ENABLED 时非 None
        "edge_index_stroke": edge_index_stroke_t,  # (2, E_s) 笔画级双向边, 仅 STROKE_COARSE_ENABLED 时非 None
        "cont_stroke_id": cont_stroke_id_t,   # (N_l,) 连续笔画归属 (方向一致), 仅 STROKE_COARSE_ENABLED 时非 None
        "edge_index_cont_stroke": edge_index_cont_stroke_t,  # (2, E_cs) 连续笔画级双向边
        "edge_index_intra_stroke": edge_index_intra_stroke_t,  # (2, 2E) 笔画组内双向边
        "edge_index_intra_cont": edge_index_intra_cont_t,      # (2, 2E) 连续笔画组内双向边
        "edge_index_intra_ring": edge_index_intra_ring_t,      # (2, 2E) 环序相邻线对 (仅 MRF pairwise 统计)
    }

# ==================== 3. 数据集与 Collate ====================

def _process_single_graph(pair, line_palette, color_palette, cache_name, ablation=None):
    """多进程 worker: 处理单张图片构建图 (模块顶层, 可被 pickle)

    ablation: 显式传入三大消融开关 (spawn 下 worker 不继承主进程全局, 必须传参)。
    """
    idx, l_path, c_path = pair
    try:
        l_raw = cv2.imdecode(np.fromfile(l_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        c_raw = cv2.imdecode(np.fromfile(c_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if l_raw is None or c_raw is None:
            return None
        l_rgba = process_alpha(l_raw)
        l_rgba = prepare_lineart_for_graph(l_rgba)  # 微小闭区就近填墨，训练与推理一致
        c_rgba = process_alpha(c_raw)
        debug_pfx = f"{cache_name}_{idx}" if idx < 3 else None
        graph_data = extract_ll_graph(l_rgba, c_rgba, line_palette, color_palette,
                                      debug_prefix=debug_pfx, ablation=ablation)
        return graph_data
    except Exception as e:
        print(f"[警告] 处理 {os.path.basename(l_path)} 失败: {e}")
        return None


def _graph_cache_identity(line_paths,color_paths,line_palette,color_palette):
    """验证缓存是否来自当前输入文件和当前色号映射；避免静默使用过期图。"""
    def _paths(paths):
        out=[]
        for p in paths:
            p=os.path.abspath(os.fspath(p))
            try:
                st=os.stat(p)
                out.append((p,int(st.st_size),int(st.st_mtime_ns)))
            except OSError:
                out.append((p,None,None))
        return out
    return {
        'schema': 2,
        'graph_algorithm': GRAPH_CACHE_VERSION,
        'lineart': _paths(line_paths), 'ground_truth': _paths(color_paths),
        'line_palette': [(int(k),tuple(map(int,line_palette[k]))) for k in sorted(line_palette)],
        'color_palette': [(int(k),tuple(map(int,color_palette[k]))) for k in sorted(color_palette)],
        'stroke_coarse':bool(STROKE_COARSE_ENABLED),
        'line_stream':bool(ENABLE_LINE_STREAM),
        'rcolor_stream':bool(ENABLE_RCOLOR_STREAM),
    }


class LineartLLDataset(Dataset):
    def __init__(self, line_paths, color_paths, line_palette, color_palette,
                 cache_name="train", save_cache=True, triple_map=None,
                 line_color_to_fill=None, pair_transition=None):
        self.graphs = []
        self.triple_map = None
        self.num_triples = 0
        self.inv_triple_map = None
        self.line_color_to_fill = None
        self.pair_transition = pair_transition   # MRF pairwise transition 统计 (训练集自统计, val 复用)
        self.num_classes = len(color_palette)
        self._graph_cache_identity = _graph_cache_identity(
            line_paths,color_paths,line_palette,color_palette)
        cache_path = f"Cache/gnn_v6_{GRAPH_CACHE_VERSION}_{cache_name}.pkl"
        # 注意: 命中缓存时**不能**直接 return — 三元组映射 (triple_map) 与每条线的
        # triple_label 必须在每次构造时重新推导 (它们依赖当前 palette 与全部图)。
        # 早期版本在缓存命中处 return, 导致 num_triples 停留在初始值 0,
        # 模型输出 0 类 → CrossEntropy 触发 device-side assert (t < n_classes)。
        cache_loaded = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_graphs = pickle.load(f)
                    try:
                        cache_meta = pickle.load(f)
                    except EOFError:
                        cache_meta = None  # 无来源标识的旧缓存必须重新构建
                if cache_meta != self._graph_cache_identity:
                    print(f"[{cache_name}] 文件或调色板已变更，重建图缓存...")
                    self.graphs = []
                    os.remove(cache_path)
                elif (len(cached_graphs)>0 and cached_graphs[0]['X_l'].shape[1] == LINE_FEAT_DIM):
                    self.graphs = cached_graphs
                    print(f"[{cache_name}] 从缓存加载了 {len(self.graphs)} 个图 (v6 L-L)")
                    cache_loaded = True
                else:
                    print(f"[{cache_name}] 旧缓存维度不匹配, 重建...")
                    os.remove(cache_path)
                    self.graphs = []
            except Exception as e:
                print(f"[{cache_name}] 加载失败 ({e}), 重建...")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                self.graphs = []

        if not cache_loaded:
            self._build_graphs(line_paths, color_paths, line_palette, color_palette,
                               cache_name, save_cache, cache_path)

        # ---- 统计或使用传入的三元组映射 (缓存/新建都要执行) ----
        if triple_map is None:
            triple_set = set()
            for g in self.graphs:
                la = g['line_label_a'].numpy() if torch.is_tensor(g['line_label_a']) else np.array(g['line_label_a'])
                lb = g['line_label_b'].numpy() if torch.is_tensor(g['line_label_b']) else np.array(g['line_label_b'])
                lc = g['line_color_label'].numpy() if torch.is_tensor(g['line_color_label']) else np.array(g['line_color_label'])
                is_cl = g['is_color_line'].numpy() if torch.is_tensor(g['is_color_line']) else np.array(g['is_color_line'])
                main_col = g['main_line_color']
                for i in range(len(la)):
                    a = int(la[i]); b = int(lb[i])
                    c = int(lc[i]) if is_cl[i] else int(main_col)
                    # 无序目标: 两侧颜色对 (a,b) 规范化为 (min,max) — (a,b,c) 与 (b,a,c) 并为一类,
                    # 消除"大区域/小区域"槽位语义, 分类目标天然对称。
                    triple_set.add((min(a, b), max(a, b), c))
            sorted_triples = sorted(triple_set)
            self.triple_map = {t: idx for idx, t in enumerate(sorted_triples)}
            self.num_triples = len(self.triple_map)
            self.inv_triple_map = {idx: t for t, idx in self.triple_map.items()}
        else:
            self.triple_map = triple_map
            self.num_triples = len(self.triple_map)
            self.inv_triple_map = {idx: t for t, idx in self.triple_map.items()}

        # 投票线色硬约束的"合法集合"来源: 从训练集里色线 (color_index → comp_target 填色) 的
        # 出现关系推导。约束表只应从训练集统计 (val 经 line_color_to_fill 参数复用, 避免验证集泄漏)。
        if line_color_to_fill is None:
            self.line_color_to_fill = {}
            for g in self.graphs:
                ci = g['color_index'].numpy() if torch.is_tensor(g['color_index']) else np.array(g['color_index'])
                is_cl = g['is_color_line'].numpy() if torch.is_tensor(g['is_color_line']) else np.array(g['is_color_line'])
                lc = g['line_color_label'].numpy() if torch.is_tensor(g['line_color_label']) else np.array(g['line_color_label'])
                for i in range(len(ci)):
                    if is_cl[i]:
                        self.line_color_to_fill.setdefault(int(ci[i]), set()).add(int(lc[i]))
        else:
            self.line_color_to_fill = line_color_to_fill

        # 为每个图生成 triple_label
        for g in self.graphs:
            la = g['line_label_a'].numpy() if torch.is_tensor(g['line_label_a']) else np.array(g['line_label_a'])
            lb = g['line_label_b'].numpy() if torch.is_tensor(g['line_label_b']) else np.array(g['line_label_b'])
            lc = g['line_color_label'].numpy() if torch.is_tensor(g['line_color_label']) else np.array(g['line_color_label'])
            is_cl = g['is_color_line'].numpy() if torch.is_tensor(g['is_color_line']) else np.array(g['is_color_line'])
            main_col = g['main_line_color']
            labels = []
            for i in range(len(la)):
                a = int(la[i]); b = int(lb[i])
                c = int(lc[i]) if is_cl[i] else int(main_col)
                # 无序目标 key: (min,max,c), 与 triple_map 构建一致
                labels.append(self.triple_map.get((min(a, b), max(a, b), c), -1))
            g['triple_label'] = torch.tensor(labels, dtype=torch.long)

        # ---- 统计 MRF pairwise transition (训练集: 环序相邻线的 GT 色对共现) ----
        # 仅当未传入 pair_transition 时统计 (训练集自统计; val 复用 train 的, 避免验证集泄漏)。
        # 色对空间按 triple_id 顺序首次出现编号。
        if self.pair_transition is None:
            pair_order = []
            pair_id_of = {}
            for tid in range(self.num_triples):
                t = self.inv_triple_map[tid]
                key = (int(t[0]), int(t[1]))
                if key not in pair_id_of:
                    pair_id_of[key] = len(pair_order)
                    pair_order.append(key)
            num_pairs = len(pair_order)
            trans = np.zeros((num_pairs, num_pairs), dtype=np.float64)
            for g in self.graphs:
                la = g['line_label_a'].numpy() if torch.is_tensor(g['line_label_a']) else np.array(g['line_label_a'])
                lb = g['line_label_b'].numpy() if torch.is_tensor(g['line_label_b']) else np.array(g['line_label_b'])
                eir = g.get('edge_index_intra_ring')
                if eir is None:
                    continue
                eir_np = eir.numpy() if torch.is_tensor(eir) else np.asarray(eir)
                if eir_np.size == 0:
                    continue
                src = eir_np[0].astype(np.int64)
                dst = eir_np[1].astype(np.int64)
                keep = src < dst
                src, dst = src[keep], dst[keep]
                for i, j in zip(src.tolist(), dst.tolist()):
                    pai = pair_id_of.get((min(int(la[i]), int(lb[i])), max(int(la[i]), int(lb[i]))))
                    paj = pair_id_of.get((min(int(la[j]), int(lb[j])), max(int(la[j]), int(lb[j]))))
                    if pai is not None and paj is not None:
                        trans[pai, paj] += 1.0
                        trans[paj, pai] += 1.0   # 对称 (无向边)
            # Laplace 平滑 + 行归一化 → 转移概率 (float32)
            trans_smooth = trans + 1.0
            row_sum = trans_smooth.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            self.pair_transition = (trans_smooth / row_sum).astype(np.float32)

        if save_cache and not cache_loaded:
            os.makedirs("Cache", exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.graphs, f)
                pickle.dump(self._graph_cache_identity, f)
            print(f"[{cache_name}] 图缓存已保存（含输入/调色板指纹）→ {cache_path}")

        # --- 导出预处理图到 Cache/preprocessed/ ---
        #   1) 预处理彩图: 以原始线稿(非骨架)为边界构建连通域, 取众数平铺色; 线稿笔触处填黑。
        #   2) 预处理线稿: 骨架化后的线稿 (黑线白底)。
        os.makedirs("Cache/preprocessed", exist_ok=True)
        for i, g in enumerate(self.graphs):
            lineart_rgba = np.asarray(g["lineart_rgb"])
            line_mask = lineart_rgba[..., 3] > 0

            # --- 预处理彩图 (原始线稿边界 + 众数平铺色) ---
            u = g.get("unified_target")
            if u is not None:
                color_img = np.asarray(u)[..., :3].copy()
            else:
                # 旧缓存回退: 用 markers + y 重建 (骨架边界)
                markers = np.asarray(g["markers"])
                y = np.asarray(g["y"]).flatten()
                num_regions = int(g["num_regions"])
                color_lut = np.zeros((num_regions + 1, 3), dtype=np.uint8)
                for r_id in range(1, num_regions + 1):
                    c_idx = int(y[r_id - 1])
                    if 0 <= c_idx < len(color_palette):
                        color_lut[r_id] = np.asarray(color_palette[c_idx], dtype=np.uint8)[:3]
                color_img = color_lut[np.clip(markers, 0, num_regions)]
            color_img[line_mask] = (0, 0, 0)   # 线稿笔触填黑
            cv2.imencode(".png", cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))[1].tofile(
                f"Cache/preprocessed/{cache_name}_{i:04d}_color.png")

            # --- 预处理线稿 (骨架化) ---
            skel = np.asarray(g["skeleton_rgb"])
            skel_bin = skel[..., 3] > 0
            lineart_img = np.full(skel.shape[:2] + (3,), 255, dtype=np.uint8)
            lineart_img[skel_bin] = skel[..., :3][skel_bin]
            cv2.imencode(".png", cv2.cvtColor(lineart_img, cv2.COLOR_RGB2BGR))[1].tofile(
                f"Cache/preprocessed/{cache_name}_{i:04d}_lineart.png")
        print(f"[{cache_name}] 已保存 {len(self.graphs)} 张预处理彩图 + 线稿 → Cache/preprocessed/")

        # 训练/collate 只需轻量特征张量; 下面这些键是逐图的大对象, 仅用于图构建 / GT 彩图 /
        # 推理可视化, 训练阶段用不到。它们留存在内存会被 DataLoader(num_workers>0) 在 spawn 时
        # pickle 进每个子进程 (persistent_workers=True 下每个 worker 各持一份) → 内存翻倍导致
        # OOM / MemoryError (实测 Windows spawn 报 `reduction.pickle.load: MemoryError`)。
        # 推理(colorize_image_ll)会重新从图文件 extract_ll_graph 建图, 不依赖此处缓存, 故可安全剔除。
        # (缓存文件本身在上方 save_cache 时仍含这些键, GUI/调试读取不受影响)
        #
        # 剔除清单 (逐图大对象):
        #   markers / skeleton_rgb / skeleton_bin / lineart_rgb : 全分辨率图 (4 通道 RGBA, 数十 MB)
        #   line_label_map : 全分辨率 (H,W) int32 骨架→线索引映射 (每张数十 MB, 最大单键)
        #   line_info      : 每条线的 bfs_order 像素坐标列表 (数百线 × 数百坐标)
        #   keypoints      : 关键点列表 (调试/可视化用)
        for _g in self.graphs:
            for _k in ("markers", "skeleton_rgb", "skeleton_bin", "lineart_rgb",
                       "unified_target", "line_label_map", "line_info", "keypoints"):
                _g.pop(_k, None)

    def _build_graphs(self, line_paths, color_paths, line_palette, color_palette,
                      cache_name, save_cache, cache_path):
        """从图像对构建图 (缓存未命中时调用)."""
        print(f"[{cache_name}] 正在构建 v6 L-L 图...")
        t_build_start = time.time()

        # 准备有效的图片对
        valid_pairs = []
        for idx, (l_path, c_path) in enumerate(zip(line_paths, color_paths)):
            if not os.path.exists(l_path) or not os.path.exists(c_path):
                continue
            valid_pairs.append((idx, l_path, c_path))

        if len(valid_pairs) == 0:
            print(f"[{cache_name}] 未找到有效图片对")
            return

        # 多进程并行构建图
        num_workers = min(os.cpu_count() or 4, len(valid_pairs), 8)
        if num_workers > 1:
            print(f"[{cache_name}] 使用 {num_workers} 个进程并行处理...")
            from multiprocessing import Pool
            from functools import partial

            # 关键修复: 消融开关必须显式传给 worker。spawn 下 worker 是新解释器,
            #   重新执行模块顶层 → 全局 ENABLE_*_STREAM 回到默认值, 主进程的修改丢失。
            #   把当前生效值打包成 dict 传参, worker 内 extract_ll_graph 用该值置零。
            ablation = {
                'line': ENABLE_LINE_STREAM,
                'rcolor': ENABLE_RCOLOR_STREAM,
                'stroke_coarse': STROKE_COARSE_ENABLED,
            }
            worker_fn = partial(_process_single_graph, line_palette=line_palette,
                               color_palette=color_palette, cache_name=cache_name,
                               ablation=ablation)
            with Pool(processes=num_workers) as pool:
                results = pool.map(worker_fn, valid_pairs)
            self.graphs = [g for g in results if g is not None]
        else:
            # 单进程回退 (主进程内, 全局开关已由 main() 正确修改; 此处仍显式传 ablation 保持一致)
            ablation = {
                'line': ENABLE_LINE_STREAM,
                'rcolor': ENABLE_RCOLOR_STREAM,
                'stroke_coarse': STROKE_COARSE_ENABLED,
            }
            for idx, l_path, c_path in valid_pairs:
                l_raw = cv2.imdecode(np.fromfile(l_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                c_raw = cv2.imdecode(np.fromfile(c_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if l_raw is None or c_raw is None:
                    continue
                l_rgba = process_alpha(l_raw)
                c_rgba = process_alpha(c_raw)
                debug_pfx = f"{cache_name}_{idx}" if idx < 3 else None
                graph_data = extract_ll_graph(l_rgba, c_rgba, line_palette, color_palette,
                                              debug_prefix=debug_pfx, ablation=ablation)
                if graph_data is not None:
                    self.graphs.append(graph_data)

        t_build = time.time() - t_build_start
        print(f"[{cache_name}] 构建了 {len(self.graphs)} 个图, 耗时 {t_build:.1f}s")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        # 确保返回包含 triple_label
        g['triple_label'] = g.get('triple_label', torch.tensor([], dtype=torch.long))
        return g


def ll_collate_fn(batch):
    """
    批量合并 L-L 图.
    每个图有独立的 region 编号, 需要 offset.
    """
    X_l_list = []
    color_index_list = []
    edge_index_ll_list = []
    edge_attr_ll_list = []
    edge_relation_mask_list = []
    edge_region_a_list = []
    edge_region_b_list = []
    edge_label_a_list = []
    edge_label_b_list = []
    line_region_a_list = []
    line_region_b_list = []
    line_label_a_list = []
    line_label_b_list = []
    line_plus_region_a_list = []      # (N_l,) bool: region_a 是否在 +n 侧
    line_color_label_list = []
    is_color_line_list = []
    line_pixel_count_list = []
    line_pos_list = []                 # (N_l, 2) line 相对位置
    region_pos_list = []               # (N_r, 2) region 相对位置
    region_shape_feat_list = []
    region_line_color_geo_list = []   # (N_r, nc*3) 逐像素线色几何特征
    edge_index_rr_list = []
    y_list = []
    batch_l_list = []
    batch_r_list = []
    triple_label_list = []
    stroke_id_list = []
    edge_index_stroke_list = []
    edge_index_intra_stroke_list = []
    cont_stroke_id_list = []
    edge_index_cont_stroke_list = []
    edge_index_intra_cont_list = []

    l_offset = 0
    r_offset = 0
    s_offset = 0
    cs_offset = 0

    for idx, g in enumerate(batch):
        num_l = g["X_l"].size(0)
        num_r = g["num_regions"]

        X_l_list.append(g["X_l"])
        color_index_list.append(g["color_index"])
        line_region_a_list.append(g["line_region_a"] + r_offset)
        line_region_b_list.append(g["line_region_b"] + r_offset)
        line_label_a_list.append(g["line_label_a"])
        line_label_b_list.append(g["line_label_b"])
        line_plus_region_a_list.append(g["line_plus_region_a"])
        line_color_label_list.append(g["line_color_label"])
        is_color_line_list.append(g["is_color_line"])
        line_pixel_count_list.append(g["line_pixel_count"])
        line_pos_list.append(g["line_pos"])
        region_pos_list.append(g["region_pos"])
        region_shape_feat_list.append(g["region_shape_feat"])
        # 缺该键说明是旧缓存 → 直接 KeyError 而非补零: 补零会让 region_color_compress
        # 收到全零输入, 静默退化成"无区域线色信息"且极难排查 (缓存版本号已保证不会命中)
        region_line_color_geo_list.append(g["region_line_color_geo"])
        batch_l_list.append(torch.full((num_l,), idx, dtype=torch.long))
        batch_r_list.append(torch.full((num_r,), idx, dtype=torch.long))
        y_list.append(g["y"])
        triple_label_list.append(g["triple_label"])

        if g["edge_index_ll"].size(1) > 0:
            edge_index_ll_list.append(g["edge_index_ll"] + l_offset)
            edge_attr_ll_list.append(g["edge_attr_ll"])
            edge_relation_mask_list.append(g["edge_relation_mask"])
            edge_region_a_list.append(g["edge_region_a"] + r_offset)
            edge_region_b_list.append(g["edge_region_b"] + r_offset)
            edge_label_a_list.append(g["edge_label_a"])
            edge_label_b_list.append(g["edge_label_b"])

        if g["edge_index_rr"].size(1) > 0:
            edge_index_rr_list.append(g["edge_index_rr"] + r_offset)

        # 笔画粗支路 (CoRe-GNN): stroke_id / edge_index_stroke 按 batch 累计 offset
        _sid = g.get("stroke_id")
        _es = g.get("edge_index_stroke")
        _eis = g.get("edge_index_intra_stroke")
        if _sid is not None and _es is not None:
            stroke_id_list.append(_sid + s_offset)
            if _es.size(1) > 0:
                edge_index_stroke_list.append(_es + s_offset)
            if _eis is not None and _eis.size(1) > 0:
                edge_index_intra_stroke_list.append(_eis + l_offset)   # 组内边是 line 索引 → l_offset
            s_offset += int(_sid.max().item()) + 1

        # 方向连续笔画分组 (CoRe-GNN 第二套分组): 同样按 batch 累计 offset
        _csid = g.get("cont_stroke_id")
        _ces = g.get("edge_index_cont_stroke")
        _eic = g.get("edge_index_intra_cont")
        if _csid is not None and _ces is not None:
            cont_stroke_id_list.append(_csid + cs_offset)
            if _ces.size(1) > 0:
                edge_index_cont_stroke_list.append(_ces + cs_offset)
            if _eic is not None and _eic.size(1) > 0:
                edge_index_intra_cont_list.append(_eic + l_offset)
            cs_offset += int(_csid.max().item()) + 1

        l_offset += num_l
        r_offset += num_r

    return {
        "X_l": torch.cat(X_l_list, dim=0) if X_l_list else torch.empty((0, LINE_FEAT_DIM)),
        "color_index": torch.cat(color_index_list, dim=0) if color_index_list else torch.empty(0, dtype=torch.long),
        "line_region_a": torch.cat(line_region_a_list, dim=0) if line_region_a_list else torch.empty(0, dtype=torch.long),
        "line_region_b": torch.cat(line_region_b_list, dim=0) if line_region_b_list else torch.empty(0, dtype=torch.long),
        "line_label_a": torch.cat(line_label_a_list, dim=0) if line_label_a_list else torch.empty(0, dtype=torch.long),
        "line_label_b": torch.cat(line_label_b_list, dim=0) if line_label_b_list else torch.empty(0, dtype=torch.long),
        "line_plus_region_a": torch.cat(line_plus_region_a_list, dim=0) if line_plus_region_a_list else torch.empty(0, dtype=torch.bool),
        "line_color_label": torch.cat(line_color_label_list, dim=0) if line_color_label_list else torch.empty(0, dtype=torch.long),
        "is_color_line": torch.cat(is_color_line_list, dim=0) if is_color_line_list else torch.empty(0, dtype=torch.bool),
        "line_pixel_count": torch.cat(line_pixel_count_list, dim=0) if line_pixel_count_list else torch.empty(0, dtype=torch.float32),
        "line_pos": torch.cat(line_pos_list, dim=0) if line_pos_list else torch.empty((0, 2), dtype=torch.float32),
        "region_pos": torch.cat(region_pos_list, dim=0) if region_pos_list else torch.empty((0, 2), dtype=torch.float32),
        "region_shape_feat": torch.cat(region_shape_feat_list, dim=0) if region_shape_feat_list else torch.empty((0, 2), dtype=torch.float32),
        "region_line_color_geo": torch.cat(region_line_color_geo_list, dim=0) if region_line_color_geo_list else torch.empty((0, 0)),
        "edge_index_ll": torch.cat(edge_index_ll_list, dim=1) if edge_index_ll_list else torch.empty((2, 0), dtype=torch.long),
        "edge_attr_ll": torch.cat(edge_attr_ll_list, dim=0) if edge_attr_ll_list else torch.empty((0, EDGE_ATTR_DIM)),
        "edge_relation_mask": torch.cat(edge_relation_mask_list, dim=0) if edge_relation_mask_list else torch.empty((0, EDGE_RELATION_DIM)),
        "edge_region_a": torch.cat(edge_region_a_list, dim=0) if edge_region_a_list else torch.empty(0, dtype=torch.long),
        "edge_region_b": torch.cat(edge_region_b_list, dim=0) if edge_region_b_list else torch.empty(0, dtype=torch.long),
        "edge_label_a": torch.cat(edge_label_a_list, dim=0) if edge_label_a_list else torch.empty(0, dtype=torch.long),
        "edge_label_b": torch.cat(edge_label_b_list, dim=0) if edge_label_b_list else torch.empty(0, dtype=torch.long),
        "edge_index_rr": torch.cat(edge_index_rr_list, dim=1) if edge_index_rr_list else torch.empty((2, 0), dtype=torch.long),
        "y": torch.cat(y_list, dim=0),
        "batch_l": torch.cat(batch_l_list, dim=0),
        "batch_r": torch.cat(batch_r_list, dim=0),
        "triple_label": torch.cat(triple_label_list, dim=0) if triple_label_list else torch.empty(0, dtype=torch.long),
        "stroke_id": torch.cat(stroke_id_list, dim=0) if stroke_id_list else torch.empty(0, dtype=torch.long),
        "edge_index_stroke": torch.cat(edge_index_stroke_list, dim=1) if edge_index_stroke_list else torch.empty((2, 0), dtype=torch.long),
        "edge_index_intra_stroke": torch.cat(edge_index_intra_stroke_list, dim=1) if edge_index_intra_stroke_list else torch.empty((2, 0), dtype=torch.long),
        "cont_stroke_id": torch.cat(cont_stroke_id_list, dim=0) if cont_stroke_id_list else torch.empty(0, dtype=torch.long),
        "edge_index_cont_stroke": torch.cat(edge_index_cont_stroke_list, dim=1) if edge_index_cont_stroke_list else torch.empty((2, 0), dtype=torch.long),
        "edge_index_intra_cont": torch.cat(edge_index_intra_cont_list, dim=1) if edge_index_intra_cont_list else torch.empty((2, 0), dtype=torch.long),
    }


# ==================== 4. GNN 模型 ====================

def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """按 index 分组做 softmax (数值稳定版: 先减去组内最大值)。

    Args:
        src: (E,) 每条边的注意力分数
        index: (E,) 每条边归属的目标节点 id
        dim_size: 目标节点总数

    Returns:
        (E,) 组内 softmax 结果
    """
    max_val = torch.full((dim_size,), -float('inf'), device=src.device, dtype=src.dtype)
    max_val.scatter_reduce_(0, index, src, reduce='amax', include_self=False)
    # NaN 保护: -inf → 0 (当组内全被 mask 时, exp(-inf - (-inf)) = NaN)
    max_val = torch.where(max_val == float('-inf'), torch.zeros_like(max_val), max_val)
    max_val = max_val.gather(0, index)

    exp_val = torch.exp(src - max_val)

    sum_val = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    sum_val.scatter_add_(0, index, exp_val)
    sum_val = sum_val.gather(0, index)

    return exp_val / (sum_val + 1e-16)

# ==================== 节点级 NBA-GNN ====================
def _make_stroke_proj(ctx_dim, out_dim):
    """Stroke 支路末端投影的构造入口 (S1-LR 低秩 / 深支路两种开关)。

    baseline: Linear(ctx,48) -> GELU                                   (S0 原结构)
    lowrank : Linear(ctx,r,False) -> Linear(r,48) -> GELU             (SVD 低秩近似)
    deep    : Linear(ctx,h) -> ReLU -> (m-1)×[Linear(h,h) -> ReLU] -> Linear(h,48)
              (拼接后至少 m>=2 个带激活的隐层, 输出直接投回主路维度, 无额外激活)
    "deep" 模式下每个统计量先经 Linear(48,k)+ReLU 压到 k 维 (见 branch 构造)。
    """
    if STROKE_BRANCH_MODE == "deep":
        h = STROKE_MLP_HIDDEN
        drop = (nn.Dropout(STROKE_BRANCH_DROPOUT)
                if STROKE_BRANCH_DROPOUT > 0 else nn.Identity())
        layers = [nn.Linear(ctx_dim, h), nn.ReLU(), drop]
        for _ in range(int(STROKE_MLP_LAYERS) - 1):
            layers += [nn.Linear(h, h), nn.ReLU(), drop]
        layers += [nn.Linear(h, out_dim)]
        return nn.Sequential(*layers)
    if STROKE_LOWRANK_RANK > 0:
        return nn.Sequential(
            nn.Linear(ctx_dim, STROKE_LOWRANK_RANK, bias=False),
            nn.Linear(STROKE_LOWRANK_RANK, out_dim, bias=True),
            nn.GELU(),
        )
    return nn.Sequential(nn.Linear(ctx_dim, out_dim), nn.GELU())


def _make_stat_proj(in_dim):
    """deep 模式: 每个组内统计量 48 -> k 的感知机。

    STROKE_STAT_HIDDEN=0 (默认): Linear(48,k)+ReLU (单层)
    STROKE_STAT_HIDDEN=h>0     : Linear(48,h)+ReLU -> Linear(h,k)+ReLU (两层)
    """
    if STROKE_BRANCH_MODE == "deep" and STROKE_STAT_DIM > 0:
        if STROKE_STAT_HIDDEN > 0:
            return nn.Sequential(
                nn.Linear(in_dim, STROKE_STAT_HIDDEN), nn.ReLU(),
                nn.Linear(STROKE_STAT_HIDDEN, STROKE_STAT_DIM), nn.ReLU(),
            )
        return nn.Sequential(nn.Linear(in_dim, STROKE_STAT_DIM), nn.ReLU())
    if STROKE_STAT_DIM > 0:
        return nn.Sequential(nn.Linear(in_dim, STROKE_STAT_DIM), nn.GELU())
    return None


class LineGNNLayer(nn.Module):
    """节点级 NBA-GNN: GATv2 (edge-aware QKV) + 度归一化求和 + 非回溯 mask

    GATv2: q/k/v 均拼接边属性
      attn = a^T · LeakyReLU(W_q·[x_dst, e] + W_k·[x_src, e])
    NBA: 每层屏蔽上一层最强影响邻居, 防止消息来回震荡

    正则化 (2026-09-27 修订为标准 GAT 口径):
      - 注意力 dropout: 作用在 softmax 归一化后的注意力系数 α 上 (标准 GAT 做法),
        被置 0 的边贡献真正归零 → 稀疏化聚焦边; 此前打在 logits 上是"软化"而非"剪边"
      - 消息 dropout: out_proj(aggr) 后做 dropout, 仅打 GNN 增量、不碰主干
      - Post-Norm LayerNorm: 残差输出归一化, 防止 5 层串联的分布漂移
    """

    def __init__(self, node_dim, edge_dim, heads=3, attn_dropout=0.0, msg_dropout=0.0,
                 relation_dim=EDGE_RELATION_DIM,
                 relation_droppath_rate=RELATION_DROPPATH_RATE):
        super().__init__()
        self.heads = heads
        self.node_dim = node_dim
        self.head_dim = node_dim // heads
        self.relation_droppath_rate = relation_droppath_rate

        # --- GATv2: Q/K 纯节点注意力, V 拼接边属性 ---
        # 打分路径: 节点特征走 Q/K, 边特征由下方 edge_bias 直接注入; 二者不再重复。
        # 消息路径: V 仍拼接 edge_attr, 承载 tan_half/len_ratio 等几何关系作为消息内容。
        self.W_q = nn.Linear(node_dim, node_dim)
        self.W_k = nn.Linear(node_dim, node_dim)
        self.W_v = nn.Linear(node_dim + edge_dim, node_dim)
        self.attn_proj = nn.Linear(self.head_dim, 1, bias=False)
        self.relation_v = nn.ModuleList([
            nn.Linear(node_dim + edge_dim, node_dim) for _ in range(relation_dim)
        ])

        # --- 显式边偏置 (edge bias): edge_attr 直接注入注意力分数 ---
        # 小样本下 tan_half/len_ratio 等几何信号若只藏在 W_q/W_k 拼接里, 需经
        # 两层线性才影响分数, 难以学出。此捷径让边特征直接贡献一个加性偏置。
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.relation_bias = nn.ModuleList([
            nn.Linear(edge_dim, heads) for _ in range(relation_dim)
        ])
        self.relation_gate = nn.Linear(node_dim + relation_dim, relation_dim)

        self.out_proj = nn.Linear(node_dim, node_dim)
        self.activation = nn.GELU()   # 消息变换后的逐通道非线性 (补齐层间表达能力)
        self.res_scale = nn.Parameter(torch.tensor(0.6))

        # --- 正则化: LayerNorm + Dropout ---
        self.norm_out = nn.LayerNorm(node_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.msg_dropout = nn.Dropout(msg_dropout)

        # --- 支路融合归一化: 3 条支路异源, 幅度差异大, 逐支路 LayerNorm 对齐尺度后再加权求和 ---
        # 消除"region 桥接幅度失控盖过主路"的风险, 让可学习 scale 门控在单位尺度上公平竞争。
        self.branch_norm_msg = nn.LayerNorm(node_dim)
        self.branch_norm_coarse = nn.LayerNorm(node_dim)
        self.branch_norm_cont = nn.LayerNorm(node_dim)
        self.branch_norm_region = nn.LayerNorm(node_dim)

        # --- 笔画级粗支路 (CoRe-GNN 并行): 与细图局部消息并行, 注入笔画级长程语义 ---
        # 两套分组(corner/cont)各自一条支路, 均 soft 残差注入 (scale 初始极小, 可学习自调节)。
        # coarse_proj 输出依赖节点自身 x (cat[x, ctx]), 保住簇内区分度 (CoRe 失败模式修复)。
        # 组内链式消息投影 (层次3): 沿组内双向边聚合前驱+后继, 注入连接序上下文。
        # 注意: 两套投影保持独立 —— 两种分组的组语义差异巨大 (corner=2~3线碎片 / cont=数十线连续笔画),
        # 组统计量分布不同, 共享投影会被两种语义拉扯(实测退化)。各自独立投影更适配。
        # ring 支路已移除: 其"区域成员"语义与 region 支路冗余, 且并查集在线桥接下退化为骨架连通分量。
        # 支路开关 (STROKE_BRANCH) 与统计量压缩开关 (STROKE_STAT_DIM) 决定这里的参数量。
        self.use_corner_branch = STROKE_BRANCH in ("both", "corner_only")
        self.use_cont_branch = STROKE_BRANCH in ("both", "cont_only")
        _n_stat = 3                                      # mean / max / std
        self._n_stat = _n_stat
        if STROKE_STAT_DIM > 0:
            # 先自压缩: 每个统计量 48 -> k, 再拼接综合
            _ctx_dim = node_dim + STROKE_STAT_DIM * _n_stat + 1
        else:
            _ctx_dim = node_dim * (1 + _n_stat) + 1      # cat[x, *stats, log_size]
        if self.use_corner_branch:
            self.intra_proj = nn.Linear(node_dim, node_dim)
            if STROKE_STAT_DIM > 0:
                self.coarse_stat_proj = nn.ModuleList(
                    [_make_stat_proj(node_dim) for _ in range(_n_stat)])
            else:
                self.coarse_stat_proj = None
            self.coarse_proj = _make_stroke_proj(_ctx_dim, node_dim)
        else:
            self.intra_proj = None
            self.coarse_stat_proj = None
            self.coarse_proj = None
        if self.use_cont_branch:
            self.cont_intra_proj = nn.Linear(node_dim, node_dim)
            if STROKE_STAT_DIM > 0:
                self.cont_coarse_stat_proj = nn.ModuleList(
                    [_make_stat_proj(node_dim) for _ in range(_n_stat)])
            else:
                self.cont_coarse_stat_proj = None
            self.cont_coarse_proj = (self.coarse_proj
                                     if (STROKE_SHARE_MLP and STROKE_BRANCH_MODE == "deep")
                                     else _make_stroke_proj(_ctx_dim, node_dim))
        else:
            self.cont_intra_proj = None
            self.cont_coarse_stat_proj = None
            self.cont_coarse_proj = None

        # --- Region 粗支路 (方案A): line→region 聚合 → region 长程(edge_index_rr) → line 广播 ---
        # 补组间/全局长程盲区 (现有 CoRe 组级邻接 edge_index_stroke 未用, 组间只能细图逐跳);
        # region 邻接图直径小, 远处 line 经 region 桥接几次跳即互见。scale 初始极小(软注入)防小样本过拟合。
        self.region_proj = nn.Linear(node_dim, node_dim)
        self.region_out_proj = nn.Linear(node_dim, node_dim)
        # 各粗支路融合强度 (软注入): 主消息支路隐式权重 1 为绝对主干。
        #   初值由 BRANCH_SCALE_INIT 统一控制 (可被环境变量覆盖, 见文件头常量区)。
        self.coarse_scale = nn.Parameter(torch.tensor(BRANCH_SCALE_INIT))
        self.cont_coarse_scale = nn.Parameter(torch.tensor(BRANCH_SCALE_INIT))
        self.region_scale = nn.Parameter(torch.tensor(BRANCH_SCALE_INIT))
        # 逐 Node 逐支路门控 (BRANCH_GATE_MODE="node" 时启用, 默认不参与前向):
        #   g_{i,b} = amp_b * sigmoid(W_b [h_i ; m_{i,b}] + c_b)
        #   W_b / c_b 零初始化 => sigmoid(0)=0.5, 初值 g = amp_b/2 = BRANCH_SCALE_INIT,
        #   与原全局标量融合强度逐元素相等 (残差尺度不变); amp_b 学整体强度, W_b 学逐线调制。
        #   **RNG 隔离**: 构造 nn.Linear 会消耗全局随机流, 从而改变后续所有初始化与数据顺序,
        #   使 global 模式无法逐位复现基准 (实测 V 差 2/703 个 region)。这里存取/还原 RNG 状态,
        #   保证新增的参数不扰动既有的随机序列。
        _rng_state = torch.random.get_rng_state()
        self.coarse_gate = nn.Linear(node_dim * 2, 1)
        self.cont_coarse_gate = nn.Linear(node_dim * 2, 1)
        self.region_gate = nn.Linear(node_dim * 2, 1)
        self.coarse_amp = nn.Parameter(torch.tensor(2.0 * BRANCH_SCALE_INIT))
        self.cont_coarse_amp = nn.Parameter(torch.tensor(2.0 * BRANCH_SCALE_INIT))
        self.region_amp = nn.Parameter(torch.tensor(2.0 * BRANCH_SCALE_INIT))
        for _g in (self.coarse_gate, self.cont_coarse_gate, self.region_gate):
            nn.init.zeros_(_g.weight)
            nn.init.zeros_(_g.bias)
        torch.random.set_rng_state(_rng_state)

    def _coarse_branch(self, x, group_id, intra_edge_index, intra_proj, proj,
                       stat_projs=None):
        """单分组粗支路 (层次3): 组内链式消息传递 → 四统计量池化 → 广播 → concat 融合。

        先沿组内双向边 (corner/cont 线对) 做 1 层邻居消息传递: 每个节点吸收前驱+后继
        的顺序上下文, 突破无序集合聚合; 再做 mean+max+std+log_size 四统计量池化:
          - mean : 平均语义 (稳定)
          - max  : 最强响应 (保边角/突显信号)
          - std  : 组内一致性 (连续笔画 vs 散乱线段)
          - log_size : 组大小 (大结构 vs 碎片; 显式携带, 不被 LayerNorm 抹掉, 优于 sum 的隐式幅度)
        返回原始输出 (尺度由 forward 的 branch_norm + scale 统一处理)。
        """
        if group_id is None or group_id.numel() == 0:
            return torch.zeros_like(x)
        if intra_edge_index is not None and intra_edge_index.size(1) > 0:
            src, dst = intra_edge_index[0], intra_edge_index[1]
            m = intra_proj(x[src])                                  # (E, H) 邻居消息
            aggr_intra = scatter_mean(m, dst, dim_size=x.size(0))   # (N, H) 前驱+后继平均
            x_ref = x + aggr_intra                                  # 残差注入顺序上下文
        else:
            x_ref = x
        num_groups = int(group_id.max().item()) + 1
        h_mean = scatter_mean(x_ref, group_id, dim_size=num_groups)
        h_max = scatter_max(x_ref, group_id, dim_size=num_groups)
        # std: 组内二阶统计 (散乱度)。
        # 数值安全: sqrt 在 0 处梯度 1/(2√x)=inf → 单成员组/组内同值时反向 NaN。
        #   故加 epsilon: sqrt(var + eps), 梯度上界 1/(2√eps) 有限, 同时防 E[x²]≈E[x]² 浮点负残差。
        h_sq = scatter_mean(x_ref * x_ref, group_id, dim_size=num_groups)
        h_std = torch.sqrt(torch.clamp(h_sq - h_mean * h_mean, min=0.0) + 1e-6)
        stats = [h_mean[group_id], h_max[group_id], h_std[group_id]]
        if stat_projs is not None:
            # 先自压缩: 每个统计量独立 48 -> k (含非线性), 再拼接综合
            stats = [sp(s_) for sp, s_ in zip(stat_projs, stats)]
        # log_size: 组大小 (成员数), 显式携带; log1p 压缩大组量级
        count = scatter_sum(torch.ones_like(group_id, dtype=x.dtype), group_id, dim_size=num_groups)
        log_size = torch.log1p(count)                                # (G,)
        parts = [x] + stats + [log_size[group_id].unsqueeze(-1)]      # (N, H + k*n_stat + 1)
        out = proj(torch.cat(parts, dim=-1))                          # 依赖节点自身, 保留簇内区分度
        if STROKE_BRANCH_RESIDUAL and STROKE_BRANCH_MODE == "deep":
            out = out + x                                             # 深支路恒等短路 (可选)
        return out

    def _region_branch(self, x, line_region_a, line_region_b, edge_index_rr, num_regions):
        """Region 粗支路 (方案A): line→region 聚合 → region 长程 → line 广播。

        补组间/全局长程盲区: 现有 CoRe 只做组内池化, 组级邻接 (edge_index_stroke) 被计算
        但未使用, stroke 组之间只能靠细图 L-L 逐跳慢传。region 邻接图 (edge_index_rr) 直径小,
        远处 line 经 region 桥接几次跳即可互见, 并带回两侧 region 上下文。
        """
        if (line_region_a is None or line_region_a.numel() == 0
                or num_regions is None or num_regions <= 0):
            return torch.zeros_like(x)
        ra = line_region_a.long()
        rb = line_region_b.long()
        if REGION_POOL_MODE == "split":
            # 旧行为 (保留用于复现): 两侧各求一次均值再相加。
            #   问题: (1) 两侧各占 50% 权重, 与"所有接触该 region 的线的统一均值"不等价;
            #        (2) 内部线 (ra==rb) 在两个均值里各算一次;
            #        (3) 结果依赖 A/B 编号, 而 A/B 只是按面积排序的图提取约定;
            #        (4) 幅度天然约 2.7x 于统一均值。
            reg = (scatter_mean(x, ra, dim_size=num_regions)
                   + scatter_mean(x, rb, dim_size=num_regions))
        else:
            # 统一 incidence 池化 (默认): 每条线对每个**不同**的接触 region 只贡献一次。
            #   内部线只计一次 -> 与 A/B 编号无关, 权重等于真实 incidence 均值。
            _n = x.size(0)
            _internal = ra == rb
            _src = torch.cat([torch.arange(_n, device=x.device),
                              torch.arange(_n, device=x.device)[~_internal]])
            _dst = torch.cat([ra, rb[~_internal]])
            reg = scatter_mean(x[_src], _dst, dim_size=num_regions)
        if edge_index_rr is not None and edge_index_rr.size(1) > 0:
            src = edge_index_rr[0].long()
            dst = edge_index_rr[1].long()
            reg = reg + scatter_mean(self.region_proj(reg[src]), dst, dim_size=num_regions)
        ctx = reg[ra] + reg[rb]                                # (N, H) 两侧 region 上下文
        return self.region_out_proj(ctx)

    def forward(self, x, edge_index, edge_attr, relation_mask=None, backtrack_node=None,
                stroke_id=None, edge_index_stroke=None, edge_index_intra_stroke=None,
                cont_stroke_id=None, edge_index_cont_stroke=None, edge_index_intra_cont=None,
                line_region_a=None, line_region_b=None, edge_index_rr=None, num_regions=None):
        """
        Args:
            x: (N, node_dim) 线特征
            edge_index: (2, E) 有向 L-L 边
            edge_attr: (E, edge_dim) L-L 边属性 (已编码)
            backtrack_node: (N,) long, 上层最强影响邻居 (-1 = 无)
        Returns:
            h_out: (N, node_dim) 更新后的线特征
            strongest_src: (N,) long, 本层最强影响邻居 (供下层 mask)
        """
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)
        if relation_mask is None:
            relation_mask = torch.zeros(
                (edge_index.size(1), EDGE_RELATION_DIM),
                device=x.device, dtype=x.dtype)
        relation_mask_structural = relation_mask
        if self.training and self.relation_droppath_rate > 0.0:
            keep_prob = 1.0 - self.relation_droppath_rate
            keep = (torch.rand(
                EDGE_RELATION_DIM, device=x.device) < keep_prob).to(x.dtype)
            relation_mask = relation_mask * keep / keep_prob

        # Q/K 吃纯节点特征 (边特征走 edge_bias), V 仍拼接 edge_attr 承载消息内容
        src_edge = torch.cat([x[src], edge_attr], dim=-1)
        q = self.W_q(x[dst]).view(-1, self.heads, self.head_dim)
        k = self.W_k(x[src]).view(-1, self.heads, self.head_dim)
        v = self.W_v(src_edge).view(-1, self.heads, self.head_dim)

        # GATv2: q + k → 注意力分数
        attn_input = q + k
        attn = F.leaky_relu(attn_input, 0.2)
        attn = self.attn_proj(attn).squeeze(-1)       # (E, heads) 原始注意力 logits

        # 显式边偏置: edge_attr → (E, heads) 加性项, 直接进注意力分数
        attn = attn + self.edge_bias(edge_attr)
        relation_bias = torch.stack(
            [layer(edge_attr) for layer in self.relation_bias], dim=1)
        attn = attn + (relation_mask.unsqueeze(-1) * relation_bias).sum(dim=1)
        if GAT_ABLATE:
            attn = torch.zeros_like(attn)   # 均匀注意力 (softmax 后 = 1/deg), 消融学习注意力

        # NBA: 屏蔽上层最强影响邻居 (softmax 前 mask, 被屏蔽边得 -inf)
        attn_pre_mask = attn
        if not NBA_ABLATE and backtrack_node is not None:
            mask_edge = (backtrack_node[dst] == src)
            attn = attn.masked_fill(mask_edge.unsqueeze(-1), float('-inf'))

        # Softmax (per dst node, per head 独立归一化)
        E = attn.size(0)
        attn_flat = attn.reshape(E * self.heads)
        head_off = torch.arange(self.heads, device=x.device).repeat(E) * N
        idx_flat = dst.repeat_interleave(self.heads) + head_off
        attn_flat = scatter_softmax(attn_flat, idx_flat, dim_size=N * self.heads)
        attn = attn_flat.reshape(E, self.heads)

        # 注意力 dropout: 标准 GAT 做法 —— 作用在 softmax 归一化后的注意力系数 α 上,
        # 被置 0 的边其贡献真正归零 (稀疏化聚焦边), 防注意力过拟合。仅训练期生效。
        attn = self.attn_dropout(attn)

        # 本层最强影响邻居
        if NBA_ABLATE:
            strongest_src = None
        else:
            attn_for_pick = attn_pre_mask.clone()
            if backtrack_node is not None:
                mask_edge = (backtrack_node[dst] == src)
                attn_for_pick = attn_for_pick.masked_fill(
                    mask_edge.unsqueeze(-1), float('-inf'))
            attn_max_per_edge = attn_for_pick.max(dim=1).values
            order = dst.float() * (E + 1) - attn_max_per_edge
            sort_idx = torch.argsort(order)
            sorted_dst = dst[sort_idx]
            sorted_src = src[sort_idx]
            sorted_score = attn_max_per_edge[sort_idx]
            first_mask = torch.ones(E, dtype=torch.bool, device=x.device)
            first_mask[1:] = sorted_dst[1:] != sorted_dst[:-1]
            strongest_src = torch.full((N,), -1, dtype=torch.long, device=x.device)
            first_dst = sorted_dst[first_mask]
            first_src = sorted_src[first_mask]
            first_score = sorted_score[first_mask]
            valid = first_score > float('-inf')
            strongest_src[first_dst[valid]] = first_src[valid]

        # --- 注意力加权求和 ---
        # 标准 GAT: α 已在 per-dst softmax 后 Σα=1, scatter_sum(α·v) 即加权平均。
        # 不再除 degree —— 那样是"双重归一化", 会把高度数交点(语义最丰富处)的消息
        # 系统性压低到毛刺/末梢线水平, 稀释关键节点的信息。
        msg = v * attn.unsqueeze(-1)
        relation_values = torch.stack(
            [layer(src_edge).view(-1, self.heads, self.head_dim)
             for layer in self.relation_v], dim=1)
        relation_gate = torch.sigmoid(
            self.relation_gate(torch.cat([x[dst], relation_mask_structural], dim=-1)))
        relation_msg = (relation_values * relation_gate.unsqueeze(-1).unsqueeze(-1)
                        * relation_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        msg = msg + relation_msg * attn.unsqueeze(-1)
        aggr = scatter_sum(msg, dst, dim_size=N)
        aggr = aggr.reshape(N, self.node_dim)

        # 消息 dropout: 仅打 GNN 增量, 不碰主干 x
        # 补非线性: out_proj 是纯线性, 这里接 GELU 后每层消息变换才具备非线性表达
        # (注意力 softmax 之外的第二处非线性), 避免 5 层堆叠退化为近线性映射。
        msg_out = self.msg_dropout(self.activation(self.out_proj(aggr)))

        # 笔画级粗支路 (CoRe-GNN 并行, 两套分组): 保留细图完整信息 (角点边不删),
        # 组图只作长程语义注入; cat[x, ctx] 保留簇内区分度, scale 初始极小(弱补充)。
        coarse_out = (self._coarse_branch(x, stroke_id, edge_index_intra_stroke,
                                         self.intra_proj, self.coarse_proj,
                                         self.coarse_stat_proj)
                      if self.use_corner_branch else None)
        cont_out = (self._coarse_branch(x, cont_stroke_id, edge_index_intra_cont,
                                       self.cont_intra_proj, self.cont_coarse_proj,
                                       self.cont_coarse_stat_proj)
                    if self.use_cont_branch else None)
        region_out = self._region_branch(x, line_region_a, line_region_b, edge_index_rr, num_regions)

        # 归一化融合: 各支路先 LayerNorm 对齐尺度, 再加权求和 (主消息支路隐式权重 1 为主干)。
        m_msg = self.branch_norm_msg(msg_out)
        m_c = self.branch_norm_coarse(coarse_out) if coarse_out is not None else None
        m_cc = self.branch_norm_cont(cont_out) if cont_out is not None else None
        m_r = self.branch_norm_region(region_out)
        if BRANCH_GATE_MODE == "node":
            # 逐 Node 逐支路自适应门控 (清单第 3 项)
            g_c = self.coarse_amp * torch.sigmoid(
                self.coarse_gate(torch.cat([x, m_c], dim=-1)))
            g_cc = self.cont_coarse_amp * torch.sigmoid(
                self.cont_coarse_gate(torch.cat([x, m_cc], dim=-1)))
            g_r = self.region_amp * torch.sigmoid(
                self.region_gate(torch.cat([x, m_r], dim=-1)))
        else:
            # 基准: 每支路一个全局可学标量
            g_c, g_cc, g_r = self.coarse_scale, self.cont_coarse_scale, self.region_scale
        branch_sum = m_msg + g_r * m_r
        if m_c is not None:
            branch_sum = branch_sum + g_c * m_c
        if m_cc is not None:
            branch_sum = branch_sum + g_cc * m_cc

        # Post-Norm: 残差输出归一化防分布漂移
        h_out = self.norm_out(x + self.res_scale * branch_sum)
        return h_out, strongest_src


class LineColorizerBackbone(nn.Module):
    """共享判别器 backbone (跨数据集复用): Line/Edge/Region 编码 + GNN + Region 侧编码。

    输出 (h_l, h_r); 再经 encode_sides 得到 side_sum(两侧调制相加) 与 z_plus/z_minus(逐侧)。
    分类器 (每数据集独立) 见 ClassifierHead。所有维度 (color_embed / region_color_compress)
    均用固定常量 MAX_LINE_COLORS, 与具体数据集的调色板无关 → backbone 可跨数据集共享。
    """

    def __init__(self, line_in_dim=LINE_FEAT_DIM, edge_attr_dim=EDGE_ATTR_DIM,
                 hidden_dim=HIDDEN_DIM,
                 line_enc_dim=LINE_ENC_DIM,
                 edge_enc_dim=EDGE_ENC_DIM, num_gnn_layers=GNN_LAYERS,
                 max_line_colors=MAX_LINE_COLORS, color_embed_dim=COLOR_EMB_DIM,
                 cls_side_dim=CLASSIFIER_SIDE_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_colors = max_line_colors
        self.cls_side_dim = cls_side_dim
        self.color_embed_dim = color_embed_dim

        # --- Line 编码器 (局部坐标系几何特征) ---
        self.line_encoder = nn.Sequential(
            nn.Linear(line_in_dim, line_enc_dim * 2),
            nn.LayerNorm(line_enc_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(line_enc_dim * 2, line_enc_dim),
            nn.LayerNorm(line_enc_dim),
            nn.ReLU(),
        )
        # --- Line 颜色嵌入 (离散类别 → 稠密向量) ---
        self.color_embed = nn.Embedding(max_line_colors, color_embed_dim)
        self.color_proj = nn.Sequential(
            nn.Linear(color_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.line_proj = nn.Linear(line_enc_dim, hidden_dim)

        # --- 边编码器: 2D 原始边属性 → edge_enc_dim ---
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_enc_dim * 2),
            nn.LayerNorm(edge_enc_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(edge_enc_dim * 2, edge_enc_dim),
        )

        # --- 消息传递层 (节点级 NBA-GNN, 每层内嵌 CoRe 并行粗支路) ---
        # 标准 GAT 做法: 前面层多 head 保留跨子空间表达能力, 最后一层单 head 做输出融合。
        # head 数 3 (与 HIDDEN_DIM 48 配套, head_dim = 48//3 = 16)。
        gnn_heads = 3
        self.gnn_layers = nn.ModuleList([
            LineGNNLayer(
                hidden_dim, edge_enc_dim,
                heads=(gnn_heads if i < num_gnn_layers - 1 else 1),
                attn_dropout=ATTN_DROPOUT, msg_dropout=MSG_DROPOUT,
            )
            for i in range(num_gnn_layers)
        ])
        self.gnn_dropout = nn.Dropout(OUTPUT_DROPOUT)

        # Region 线色几何特征压缩: max_line_colors*3 → COLOR_EMB_DIM (固定维度)
        self.region_color_compress = nn.Sequential(
            nn.Linear(max_line_colors * 3, color_embed_dim),
            nn.LayerNorm(color_embed_dim),
            nn.ReLU(),
        )

        # --- Region 侧编码器 (对称分类器的共享部分) ---
        #   输入 h_r (区域上下文: 线色几何 + 位置 + 形状, COLOR_EMB_DIM 维)。
        # --- 两套彼此独立的 region 编码器 ---
        #   A) side_region_encoder : 只看**单个** region, 供 side 分类器 (同一套权重对两侧各调一次)
        #   B) triple_pair_encoder : 吃 (region_plus, region_minus) 的**拼接**, 供 triple 分类器。
        #      对两个顺序各前向一次再取平均 —— DeepSets 式对称化 (g(x,y)+g(y,x))/2,
        #      严格顺序无关, 且比"各自编码后相加"更有表达力 (可以学到 x、y 之间的交互)。
        #   两者权重不共享: triple 判"两个 region 的组合", side 只判"这一侧是哪个 region"。
        self.side_region_encoder = nn.Sequential(
            nn.Linear(color_embed_dim, cls_side_dim),
            nn.LayerNorm(cls_side_dim), nn.GELU(),
        )
        self.triple_pair_encoder = nn.Sequential(
            nn.Linear(2 * color_embed_dim, cls_side_dim),
            nn.LayerNorm(cls_side_dim), nn.GELU(),
        )
        # 侧编码 (cls_side_dim) → node 空间 (hidden_dim), 供 z_side = h_line + proj(s_side) 相加。
        self.side_node_proj = nn.Linear(cls_side_dim, hidden_dim, bias=False)
        # --- 逐 node、逐侧的 region 注入幅度 alpha ---
        #   分类哪一侧, 就只用 "当前 node 特征 + 该侧 region 上下文" 决定该侧的注入强度。
        #   两侧共享同一门控网络 => A/B 交换等变; 不含对侧 region, 避免信息泄漏。
        #   末层零初始化 + 2*sigmoid => 初始 alpha 恒为 1.0, 起手与基线完全一致。
        self.side_region_gate = nn.Sequential(
            nn.Linear(hidden_dim + color_embed_dim, POS_HIDDEN),
            nn.GELU(),
            nn.Linear(POS_HIDDEN, 1),
        )
        nn.init.zeros_(self.side_region_gate[-1].weight)
        nn.init.zeros_(self.side_region_gate[-1].bias)

        # --- 位置嵌入 (独立分支, E4): 低维 MLP 编码相对坐标 → 加性注入 ---
        # line 位置 → hidden_dim, 注入 h_l (GNN 之后); region 位置 → color_embed_dim, 注入 h_r。
        # 不参与 GNN 消息传递, 仅作分类前的绝对几何补充 (见 feature_invariance_report.md 类别 B)。
        self.line_pos_encoder = nn.Sequential(
            nn.Linear(2, POS_HIDDEN),
            nn.GELU(),
            nn.Linear(POS_HIDDEN, hidden_dim),
        )
        self.region_pos_encoder = nn.Sequential(
            nn.Linear(2, POS_HIDDEN),
            nn.GELU(),
            nn.Linear(POS_HIDDEN, color_embed_dim),
        )
        # region 形状特征独立 encoder + 可学习标量 (初值 1.0 = 全强度加性, 再让模型自行缩放)
        self.region_shape_encoder = nn.Sequential(
            nn.Linear(2, POS_HIDDEN),
            nn.GELU(),
            nn.Linear(POS_HIDDEN, color_embed_dim),
        )
        self.region_shape_scale = nn.Parameter(torch.tensor(1.0))

        # --- X_l 输入标准化 (z-score) 统计量: 训练前由 compute_x_l_stats 填充 ---
        #   默认 mean=0 / std=1 → 恒等变换, 未注入统计量时安全退化 (旧 checkpoint 无 buffer)。
        #   作为 buffer 注册: 随 state_dict 自动保存/加载, 训练与推理共享同一组统计量。
        self.register_buffer("x_l_mean", torch.zeros(line_in_dim))
        self.register_buffer("x_l_std", torch.ones(line_in_dim))

    def set_x_l_stats(self, mean, std):
        """注入 X_l 逐维 z-score 标准化统计量 (形状 = line_in_dim)。"""
        self.x_l_mean.copy_(mean.to(self.x_l_mean.device, self.x_l_mean.dtype))
        self.x_l_std.copy_(std.to(self.x_l_std.device, self.x_l_std.dtype))

    def _encode_edge_attr(self, edge_attr):
        if edge_attr.size(0) == 0:
            return torch.zeros((0, self.edge_encoder[-1].out_features),
                               device=edge_attr.device, dtype=edge_attr.dtype)
        return self.edge_encoder(edge_attr)

    def forward(self, X_l, edge_index_ll, edge_attr_ll,
                color_index=None, edge_index_rr=None,
                line_region_a=None, line_region_b=None,
                region_line_color_geo=None,
                stroke_id=None, edge_index_stroke=None, edge_index_intra_stroke=None,
                cont_stroke_id=None, edge_index_cont_stroke=None, edge_index_intra_cont=None,
                edge_relation_mask=None, line_pos=None, region_pos=None, region_shape_feat=None):
        # X_l 输入标准化 (z-score): 训练前注入的逐维统计量, 消除各维尺度异质
        #   (straightness/弧长量级大, frt/boundary_share ∈[0,1], 不标准化则首层
        #   Linear 被高方差维度主导)。未注入时 buffer 为 mean=0/std=1 → 恒等。
        X_l_enc = (X_l - self.x_l_mean) / self.x_l_std

        h_l = self.line_proj(self.line_encoder(X_l_enc))
        if ENABLE_LINE_STREAM and color_index is not None:
            color_index = color_index.clamp(0, self.num_colors - 1)
            h_color = self.color_proj(self.color_embed(color_index))
            h_l = h_l + h_color

        h_l_base = h_l
        edge_attr_ll_enc = self._encode_edge_attr(edge_attr_ll)

        # 无向 L-L 边 → 双向展开: NBA-GAT 消息传递需双向有向边; edge_attr 已无向, 直接复制一份。
        if edge_index_ll.size(1) > 0:
            _src, _dst = edge_index_ll[0], edge_index_ll[1]
            edge_index_ll = torch.cat(
                [edge_index_ll, torch.stack([_dst, _src], dim=0)], dim=1)
            edge_attr_ll_enc = torch.cat([edge_attr_ll_enc, edge_attr_ll_enc], dim=0)
            if edge_relation_mask is not None:
                edge_relation_mask = torch.cat([edge_relation_mask, edge_relation_mask], dim=0)
        if edge_relation_mask is None:
            edge_relation_mask = torch.zeros(
                (edge_index_ll.size(1), EDGE_RELATION_DIM),
                device=h_l.device, dtype=h_l.dtype)

        # DropEdge (共享 mask): 每个训练步采样一次边 mask, 所有 GNN 层复用。
        backtrack_node = None
        if DROPEDGE_RATE > 0.0 and self.training and edge_index_ll.size(1) > 0:
            if DROPEDGE_GROUP:
                keep = dropedge_group_mask(edge_index_ll, stroke_id, cont_stroke_id, DROPEDGE_RATE)
            else:
                keep = dropedge_keep_connected_mask(edge_index_ll, h_l.size(0), DROPEDGE_RATE)
            ei = edge_index_ll[:, keep]
            ea = edge_attr_ll_enc[keep]
            erm = edge_relation_mask[keep]
        else:
            ei = edge_index_ll
            ea = edge_attr_ll_enc
            erm = edge_relation_mask

        # 每层细图 GNN 内嵌 CoRe 并行粗支路 (两套分组: 角点→笔画 / 方向连续→笔画) + Region 粗支路。
        num_regions = region_line_color_geo.size(0) if region_line_color_geo is not None else (
            int(line_region_a.max().item()) + 1 if line_region_a is not None and line_region_a.numel() > 0 else 0)
        for layer in self.gnn_layers:
            h_l, backtrack_node = layer(h_l, ei, ea, erm, backtrack_node,
                                        stroke_id, edge_index_stroke, edge_index_intra_stroke,
                                        cont_stroke_id, edge_index_cont_stroke, edge_index_intra_cont,
                                        line_region_a, line_region_b, edge_index_rr, num_regions)

        if self.num_gnn_layers > 0:
            residual = h_l - h_l_base
            h_l = h_l_base + self.gnn_dropout(residual)
        else:
            h_l = self.gnn_dropout(h_l)

        if ENABLE_RCOLOR_STREAM and region_line_color_geo is not None:
            h_r = self.region_color_compress(region_line_color_geo)
        else:
            h_r = torch.zeros(region_line_color_geo.size(0), self.color_embed_dim,
                                    device=region_line_color_geo.device, dtype=region_line_color_geo.dtype)

        # --- 位置嵌入注入 (独立分支, E4): GNN 之后加性注入, 不参与消息传递 ---
        # line_pos → h_l; region_pos → h_r。默认 None 时安全退化 (不注入)。
        if line_pos is not None:
            h_l = h_l + self.line_pos_encoder(line_pos)
        if region_pos is not None:
            h_r = h_r + self.region_pos_encoder(region_pos)
        if region_shape_feat is not None:
            h_r = h_r + self.region_shape_scale * self.region_shape_encoder(region_shape_feat)

        return h_l, h_r

    def encode_sides(self, h_l, h_r, line_region_a, line_region_b, line_plus_region_a=None):
        """侧编码 → (h_line, side_sum, z_plus, z_minus)。

        - side_sum:  对两侧调制后的 Region 表示，分别按正序/反序输入共享 MLP 再平均；
          A↔B 交换不变，供无序 Triple Head。
        - z_plus / z_minus: 两侧分别调制后加回同一 Line 特征 (h_line+s_side)，
          供共享权重的 Side Head 逐侧判色；没有独立 Direction Head。
        - line_plus_region_a: (N,) bool, region_a 是否在 +n 几何法向侧。提供时按几何法向侧重排
          (side_plus=+n 侧), 方向不再绑定大/小区域人为顺序; None 时退化为大/小区域(旧行为)。
        """
        n_lines = line_region_a.size(0)
        device = line_region_a.device
        line_idx = torch.arange(n_lines, device=device)
        h_line = h_l[line_idx]
        h_r_a = h_r[line_region_a]
        h_r_b = h_r[line_region_b]
        if line_plus_region_a is not None:
            plus_a = line_plus_region_a.to(torch.bool).unsqueeze(1)
            h_r_plus = torch.where(plus_a, h_r_a, h_r_b)
            h_r_minus = torch.where(plus_a, h_r_b, h_r_a)
        else:
            h_r_plus = h_r_a
            h_r_minus = h_r_b
        # --- 逐 node、逐侧的 region 注入幅度 ---
        #   分类哪一侧, 就只用那一侧的 region 上下文生成幅度 —— 不能把对侧的 h_r 也喂进来,
        #   否则等于在判断 +n 侧时偷看了 -n 侧的 region。
        #   两侧共用同一个门控网络 (权重共享), 因此 A/B 交换等变。
        #   2*sigmoid 使初始值恰为 1.0 (末层零初始化), 起手与基线逐位一致。
        alpha_plus = 2.0 * torch.sigmoid(
            self.side_region_gate(torch.cat([h_line, h_r_plus], dim=-1)))
        alpha_minus = 2.0 * torch.sigmoid(
            self.side_region_gate(torch.cat([h_line, h_r_minus], dim=-1)))
        side_plus = alpha_plus * self.side_region_encoder(h_r_plus)
        side_minus = alpha_minus * self.side_region_encoder(h_r_minus)
        # triple: DeepSets 对称化 —— 两个 region 拼接后按两种顺序各过同一个 MLP, 再平均。
        #   (g(x,y)+g(y,x))/2 严格顺序无关 (A↔B 交换不变), 且允许网络学习两侧的交互项。
        r_plus_m = alpha_plus * h_r_plus
        r_minus_m = alpha_minus * h_r_minus
        t_fwd = self.triple_pair_encoder(torch.cat([r_plus_m, r_minus_m], dim=-1))
        t_rev = self.triple_pair_encoder(torch.cat([r_minus_m, r_plus_m], dim=-1))
        side_sum = 0.5 * (t_fwd + t_rev)
        # side: 两侧分别调制, 再各自加回本 node 一次 (z_side 只含该侧自己的 region 信息)
        #   s_side 维度是 cls_side_dim, h_line 是 hidden_dim, 需先投影到同一空间才能相加。
        z_plus = h_line + self.side_node_proj(side_plus)
        z_minus = h_line + self.side_node_proj(side_minus)
        return h_line, side_sum, z_plus, z_minus


class ClassifierHead(nn.Module):
    """每数据集独立的分类头: 输入 (h_line, side_sum) → logits (标准 CE)。

    logits 供三元组联合分类 (标准线性头)。num_triples 随数据集的目标调色板
    (color_palette) 变化, 故此头必须每个数据集独立, 不参与 backbone 共享。
    """

    def __init__(self, num_triples, num_classes, hidden_dim=HIDDEN_DIM, cls_side_dim=CLASSIFIER_SIDE_DIM, inv_map=None):
        super().__init__()
        self.num_triples = num_triples
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.cls_side_dim = cls_side_dim
        self.inv_map = inv_map  # (num_triples,3) triple 色对与线色映射；无独立方向头
        cls_input_dim = hidden_dim + cls_side_dim

        self.classifier_trunk_p1 = nn.Sequential(
            nn.Linear(cls_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.1),
        )
        # triple 主干: 投影 + 2 个残差 FFN。容量刻意大于 side —— triple 做 NT(85~101) 类
        #   联合分类, 任务显著更难; 实测把两侧容量拉平 (triple 减为 1 个 FFN + side 加 FFN)
        #   会让 B -1.27pp / V -1.57pp, 故保留不对称。
        self.cls_norm1 = nn.LayerNorm(hidden_dim)
        self.cls_ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cls_norm2 = nn.LayerNorm(hidden_dim)
        self.cls_ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cls_dropout = nn.Dropout(0.1)
        self.cls_norm_out = nn.LayerNorm(hidden_dim)
        self.cls_act_out = nn.GELU()

        # 独立 Direction Head 已删除；方向由 Side Head 的 +n/-n 条件分布提供，
        # Triple 色对先验与 Side 两侧似然融合后进入 MRF unary。

        # Side-head (P1, deep-research-report2): 直接预测每线两侧颜色 (解耦 triple 联合分类)。
        #   输入 z_side = h_line + s_side → num_classes, 同一个头对两侧各调一次, 堆叠为 (N, 2, C)。
        # Side-head (P1): 逐侧颜色分类, 余弦分类器。
        #   注意: **不共用 triple 的表征空间** —— triple 判"两侧色对的组合"(纹理语义),
        #   side 只判"这一侧属于哪个 region"(单侧语义), 两者语义不同, 因此 side 投影到
        #   自己的 cls_side_dim(32) 维子空间, 而不是复用 hidden_dim(48)。
        #   余弦形式 (归一化特征 · 归一化类权重 × 温度) 与 triple 一致, 使两者 logits 尺度可比,
        #   这样 "置信度加权融合" 里的平权才有意义。
        self.side_head = nn.Sequential(
            nn.Linear(hidden_dim, cls_side_dim),
            nn.LayerNorm(cls_side_dim), nn.GELU(),
            nn.Dropout(0.1),
        )
        self.side_weight = nn.Parameter(torch.empty(num_classes, cls_side_dim))
        nn.init.xavier_uniform_(self.side_weight)
        self.side_scale = nn.Parameter(torch.tensor(COS_SCALE_INIT))   # 温度, 与 triple 的 cls_scale 对齐
        # RBF 核宽度 (可学习): γ = exp(log_γ) / S, 初始使 exp(-γ·2S) ≈ e^-1
        self.side_log_gamma = nn.Parameter(torch.tensor(float(np.log(1.0 / max(cls_side_dim, 1)))))

        # --- 余弦相似度分类头 (分布无关, AnT 式): 替换线性 logits 的分布依赖 ---
        # 线性分类器权重范数 ≈ 类频率 → 稀有类(鼻子/爪子)权重小 → logits 恒低。余弦头把
        #   决策从"权重范数(频率)"解耦为"角度对齐", 稀有类与常见类同一起跑线。
        # 只作用于色对 logits (num_triples 维); dir 仍走线性头 (二分类不受频率影响)。
        self.cls_weight = nn.Parameter(torch.empty(num_triples, hidden_dim))
        nn.init.xavier_uniform_(self.cls_weight)
        self.cls_scale = nn.Parameter(torch.tensor(COS_SCALE_INIT))   # 温度缩放, 控制 logits 动态范围
        # 可学习 hinge margin: margin = 0.5 + softplus(raw), 初始 raw=1.0 -> ≈1.81, 下界 0.5。
        #   固定 margin 的问题: 余弦 logits 的尺度在训练中漂移, 早期几乎全部负类都在违反 (梯度饱和),
        #   后期又全部满足 (排序项失效)。下界 0.5 防止 margin 塌缩到 0 使 hinge 失去意义。
        self.rank_margin_raw = nn.Parameter(torch.tensor(1.0))
        # RBF 核宽度 (可学习): γ = exp(log_γ)/H, 初始使 exp(-γ·2H) ≈ e^-1
        self.triple_log_gamma = nn.Parameter(torch.tensor(float(np.log(1.0 / max(hidden_dim, 1)))))

    # 三组预测头分开编排，但不改变任何 nn.Module/Parameter 名称：旧 checkpoint 严格兼容。
    # Triple/Side/Dir 共享输入表示，并非统计独立的三套观测。
    def rank_margin(self):
        """可学习 hinge margin (下界 0.5, 无上界但由 softplus 平滑增长)。"""
        return 0.5 + F.softplus(self.rank_margin_raw)

    def _trunk_feat(self, h_line, side_sum):
        """triple trunk: [h_line, side_sum] → hidden_dim 特征。"""
        cls_input = torch.cat([h_line, side_sum], dim=1)
        h = self.classifier_trunk_p1(cls_input)
        h = h + self.cls_dropout(self.cls_ffn1(self.cls_norm1(h)))
        h = h + self.cls_dropout(self.cls_ffn2(self.cls_norm2(h)))
        return self.cls_act_out(self.cls_norm_out(h))

    def forward_triple(self, h_line, side_sum):
        """无序色对分类。

        TRIPLE_CLASSIFIER="cosine": 归一化特征 · 归一化类权重 × 温度 (线性余弦头)
        TRIPLE_CLASSIFIER="rbf"   : s_t = -S·γ·‖feat-μ_t‖²,  μ_t 即 cls_weight 的第 t 行
        """
        feat = self._trunk_feat(h_line, side_sum)
        if str(TRIPLE_CLASSIFIER).lower() == "rbf":
            # 注意: 不做归一化 —— 否则 RBF 退化为余弦的单调函数。
            g = torch.exp(self.triple_log_gamma)
            d2 = (feat.unsqueeze(1) - self.cls_weight.unsqueeze(0)).pow(2).sum(dim=-1)
            return -float(TRIPLE_RBF_SCALE) * g * d2
        feat_norm = F.normalize(feat, dim=1)
        w_norm = F.normalize(self.cls_weight, dim=1)
        return self.cls_scale * (feat_norm @ w_norm.t())

    def _side_logits(self, z):
        """单个 z → (N, C) 颜色 logits。

        SIDE_CLASSIFIER="cosine": 归一化特征 · 归一化类权重 × 温度 (线性 / 余弦头)
        SIDE_CLASSIFIER="rbf"   : s_c = scale · exp(-γ‖feat-μ_c‖²)  (RBF 核分类器)
        """
        feat = self.side_head(z)
        if str(SIDE_CLASSIFIER).lower() == "rbf":
            # 注意: 此处**不做归一化** —— 归一化后 RBF 退化为余弦的单调函数, 排序完全相同。
            # 用 **log-RBF** (高斯判别形式) 而非 exp(-γd²):
            #   s_c = -S·γ·‖feat-μ_c‖²
            # 直接 exp 的版本 logits 动态范围只有 ~1.8, softmax 过平, 置信度 conf=1-H/logC 偏低,
            # 会在 "置信度加权投票" 里被自动降权 (实测 B −3.8pp)。log 形式保留同一核结构但量程正常。
            g = torch.exp(self.side_log_gamma)
            d2 = (feat.unsqueeze(1) - self.side_weight.unsqueeze(0)).pow(2).sum(dim=-1)
            return -float(SIDE_RBF_SCALE) * g * d2
        feat_norm = F.normalize(feat, dim=1)
        w_norm = F.normalize(self.side_weight, dim=1)
        return self.side_scale * (feat_norm @ w_norm.t())

    def forward_side(self, z_plus, z_minus):
        """逐侧颜色分类: 同一套余弦头对两侧各调一次 (共享权重 => A/B 等变)。"""
        return torch.stack([self._side_logits(z_plus), self._side_logits(z_minus)], dim=1)

    def forward(self, h_line, side_sum, z_plus, z_minus):
        """返回 (triple logits, side logits)。"""
        logits = self.forward_triple(h_line, side_sum)
        side_logits = self.forward_side(z_plus, z_minus)
        return logits, side_logits



class LineColorizerGNN(nn.Module):
    """Line-Centric GNN：5D Line / 2D Edge、细图与 Region/Stroke 并行支路。

    Backbone 提供 Line 与 Region 的嵌入；ClassifierHead 提供无序 Triple
    与共享参数的 +n/-n Side 两个分类头。Region 颜色经 MRF 解码。
    图提取算法 v65 保留短枝、原始微小 Region 和四边接触，并改进局部斜率和双侧归属；不改变模型参数结构。
    """

    def __init__(self, num_classes, num_triples, line_in_dim=LINE_FEAT_DIM, edge_attr_dim=EDGE_ATTR_DIM,
                 hidden_dim=HIDDEN_DIM,
                 line_enc_dim=LINE_ENC_DIM, edge_enc_dim=EDGE_ENC_DIM,
                 num_gnn_layers=GNN_LAYERS, max_line_colors=MAX_LINE_COLORS,
                 color_embed_dim=COLOR_EMB_DIM, inv_triple_map=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_triples = num_triples
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_colors = max_line_colors

        # --- 共享判别器 backbone + 单数据集分类头 ---
        self.backbone = LineColorizerBackbone(
            line_in_dim=line_in_dim, edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim, line_enc_dim=line_enc_dim,
            edge_enc_dim=edge_enc_dim, num_gnn_layers=num_gnn_layers,
            max_line_colors=max_line_colors, color_embed_dim=color_embed_dim,
            cls_side_dim=CLASSIFIER_SIDE_DIM)
        # 构建 inv_map (triple 索引 → (min色, max色, 线色))，供 Triple×Side 融合使用。
        _inv_map = build_inv_map_array(inv_triple_map, num_triples) if inv_triple_map is not None else None
        if _inv_map is not None:
            _inv_map = torch.as_tensor(_inv_map, dtype=torch.long)
        self.head = ClassifierHead(num_triples, num_classes=num_classes, hidden_dim=hidden_dim,
                                   cls_side_dim=CLASSIFIER_SIDE_DIM, inv_map=_inv_map)

    def _encode_edge_attr(self, edge_attr):
        return self.backbone._encode_edge_attr(edge_attr)

    def forward(self, X_l, edge_index_ll, edge_attr_ll,
                color_index=None, edge_index_rr=None,
                line_region_a=None, line_region_b=None,
                region_line_color_geo=None,
                stroke_id=None, edge_index_stroke=None, edge_index_intra_stroke=None,
                cont_stroke_id=None, edge_index_cont_stroke=None, edge_index_intra_cont=None,
                edge_relation_mask=None, line_pos=None, region_pos=None, region_shape_feat=None):
        """GNN 前向: 委托给共享 backbone, 返回 (h_l, h_r)。"""
        return self.backbone(X_l, edge_index_ll, edge_attr_ll,
                             color_index, edge_index_rr, line_region_a, line_region_b,
                             region_line_color_geo, stroke_id, edge_index_stroke, edge_index_intra_stroke,
                             cont_stroke_id, edge_index_cont_stroke, edge_index_intra_cont,
                             edge_relation_mask, line_pos, region_pos, region_shape_feat)

    # ==================== Line-centric 分类 ====================

    @staticmethod
    def _build_line_region_pairs(line_region_a, line_region_b,
                                  line_label_a, line_label_b):
        """直接使用 extract_ll_graph 中沿路径多点投票预计算的 per-line region.

        Returns:
            line_idx (N_l,), region_a (N_l,), region_b (N_l,),
            label_a (N_l,), label_b (N_l,)
        """
        n_lines = line_region_a.size(0)
        device = line_region_a.device
        line_idx = torch.arange(n_lines, device=device)
        return line_idx, line_region_a, line_region_b, line_label_a, line_label_b

    def classify_lines(self, h_l, h_r,
                       line_region_a, line_region_b,
                       line_label_a, line_label_b,
                       line_plus_region_a=None):
        """
        Returns:
            logits: (N_l, num_triples) 联合分类 logits
            side_logits: (N_l, 2, C) 逐侧颜色 logits
            line_idx: (N_l,)
            region_a, region_b: (N_l,) 区域索引
            label_a, label_b: (N_l,) 真实颜色标签（用于验证/解码）
        """
        line_idx, region_a, region_b, label_a, label_b = self._build_line_region_pairs(
            line_region_a, line_region_b, line_label_a, line_label_b)

        # 对称侧编码走共享 backbone (cls_side_encoder), 分类 trunk+head 走独立分类头
        h_line, side_sum, z_plus, z_minus = self.backbone.encode_sides(
            h_l, h_r, line_region_a, line_region_b, line_plus_region_a)
        logits, side_logits = self.head(
            h_line, side_sum, z_plus, z_minus)

        return (logits, side_logits,
                line_idx, region_a, region_b, label_a, label_b)


# ==================== 5. 训练与推理 ====================

def load_checkpoint(path, device=None, with_meta=False):
    """加载 checkpoint, 兼容两种格式:

    - 新格式: {'model_state': state_dict, 'line_palette': {...}, 'color_palette': {...}}
    - 旧格式: 裸 state_dict (无调色板信息)

    默认返回 state_dict; with_meta=True 时返回 (state_dict, meta|None)。
    """
    try:
        obj = torch.load(path, map_location=device or 'cpu', weights_only=True)
    except Exception:
        # 旧 checkpoint 的调色板可能含 numpy 标量, weights_only 不放行;
        # 文件为本地训练产物, 退化为常规加载。
        obj = torch.load(path, map_location=device or 'cpu', weights_only=False)
    if isinstance(obj, dict) and 'model_state' in obj:
        sd = obj['model_state']
        meta = {k: v for k, v in obj.items() if k != 'model_state'}
    else:
        sd, meta = obj, None
    return (sd, meta) if with_meta else sd


# ==================== checkpoint 安全层 (R2-A) ====================
# 目标: (1) 实验身份可判定 (dataset/seed/结构/C/T/triple 指纹/代码指纹);
#       (2) 已存在的 checkpoint 绝不被静默覆盖; (3) 写入原子化, 中断不留半成品;
#       (4) 加载失败绝不回退到"重新训练"。
# 只涉及加载/保存的 I/O 与校验, 不改变任何前向/损失/后处理数值。


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def _fp_of(obj, n=12):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:n]


def code_fingerprint():
    """主程序"结构版本"指纹 (只用于身份判定, 不参与数值)。

    读入源码后把 `INPUT_ROOT = ...` 这一行按占位符归一化再哈希:
      · 实验运行目录里的副本会被 run1.prepare 改写 INPUT_ROOT 指向具体数据集;
      · 若直接哈希文件字节, 同一份代码在不同数据集目录下会得到不同指纹,
        导致 SVD 迁移出来的 checkpoint 被结构守卫误判为"指纹不符"。
    归一化后, "代码内容(除数据路径外)"相同的副本共享同一指纹。
    """
    try:
        txt = open(os.path.abspath(__file__), encoding='utf-8').read()
        txt = re.sub(r'^INPUT_ROOT\s*=\s*["\'][^"\']*["\']',
                     'INPUT_ROOT = "<DATASET>"', txt, count=1, flags=re.M)
        return hashlib.sha256(txt.encode('utf-8')).hexdigest()[:12]
    except Exception:
        return "unknown"


def build_structure_meta(dataset, seed, line_palette, color_palette, triple_map,
                         hidden_dim=None, gnn_layers=None, stroke_branch=None,
                         stroke_stat_dim=None, stroke_lowrank_rank=None):
    """生成写入 checkpoint 的结构/实验身份信息 (纯元数据)。"""
    vs = {"hidden_dim": int(HIDDEN_DIM if hidden_dim is None else hidden_dim),
          "gnn_layers": int(GNN_LAYERS if gnn_layers is None else gnn_layers),
          "stroke_branch": str(STROKE_BRANCH if stroke_branch is None else stroke_branch),
          "stroke_stat_dim": int(STROKE_STAT_DIM if stroke_stat_dim is None else stroke_stat_dim),
          "stroke_lowrank_rank": int(STROKE_LOWRANK_RANK if stroke_lowrank_rank is None
                                     else stroke_lowrank_rank),
          "stroke_branch_mode": str(STROKE_BRANCH_MODE),
          "stroke_mlp_hidden": int(STROKE_MLP_HIDDEN),
          "stroke_mlp_layers": int(STROKE_MLP_LAYERS),
          "num_classes": int(len(color_palette)),
          "num_triples": int(len(triple_map)),
          "triple_fp": _fp_of([list(map(int, k)) for k in sorted(triple_map)]),
          "line_palette_fp": _fp_of({int(k): tuple(int(x) for x in v)
                                     for k, v in sorted(line_palette.items())}),
          "color_palette_fp": _fp_of({int(k): tuple(int(x) for x in v)
                                      for k, v in sorted(color_palette.items())}),
          "code_fp": code_fingerprint()}
    meta = dict(vs)
    meta["dataset"] = str(dataset)
    meta["seed"] = None if seed is None else int(seed)
    meta["structure_id"] = _fp_of(vs)
    return meta


def save_checkpoint_atomic(path, payload, allow_overwrite=False):
    """原子写 checkpoint: 先写临时文件再 os.replace; 默认拒绝覆盖已存在文件。"""
    path = os.fspath(path)
    if os.path.exists(path) and not allow_overwrite:
        raise FileExistsError(
            f"checkpoint 已存在, 拒绝覆盖: {path}\n"
            f"  如确需覆盖请显式 --allow-overwrite-checkpoint (会先备份) 或改用新路径")
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        torch.save(payload, tmp)
        if os.path.exists(path) and allow_overwrite:
            bak = f"{path}.bak{time.strftime('%Y%m%d-%H%M%S')}"
            os.replace(path, bak)
            print(f"  [覆盖保护] 旧 checkpoint 已备份: {bak}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_checkpoint_guarded(path, model, device, expect_meta=None, strict=True):
    """强制按精确路径加载并校验; 失败直接抛错, 绝不回退重训。

    expect_meta: 当前运行的结构身份 (build_structure_meta 的返回值)。
    返回 (meta, missing_keys, unexpected_keys)。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在 (拒绝回退训练): {path}")
    sd, meta = load_checkpoint(path, device, with_meta=True)
    try:
        result = model.load_state_dict(sd, strict=strict)
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint 与当前模型结构不兼容, 已终止 (拒绝删除/重训): {path}\n"
            f"  结构提示: {meta.get('structure_id') if isinstance(meta, dict) else None} "
            f"vs 当前 {expect_meta.get('structure_id') if expect_meta else None}\n"
            f"  原始错误: {exc}") from exc
    if expect_meta:
        # 旧格式 checkpoint 没有 *_fp 字段, 但存了调色板本体 -> 现场算指纹再比对,
        # 这样"配色表不一致"对老权重也能拦下来。
        if isinstance(meta, dict):
            for key, src in (("line_palette_fp", "line_palette"),
                             ("color_palette_fp", "color_palette")):
                if meta.get(key) is None and isinstance(meta.get(src), dict):
                    meta[key] = _fp_of({int(k): tuple(int(x) for x in v)
                                        for k, v in sorted(meta[src].items())})
        for key in ("num_classes", "num_triples", "triple_fp",
                    "line_palette_fp", "color_palette_fp"):
            got = meta.get(key) if isinstance(meta, dict) else None
            if got is not None and got != expect_meta.get(key):
                raise RuntimeError(
                    f"checkpoint 的 {key} 与当前数据不一致 ({got} != {expect_meta.get(key)}); "
                    f"拒绝在错配的类别/调色板/triple 表上推理: {path}")
        sid = meta.get("structure_id") if isinstance(meta, dict) else None
        if sid is not None and sid != expect_meta.get("structure_id"):
            raise RuntimeError(
                f"checkpoint 结构指纹不匹配 (文件中 {sid} != 当前 "
                f"{expect_meta.get('structure_id')}); 拒绝加载: {path}")
    return meta, missing, unexpected


def _plot_training_curves(history):
    """绘制训练过程折线图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history['epochs']

    # 过滤掉 None (非验证 epoch 记录 None), 仅画有值点, 避免稀疏验证下断线/空序列崩溃。
    # RegionAcc 等 val_* 曲线只含在验证 epoch 处的数据点。
    def _s(ys):
        xy = [(e, y) for e, y in zip(epochs, ys) if y is not None]
        return ([p[0] for p in xy], [p[1] for p in xy]) if xy else (None, None)

    # 左图: Loss 曲线
    ax1 = axes[0]
    tx_l, ty_l = _s(history['train_loss'])
    vx_l, vy_l = _s(history['val_loss'])
    ax1.plot(tx_l, ty_l, 'b-', label='Train Triple Loss', linewidth=2)
    if vx_l:
        ax1.plot(vx_l, vy_l, 'r-', label='Val Triple Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右图: 图节点 RegionAcc（包括未投票 Region）；同时绘制投票覆盖率。
    # Direction Head 已删除，因此不展示虚假的 DirAcc=0 曲线。
    ax2 = axes[1]
    for key, label, style in (
        ('train_region_acc', 'Train Graph RegionAcc (all)', 'c--'),
        ('val_region_acc', 'Val Graph RegionAcc (all)', 'g-'),
        ('train_region_coverage', 'Train Vote Coverage', 'b:'),
        ('val_region_coverage', 'Val Vote Coverage', 'm:'),
    ):
        xx, yy = _s(history.get(key, []))
        if xx:
            ax2.plot(xx, yy, style, label=label, linewidth=1.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Fraction', fontsize=12)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title('Graph Region Accuracy & Vote Coverage', fontsize=12, fontweight='bold')
    if ax2.lines:
        ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'Debug/training_curves_v6.png'
    os.makedirs('Debug', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 训练曲线已保存到: {save_path}")
    plt.close()

# ==================== 8. 损失景观 3D 可视化（Loss Landscape） ====================
def plot_loss_landscape_3d(
    model,
    train_loader,
    device,
    inv_triple_map,
    val_loader=None,
    steps: int = 41,
    range_alpha: float = 0.20,
    range_beta: float = 0.20,
    seed: int = 20260905,
    num_directions: int = 5,
    epoch: int = 999,
    landscape_loss: str = "main",
    output_html: str = "Debug/loss_landscape_3d.html",
):
    """完整、可比较的多方向 Loss Landscape。

    landscape_loss:
      'main'  (默认) — train 面与 val 面都取 loss_main (纯 CE). 二者同量纲, 可逐点严格
                        对比 (Train/Val/Gap 子图、center gap、曲率 gap 均直接可比)。
                        这是泛化性景观分析的正确口径。
      'total'        — train 面取 loss_total (main+proto+region, 可学习 log-var 加权), 复现
                        "优化器真正最小化的目标"; 但 val 面仍为 main (val 数据不足以算
                        proto/region), 量纲不同, train/val 不可比. 仅当你想让 train 面
                        对齐训练曲线量纲时选用.
    """
    import copy

    if landscape_loss not in ("main", "total"):
        raise ValueError(f"landscape_loss 必须是 'main' 或 'total'，当前为 {landscape_loss}")
    if steps < 3 or steps % 2 == 0:
        raise ValueError(f"Loss Landscape 的 steps 必须是 >=3 的奇数，当前为 {steps}")
    if num_directions < 1:
        raise ValueError("num_directions 必须 >= 1")
    if train_loader is None:
        raise ValueError("train_loader 不能为 None")
    if range_alpha <= 0 or range_beta <= 0:
        raise ValueError("range_alpha / range_beta 必须 > 0")

    model.eval()

    param_items = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    if not param_items:
        raise RuntimeError("模型没有 requires_grad=True 的参数，无法计算 loss landscape")

    original_weights = {name: param.detach().clone() for name, param in param_items}
    # 串行加速：编译「前向+分类」为单一计算图 (torch.compile)。
    # 失败(Windows inductor 受限 / torch<2.0)则回退未编译前向, 功能不受影响。
    # 首次调用触发编译(略慢), 之后对同一形状批量复用, CPU 上通常 1.2~2x 吞吐提升。
    _ll_use_compile = False
    _ll_compiled = None
    try:
        # 前置探测本机 C 编译器(indutor 必需): 没有则立即跳过, 避免编译失败白白耗时。
        import shutil, sys
        if sys.platform == "win32":
            _cc = shutil.which("cl") or shutil.which("cl.exe")
        else:
            _cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
        if _cc is None and os.environ.get("LANDSCAPE_COMPILE") != "1":
            raise RuntimeError("未检测到 C 编译器(cl/gcc), 跳过 torch.compile")
        if _cc is None:
            print("[损失景观] LANDSCAPE_COMPILE=1 强制尝试 torch.compile(本机可能无 C 编译器)")

        @torch.compile(dynamic=True)
        def _ll_compiled_eval(batch):
            h_l, h_r = model(
                batch["X_l"], batch["edge_index_ll"], batch["edge_attr_ll"],
                batch["color_index"], batch["edge_index_rr"],
                batch["line_region_a"], batch["line_region_b"],
                region_line_color_geo=batch["region_line_color_geo"],
                edge_relation_mask=batch["edge_relation_mask"],
                line_pos=batch.get("line_pos"),
                region_pos=batch.get("region_pos"), region_shape_feat=batch.get("region_shape_feat"),
                            )
            # 返回 classify_lines 完整 6 元组 (含 region_a/b, label_a/b),
            # 供 compute_loss 计算与训练完全一致的 loss.
            out = model.classify_lines(
                h_l, h_r,
                batch["line_region_a"], batch["line_region_b"],
                batch["line_label_a"], batch["line_label_b"],
            )
            return out

        _ll_compiled = _ll_compiled_eval
        _ll_use_compile = True
    except Exception as _ll_ce:  # pragma: no cover
        print(f"[损失景观] torch.compile 不可用/跳过({_ll_ce!r}), 回退未编译前向")

    def _ll_logits(batch):
        nonlocal _ll_use_compile
        if _ll_use_compile and _ll_compiled is not None:
            try:
                return _ll_compiled(batch)
            except Exception as _ll_ce:  # pragma: no cover
                # 编译后运行期异常(如 inductor 不支持某算子): 仅首点触发一次,
                # 之后永久回退未编译前向, 保证 landscape 不因 compile 崩溃。
                print(f"[损失景观] torch.compile 运行期失败({_ll_ce!r}), 回退未编译前向")
                _ll_use_compile = False
        h_l, h_r = model(
            batch["X_l"], batch["edge_index_ll"], batch["edge_attr_ll"],
            batch["color_index"], batch["edge_index_rr"],
            batch["line_region_a"], batch["line_region_b"],
            region_line_color_geo=batch["region_line_color_geo"],
            edge_relation_mask=batch["edge_relation_mask"],
            line_pos=batch.get("line_pos"),
            region_pos=batch.get("region_pos"), region_shape_feat=batch.get("region_shape_feat"),
                    )
        # 返回 classify_lines 完整 6 元组 (含 region_a/b, label_a/b),
        # 供 compute_loss 计算与训练完全一致的 loss.
        out = model.classify_lines(
            h_l, h_r,
            batch["line_region_a"], batch["line_region_b"],
            batch["line_label_a"], batch["line_label_b"],
        )
        return out


    # 固定完整数据：同一次 landscape 中所有方向/所有网格点都访问完全相同的数据。
    def _materialize_loader(loader, tag):
        batches = []
        valid_count = 0
        for batch in loader:
            fixed = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    fixed[k] = v.to(device, non_blocking=True)
                else:
                    fixed[k] = copy.deepcopy(v)
            batches.append(fixed)
            labels = batch.get("triple_label")
            if torch.is_tensor(labels):
                valid_count += int((labels >= 0).sum().item())
        if not batches:
            raise RuntimeError(f"{tag} loader 为空，无法计算 loss landscape")
        return batches, valid_count

    train_batches, train_valid = _materialize_loader(train_loader, "Train")
    if val_loader is not None:
        val_batches, val_valid = _materialize_loader(val_loader, "Val")
        has_val = True
    else:
        val_batches, val_valid = None, 0
        has_val = False
    print(f"[损失景观] data: train={len(train_batches)} batch(es) ({train_valid} valid) | "
          f"val={len(val_batches) if val_batches else 0} batch(es) ({val_valid} valid)")
    if landscape_loss == 'total':
        print("[损失景观] loss口径: train=total(main+proto+region) | val=main  → 注意: 量纲不同, train/val 不可比")
    else:
        print("[损失景观] loss口径: train=main | val=main  → 两网格同量纲, 可逐点对比 (Train/Val/Gap)")

    # 损失口径: 始终经 compute_loss (唯一损失实现), 但选择返回哪个标量由 landscape_loss 决定。
    #   mode='train'/'val' 均走同一分支 → 保证两网格 loss 完全同算法、可逐点对比:
    #     先算 loss_main; 仅当 landscape_loss=='total' 且 mode=='train' 时才把 proto/region
    #     计入. val 恒为 main (val 数据不足以算 proto/region, 与训练期 val 评估口径一致).
    #   聚合: 对每个 batch 的所选 loss 标量取均值 (与训练期按 batch 累积 / num_batches
    #           口径一致), 跳过无有效 triple_label 的 batch.
    def _eval_loss_on_batches(batches, mode):
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        n_batches = 0
        for batch in batches:
            out = _ll_logits(batch)
            logits, _dir, region_a, region_b, label_a, label_b = out
            valid = batch["triple_label"] >= 0
            if int(valid.sum().item()) <= 0:
                continue
            losses = compute_loss(
                logits=logits,
                triple_label=batch["triple_label"],
                region_a=region_a,
                region_b=region_b,
                line_pixel_count=batch["line_pixel_count"],
                num_regions=batch["y"].size(0),
                num_classes=model.num_classes,
                inv_triple_map=inv_triple_map,
                mode='val',
            )
            loss_sum += losses['main'].double()
            n_batches += 1
        if n_batches <= 0:
            raise RuntimeError("Landscape evaluation 中没有有效 triple_label")
        return loss_sum / n_batches
    
    def _make_gradient_directions(model, train_batches, val_batches, device, inv_triple_map):
        """
        返回 (d_mean, d_conflict):
        d_mean     = 平均梯度方向 (train+val 都下降最快的方向)
        d_conflict = (val_grad - train_grad) 方向 (两个 loss 分歧最大的方向)
        """
        def _collect_grads(batches, mode):
            grads = []
            # 保存当前训练模式, 结束时恢复 —— 否则 _collect_grads 打开 train() 后
            # 不再切回 eval(), 整个景观评估循环都带 dropout/DropEdge 随机 mask,
            # train/val 面每个 (α,β) 点都是随机采样, 不可逐点严格对比.
            _was_training = model.training
            model.train()          # 打开 dropout/dropedge 让梯度有随机性, 但要固定随机种子
            try:
                for batch in batches:
                    model.zero_grad()
                    h_l, h_rc = model(
                        batch["X_l"], batch["edge_index_ll"], batch["edge_attr_ll"],
                        batch["color_index"], batch["edge_index_rr"],
                        batch["line_region_a"], batch["line_region_b"],
                        region_line_color_geo=batch["region_line_color_geo"],
                        edge_relation_mask=batch["edge_relation_mask"],
                        line_pos=batch.get("line_pos"),
                        region_pos=batch.get("region_pos"), region_shape_feat=batch.get("region_shape_feat"),
                                            )
                    out = model.classify_lines(
                        h_l, h_rc,
                        batch["line_region_a"], batch["line_region_b"],
                        batch["line_label_a"], batch["line_label_b"],
                    )
                    logits, _dir, _, ra, rb, la, lb = out
                    losses = compute_loss(
                        logits=logits, triple_label=batch["triple_label"],
                        region_a=ra, region_b=rb,
                        line_pixel_count=batch["line_pixel_count"],
                        num_regions=batch["y"].size(0),
                        num_classes=model.num_classes, inv_triple_map=inv_triple_map,
                        mode='val',
                    )
                    losses['main'].backward()
                    g = torch.cat([p.grad.flatten() for p in model.parameters()
                                if p.grad is not None]).detach().clone()
                    grads.append(g)
                model.zero_grad()
                return torch.stack(grads)     # (B, D)
            finally:
                # 恢复调用前的模式: 上游 _make_gradient_directions 之前已 model.eval(),
                # 因此这里恢复为 eval, 保证随后所有网格评估点都走确定性前向.
                model.train(_was_training)

        def _to_param_dict(flat_vec):
            d = {}
            off = 0
            for name, p in model.named_parameters():
                n = p.numel()
                d[name] = flat_vec[off:off+n].view(p.shape).clone()
                off += n
            return d

        gt = _collect_grads(train_batches, 'train')   # (Bt, D)
        gv = _collect_grads(val_batches,   'val')     # (Bv, D)
        g_mean_train = gt.mean(0)
        g_mean_val   = gv.mean(0)

        # 方向 1: train 平均梯度 (上升方向, 乘 -1 得到下降方向)
        d_mean = -g_mean_train / (g_mean_train.norm() + 1e-12)

        # 方向 2: 冲突方向 = val 梯度 - train 梯度 (归一化)
        g_conf = g_mean_val - g_mean_train
        d_conf = g_conf / (g_conf.norm() + 1e-12)

        # 正交化 d_conf (Gram-Schmidt, 去 d_mean 分量)
        d_conf = d_conf - (d_conf @ d_mean) * d_mean
        d_conf = d_conf / (d_conf.norm() + 1e-12)

        return _to_param_dict(d_mean), _to_param_dict(d_conf)

    # 每组方向固定、跨 checkpoint 可复现；每层 Filter Normalization。
    def _make_direction_pair(direction_seed):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(direction_seed))
        d1, d2 = {}, {}
        for name, _ in param_items:
            w = original_weights[name]
            nw = torch.linalg.vector_norm(w)
            if nw <= 1e-12:
                d1[name] = torch.zeros_like(w)
                d2[name] = torch.zeros_like(w)
                continue
            r1 = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
            r2 = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
            v1 = r1 / torch.linalg.vector_norm(r1).clamp_min(1e-12) * nw
            # 在每个参数张量内 Gram-Schmidt，最后再次恢复 Filter Norm。
            proj = torch.sum(r2 * v1) / torch.sum(v1 * v1).clamp_min(1e-12)
            r2 = r2 - proj * v1
            v2 = r2 / torch.linalg.vector_norm(r2).clamp_min(1e-12) * nw
            d1[name], d2[name] = v1, v2

        dot = torch.zeros((), device=device, dtype=torch.float64)
        n1 = torch.zeros_like(dot)
        n2 = torch.zeros_like(dot)
        for name, _ in param_items:
            a = d1[name].double(); b = d2[name].double()
            dot += (a * b).sum(); n1 += (a * a).sum(); n2 += (b * b).sum()
        cos = dot / torch.sqrt((n1 * n2).clamp_min(1e-30))
        return d1, d2, float(cos.item())

    def _make_direction(d1, d2):
        """
        将梯度方向 (flat 归一化向量的 dict) 适配为景观方向对,
        与 _make_direction_pair 同结构/同尺度, 使 ±alpha 跨方向可比:
          1. Filter Normalization: 每参数张量缩放到原权重模长 nw
          2. 全局 Gram-Schmidt: d2 去 d1 分量 (flat 空间正交经 FN 后被破坏, 需重做)
          3. 重 FN d2 (Gram-Schmidt 改变模长, 缩回原权重尺度)
          4. 返回 (d1f, d2f, cos) — 供 _set_weights / dir-cos 打印 / 循环解包使用
        """
        def _fn(d):
            out = {}
            for name, _ in param_items:
                w = original_weights[name]
                nw = torch.linalg.vector_norm(w)
                if nw <= 1e-12:
                    out[name] = torch.zeros_like(w)
                else:
                    v = d[name]
                    out[name] = v / torch.linalg.vector_norm(v).clamp_min(1e-12) * nw
            return out
        d1f = _fn(d1)
        d2f = _fn(d2)
        # 全局 Gram-Schmidt (double 精度, 跨参数张量)
        dot = torch.zeros((), device=device, dtype=torch.float64)
        n1 = torch.zeros_like(dot)
        for name, _ in param_items:
            a = d1f[name].double(); b = d2f[name].double()
            dot += (a * b).sum(); n1 += (a * a).sum()
        proj = dot / n1.clamp_min(1e-30)
        for name, _ in param_items:
            d2f[name] = d2f[name] - proj.to(d2f[name].dtype) * d1f[name]
        d2f = _fn(d2f)                       # 重 FN (Gram-Schmidt 后模长改变)
        # cos (与 _make_direction_pair 同算法)
        dot = torch.zeros((), device=device, dtype=torch.float64)
        n1 = torch.zeros_like(dot); n2 = torch.zeros_like(dot)
        for name, _ in param_items:
            a = d1f[name].double(); b = d2f[name].double()
            dot += (a * b).sum(); n1 += (a * a).sum(); n2 += (b * b).sum()
        cos = dot / torch.sqrt((n1 * n2).clamp_min(1e-30))
        return d1f, d2f, float(cos.item())

    if has_val:
        d1, d2 = _make_gradient_directions(
            model, train_batches, val_batches, device,
            inv_triple_map,
        )
        direction_pairs = [_make_direction(d1, d2)]
    else:
        print("[损失景观] 无 val, 退回随机方向")
        direction_pairs = [_make_direction_pair(seed + 10007 * i) for i in range(num_directions)]

    alphas = torch.linspace(-range_alpha, range_alpha, steps, dtype=torch.float64)
    betas = torch.linspace(-range_beta, range_beta, steps, dtype=torch.float64)
    cidx = steps // 2
    # 预转 Python 标量列表, 避免内层循环重复 .item() (GPU→CPU 同步开销)
    alpha_list = [float(a) for a in alphas.tolist()]
    beta_list = [float(b) for b in betas.tolist()]

    sum_train = torch.zeros((steps, steps), dtype=torch.float64)
    sum_train_sq = torch.zeros_like(sum_train)
    if has_val:
        sum_val = torch.zeros_like(sum_train)
        sum_val_sq = torch.zeros_like(sum_train)
    else:
        sum_val = sum_val_sq = None

    def _set_weights(a, b, d1, d2):
        # 原地计算 original + a*d1 + b*d2, 避免每网格点分配新张量:
        #   param.data 先复制 original, 再 addcmul_ 两次 (bias 与 1D 参数张量统一处理).
        for name, param in param_items:
            ow = original_weights[name]
            param.data.copy_(ow)
            param.data.add_(d1[name], alpha=a)
            param.data.add_(d2[name], alpha=b)

    def _restore():
        for name, param in param_items:
            param.data.copy_(original_weights[name])

    print(
        f"[损失景观] grid={steps}x{steps} | range=±({range_alpha:.3f},{range_beta:.3f}) | "
        f"directions={num_directions} | seed={seed} | "
        f"dir-cos=" + ", ".join(f"{c:.1e}" for _, _, c in direction_pairs)
    )

    total_pts = len(direction_pairs) * steps * steps
    # 评估阶段强制确定性前向: 关闭 dropout / DropEdge, 使同一 (α,β) 的 loss 是权重的纯函数.
    # (梯度方向提取阶段 _collect_grads 内部会临时 model.train() 并已用 try/finally 恢复)
    model.eval()
    pbar = tqdm(
        total=total_pts, desc="[损失景观] 评估", leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for d_idx, (d1, d2, _) in enumerate(direction_pairs):
        try:
            with torch.no_grad():
                for i, a in enumerate(alpha_list):
                    for j, b in enumerate(beta_list):
                        _set_weights(a, b, d1, d2)
                        lt = _eval_loss_on_batches(train_batches, "train")
                        sum_train[i, j] += lt
                        sum_train_sq[i, j] += lt * lt
                        if has_val:
                            lv = _eval_loss_on_batches(val_batches, "val")
                            sum_val[i, j] += lv
                            sum_val_sq[i, j] += lv * lv
                        pbar.update(1)
                        pbar.set_postfix(dir=d_idx + 1, row=i + 1, col=j + 1, refresh=False)
        finally:
            _restore()
    pbar.close()
    _restore()

    nd = float(len(direction_pairs))
    mean_train = sum_train / nd
    std_train = torch.sqrt((sum_train_sq / nd - mean_train * mean_train).clamp_min(0.0))
    if has_val:
        mean_val = sum_val / nd
        std_val = torch.sqrt((sum_val_sq / nd - mean_val * mean_val).clamp_min(0.0))
    else:
        mean_val = std_val = None

    center_train = float(mean_train[cidx, cidx])

    if has_val:
        center_val = float(mean_val[cidx, cidx])
        gap = mean_val - mean_train
        # 差值的不确定度：这里只作为方向稳定性的可视化 proxy。
        gap_std = torch.sqrt((std_val * std_val + std_train * std_train).clamp_min(0.0))
    else:
        center_val = None
        gap = gap_std = None

    def _curv_x(z):
        h = float(alphas[cidx + 1] - alphas[cidx])
        return float(((z[cidx+1,cidx] + z[cidx-1,cidx] - 2*z[cidx,cidx]) / (h*h)).item())
    def _curv_y(z):
        h = float(betas[cidx + 1] - betas[cidx])
        return float(((z[cidx,cidx+1] + z[cidx,cidx-1] - 2*z[cidx,cidx]) / (h*h)).item())

    curv_train = (_curv_x(mean_train), _curv_y(mean_train))
    direction_cv_train = float((std_train / (mean_train.abs() + 1e-8)).mean())
    if has_val:
        curv_val = (_curv_x(mean_val), _curv_y(mean_val))
        direction_cv_val = float((std_val / (mean_val.abs() + 1e-8)).mean())
    else:
        curv_val = (None, None); direction_cv_val = None

    print(
        f"[损失景观] Train center={center_train:.6f} | curvature={curv_train} | "
        f"direction-CV={direction_cv_train:.4e}"
    )
    if has_val:
        print(
            f"[损失景观] Val center={center_val:.6f} | curvature={curv_val} | "
            f"direction-CV={direction_cv_val:.4e}"
        )
        print(f"[损失景观] center gap={center_val - center_train:.6f} | gap mean={gap.mean().item():.6f}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    x_np, y_np = alphas.numpy(), betas.numpy()

    def _surface(fig, z, z_std, title, row, col, ztitle, colorscale, show_scale=False, extra=None):
        # extra: list of (label, tensor) —— 悬停时额外展示的网格数据。
        custom_cols = []
        hover_lines = [f"α=%{{x:.4f}}<br>β=%{{y:.4f}}<br>{ztitle}=%{{z:.6f}}"]
        if z_std is not None:
            custom_cols.append(z_std.numpy()[:, :, None])
            hover_lines.append(f"direction std=%{{customdata[{len(custom_cols)-1}]:.6f}}")
        if extra:
            for label, tensor in extra:
                custom_cols.append(tensor.numpy()[:, :, None])
                hover_lines.append(f"{label}=%{{customdata[{len(custom_cols)-1}]:.6f}}")
        custom = np.concatenate(custom_cols, axis=2) if custom_cols else None
        hover = "<br>".join(hover_lines) + f"<extra>{title}</extra>"
        fig.add_trace(go.Surface(
            x=x_np, y=y_np, z=z.numpy(), customdata=custom,
            colorscale=colorscale, showscale=show_scale,
            colorbar=dict(title=ztitle) if show_scale else None,
            opacity=1.0,
            # 等高线: 带状填色 + 加深投影, 让 2D 结构在底面清晰可读
            contours={"z": {"show": True, "usecolormap": True, "width": 1,
                "project": {"z": True}}},
            # 照明: 高环境光(背光面也亮, 颜色可见) + 中漫反射(凹凸对比)
            #        + 低镜面(避免高光白斑遮挡色), 让面更清晰立体
            lighting=dict(ambient=0.88, diffuse=0.6, specular=0.1,
                          roughness=0.6, fresnel=0.1),
            lightposition=dict(x=60, y=120, z=80),
            name=title, hovertemplate=hover,
        ), row=row, col=col)
        cz=float(z[cidx,cidx]); fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[cz], mode="markers+text",
            marker=dict(size=7,color="red",symbol="diamond"),
            text=[f"center={cz:.4f}"], textposition="top center", showlegend=False,
        ), row=row, col=col)

    if has_val:
        # 仅渲染前 3 个非 Δ 子图（Train/Val Raw Loss、Train-Val Gap），
        # 跳过第 4~6 个 Δ 可视化子图（Δ 指标仍写入 CSV）。
        fig=make_subplots(rows=1,cols=3,specs=[[{"type":"surface"}]*3],
                          subplot_titles=["Train Raw Loss (Mean)","Val Raw Loss (Mean)","Train-Val Gap (Mean)"],
                          horizontal_spacing=0.03, vertical_spacing=0.08)
        _surface(fig,mean_train,std_train,"Train Raw Loss",1,1,"Loss","Viridis",True)
        _surface(fig,mean_val,std_val,"Val Raw Loss",1,2,"Loss","Viridis",False)
        # Gap = Val Loss - Train Loss：悬停时同时显示被减数(Val Loss)与减数(Train Loss)。
        _surface(fig,gap,gap_std,"Train-Val Gap",1,3,"Val-Train","RdBu",False,
                 extra=[("Val Loss", mean_val), ("Train Loss", mean_train)])
    else:
        fig=make_subplots(rows=1,cols=1,specs=[[{"type":"surface"}]],
                          subplot_titles=["Train Raw Loss (Mean)"])
        _surface(fig,mean_train,std_train,"Train Raw Loss",1,1,"Loss","Viridis",True)

    camera=dict(eye=dict(x=1.55,y=1.55,z=1.15))
    # 仅调整可视化比例（不改变任何原始数据）：加宽 x/y 轴、压低 z 轴高度。
    aspect_ratio = dict(x=1.5, y=1.5, z=0.55)
    for scene_name in ["scene","scene2","scene3"]:
        if scene_name in fig.layout:
            fig.layout[scene_name].update(
                xaxis_title="α (Dir 1)", yaxis_title="β (Dir 2)", camera=camera,
                aspectmode="manual", aspectratio=aspect_ratio,
            )
    fig.update_layout(
        title=(f"Loss Landscape | {('Total' if landscape_loss=='total' else 'Main')} Loss | "
               f"Full Dataset | {num_directions}-Direction Mean | steps={steps}, "
               f"range=±({range_alpha},{range_beta}), seed={seed}<br>"
               f"center Train={center_train:.6f}" +
               (f" | center Val={center_val:.6f} | gap={center_val - center_train:.6f}" if has_val else "") +
               "<br><span style='font-size:12px'>Surface=方向平均；hover=方向 std；(0,0)=训练模型参数中心。</span>"),
        width=1850 if has_val else 650, height=650,
        margin=dict(l=30,r=30,t=125,b=30),
    )
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="directory")
    print(f"[损失景观] HTML 已保存: {output_html}")
    return fig

def build_inv_map_array(inv_triple_map, num_triples=None):
    """
    triple_id -> (side_color0, side_color1, line_color)
    无序目标 (方案B): 前两列为候选色对 (min_color, max_color) —— 不指定 a/b 方位, 由投票消歧。
    转换为 shape=(num_triples, 3) 的 np.ndarray。
    """
    if inv_triple_map is None:
        raise ValueError("inv_triple_map 不能为 None")

    if num_triples is None:
        if not inv_triple_map:
            raise ValueError("inv_triple_map 为空")
        num_triples = max(inv_triple_map.keys()) + 1

    inv_map = np.full((num_triples, 3), -1, dtype=np.int64)

    for triple_id, triple in inv_triple_map.items():
        triple_id = int(triple_id)
        if triple_id < 0 or triple_id >= num_triples:
            raise ValueError(f"非法 triple_id={triple_id}, num_triples={num_triples}")
        inv_map[triple_id] = np.asarray(triple, dtype=np.int64)

    if np.any(inv_map < 0):
        missing = np.where(np.any(inv_map < 0, axis=1))[0]
        raise ValueError(f"inv_triple_map 存在缺失 triple ID，例如: {missing[:10]}")

    return inv_map

def decode_triple_predictions(logits, inv_map):
    """
    logits: [N_l, num_triples]
    return: pred_ids, (string of pred_a, pred_b) 两项为无序候选色 (min_color, max_color), pred_line
    无序目标 (方案B): 前两列为候选色对 (min,max) —— 不指定哪个属于 region_a/b, 由投票消歧。
    """
    pred_ids = logits.argmax(dim=1)
    inv_map_t = torch.as_tensor(inv_map, dtype=torch.long, device=logits.device)
    decoded = inv_map_t[pred_ids]
    return pred_ids, decoded[:, 0], decoded[:, 1], decoded[:, 2]

def predict_line_colors_after_vote(logits, inv_map, region_a, region_b,
                                   region_preds, voted_mask):
    """region 投票完成后, 依据两侧 region 的最终颜色重新推断每条线的线色 c。

    背景: 线色 c 原本直接取自 decode_triple_predictions 的三元组联合 argmax 第 3 维,
    但联合 argmax 会被两侧色对 (min,max) 的选择错误拖累 —— 一旦 (min,max) 选错,
    线色 c 也被连带选错。region 投票通过"多线一致性 + 方向消歧"能更可靠地确定每个
    region 的颜色, 因此线色应在 region 投票完成后, 由两侧 region 的最终颜色锁定色对,
    再在匹配色对的维度内选线色。

    规则 (每条线 i):
      1. 取两侧 region 最终颜色 a=region_preds[ra[i]], b=region_preds[rb[i]];
      2. 无序化色对 s0=min(a,b), s1=max(a,b);
      3. 在 logits 中保留 inv_map 前两列 == (s0,s1) 的维度, 取其中 logits 最高者,
         其第 3 列 (line_color) 即为该线线色;
      4. 若两侧 region 任一未投票, 或该色对无匹配维度 (region 颜色来自不同线的投票,
         组合未必出现在 triple 表中), 则回退到 argmax triple 的 c。

    Args:
        logits:        [N_l, num_triples] 线级 triple logits。
        inv_map:       [num_triples, 3] 每维对应 (min_color, max_color, line_color)。
        region_a/b:    [N_l] 每条线两侧 region 索引。
        region_preds:  [num_regions] long, region 投票后的最终颜色。
        voted_mask:    [num_regions] bool, 是否参与投票。

    Returns:
        line_color: [N_l] long, 每条线的线色预测。
    """
    device = logits.device
    inv_map_t = torch.as_tensor(inv_map, dtype=torch.long, device=device)
    ra = region_a.to(device).long()
    rb = region_b.to(device).long()

    a_col = region_preds[ra]                              # (N,) region_a 最终颜色
    b_col = region_preds[rb]                              # (N,) region_b 最终颜色
    s0 = torch.minimum(a_col, b_col)                      # (N,) 无序色对
    s1 = torch.maximum(a_col, b_col)

    c0all = inv_map_t[:, 0]                               # (NT,) 每维 min 色
    c1all = inv_map_t[:, 1]                               # (NT,) 每维 max 色
    dim_ok = (c0all == s0.unsqueeze(1)) & (c1all == s1.unsqueeze(1))  # (N, NT)

    neg_inf = torch.finfo(logits.dtype).min
    constrained = logits.masked_fill(~dim_ok, neg_inf)
    has_match = dim_ok.any(dim=1)                         # (N,) 色对维度是否存在

    both_voted = voted_mask[ra] & voted_mask[rb]          # 两侧 region 均已投票
    use_post_vote = both_voted & has_match

    k_post = constrained.argmax(dim=1)
    k_fallback = logits.argmax(dim=1)
    k = torch.where(use_post_vote, k_post, k_fallback)

    return inv_map_t[k, 2]

def build_line_c_allowed(color_index, is_color_line, line_color_to_fill, num_classes, device):
    """构建每条线允许的 triple 第三维线色 mask (N, num_classes) bool。

    主线(is_color_line=False): 仅允许 c == color_index (== main_line_color, 已知确定);
    色线(is_color_line=True):  允许 line_color_to_fill[color_index] 中训练集出现过的全部填色
                              (comp_target 是预测目标, 推理时不可精确锁定, 只能约束到"合法集合")。
    line_color_to_fill=None 时返回 None (不启用线色约束)。
    """
    if line_color_to_fill is None:
        return None
    color_index = color_index.to(device).long()
    is_color_line = is_color_line.to(device).bool()
    N = color_index.size(0)
    allowed = torch.zeros(N, num_classes, dtype=torch.bool, device=device)
    # 主线: c == color_index (== main_line_color)
    main_sel = ~is_color_line
    if main_sel.any():
        idx = main_sel.nonzero(as_tuple=True)[0]
        ci = color_index[idx].clamp(0, num_classes - 1)
        allowed[idx, ci] = True
    # 色线: c ∈ line_color_to_fill[color_index]
    for ci, fills in line_color_to_fill.items():
        sel = is_color_line & (color_index == ci)
        if sel.any():
            idx = sel.nonzero(as_tuple=True)[0]
            for c in fills:
                if 0 <= c < num_classes:
                    allowed[idx, c] = True
    return allowed



def _joint_mrf_map_refine(unary, pot_mat, er, es, bp_labels, max_passes=12):
    """将 BP 的逐节点 max-belief 决策投影到能量不下降的联合配置。

    不把 Triple 色对表当硬合法集合：未见色对可凭 Unary 赢过有限的 Pairwise 代价。
    1/2 节点连通分量精确联合 MAP；一般含环分量以 BP / Unary 两个初始化做
    确定性单点坐标上升，按 *原始联合能量* 选最高配置。
    不声称一般含环图的近似解是全局 MAP。
    返回 (labels, E_before, E_after, changed_count)。
    """
    u = unary.detach().cpu().numpy().astype(np.float64, copy=False)
    pot = pot_mat.detach().cpu().numpy().astype(np.float64, copy=False)
    src = er.detach().cpu().numpy().astype(np.int64, copy=False)
    dst = es.detach().cpu().numpy().astype(np.int64, copy=False)
    y_bp = bp_labels.detach().cpu().numpy().astype(np.int64, copy=True)
    R, C = u.shape
    adj = [[] for _ in range(R)]
    for e, (a, b) in enumerate(zip(src, dst)):
        adj[int(a)].append((e, int(b), True))
        adj[int(b)].append((e, int(a), False))

    def score_comp(labels, nodes, edges):
        val = sum(float(u[r, labels[r]]) for r in nodes)
        val += sum(float(pot[e, labels[src[e]], labels[dst[e]]]) for e in edges)
        return val

    result = y_bp.copy()
    unary_labels = np.argmax(u, axis=1).astype(np.int64)
    seen = np.zeros(R, dtype=bool)
    for root in range(R):
        if seen[root]:
            continue
        stack = [root]
        seen[root] = True
        nodes, edge_set = [], set()
        while stack:
            cur = stack.pop()
            nodes.append(cur)
            for e, other, _ in adj[cur]:
                edge_set.add(e)
                if not seen[other]:
                    seen[other] = True
                    stack.append(other)
        nodes.sort()
        edges = sorted(edge_set)
        if len(nodes) == 1:
            result[nodes[0]] = unary_labels[nodes[0]]
            continue
        if len(nodes) == 2 and len(edges) == 1:
            e = edges[0]
            a, b = int(src[e]), int(dst[e])
            joint = u[a, :, None] + u[b, None, :] + pot[e]
            ca, cb = np.unravel_index(int(np.argmax(joint)), joint.shape)
            result[a], result[b] = ca, cb
            continue

        # BP 结果必须始终是候选，任何局部修正都不能让能量下降。
        best = y_bp.copy()
        best_energy = score_comp(best, nodes, edges)
        for seed in (y_bp, unary_labels):
            candidate = seed.copy()
            for _ in range(max_passes):
                updated = False
                for r in nodes:
                    local = u[r].copy()
                    for e, other, is_src in adj[r]:
                        local += (pot[e, :, candidate[other]] if is_src
                                  else pot[e, candidate[other], :])
                    target = int(np.argmax(local))
                    current = int(candidate[r])
                    if target != current and local[target] > local[current] + 1e-10:
                        candidate[r] = target
                        updated = True
                if not updated:
                    break
            energy = score_comp(candidate, nodes, edges)
            if energy > best_energy + 1e-9:
                best, best_energy = candidate, energy
        result[nodes] = best[nodes]

    def total_energy(labels):
        score = float(u[np.arange(R), labels].sum())
        if src.size:
            score += float(pot[np.arange(src.size), labels[src], labels[dst]].sum())
        return score

    before, after = total_energy(y_bp), total_energy(result)
    if after + 1e-7 < before:
        raise AssertionError('联合 MAP 修正反而降低 MRF 能量')
    return (torch.as_tensor(result, device=unary.device, dtype=torch.long),
            before, after, int(np.count_nonzero(result != y_bp)))


def _conf_from_probs(p, eps=1e-8, floor=0.05):
    """概率分布 → 置信度 ∈ [0,1]: `1 - H(p)/log(C)`。

    用熵而不是 max prob: softmax 峰值普遍过度自信且未校准, 熵对"分布在几个类上摊平"
    更敏感, 也更适合当融合权重。
    floor 是权重下限, 防止完全无信息的一方权重为 0 导致除零 / 彻底失声。
    """
    C = p.size(-1)
    if C <= 1:
        return torch.ones(p.size(0), device=p.device, dtype=p.dtype)
    logp = torch.log(p.clamp(min=eps))
    H = -(p.clamp(min=eps) * logp).sum(dim=-1)          # (N,)
    conf = 1.0 - H / math.log(C)
    return conf.clamp(min=floor, max=1.0)


def solve_region_colors_mrf(logits, inv_map, region_a, region_b, line_pixel_count,
                            num_regions, num_classes, rounds=20, damping=0.5,
                            line_plus_region_a=None, side_logits=None, diagnostic=None):
    """显式 MRF 能量近似最小化 (max-product BP + 联合 MAP 能量修正)。

    能量函数:
        E(y) = -Σ_r unary_r(y_r) - Σ_e ψ_e(y_r, y_s)
    - unary_r(c) = 逐侧 Triple×Side 联合边际的置信加权 log 平均。
    - soft pairwise: 对已见色对用 λ·clip(log(Q/maxQ),-M,0)，
      对训练表未见色对仅使用有限的 λ·PAIRWISE_UNSEEN_PENALTY 退让代价。
      不以观察到的 Triple 集合作为可行域，允许产生新的 Region 色对。
    - BP 产生 max-belief；最终再用联合能量单调修正 MAP 配置。

    为什么用 max-product BP 而非 α-expansion:
      - 本问题的 pairwise (线色对似然) 在软概率下非严格 submodular; 若强行改成
        "相邻不同色"(anti-Potts) 也非 submodular → α-expansion 的 2-逼近界不成立,
        需要辅助节点, 收益存疑;
      - max-product BP 不要求 submodular, 是求 MAP 的通用近似 (树图精确, loopy 图逼近);
      - max 而非 sum: 不会像 sum-product 在环形图上把信号平均稀释、塌缩到均匀。

        防塌缩 (关键):
            - 违约是有限负分而非"不奖励": 全同色对 argmax 异色线会主动积累 mismatch 惩罚;
            - MRF_MISMATCH_WEIGHT 控制约束强度, 惩罚上限为 λ·w_i·conf_i, 不会像 -log(ε) 一样失控;
      - damping 0.5 抑制环上消息自我强化。

    Args:
        logits: (N, NT) 线级 triple logits。inv_map: (NT, 3) 每维 (min色, max色, 线色)。
        region_a/b: (N,) 每条线两侧 region 索引。line_pixel_count: (N,) 证据权重。
        num_regions / num_classes: 规模。rounds / damping: BP 迭代轮数与阻尼。
    Returns:
        (region_probs, vote_weight, region_preds, region_confidence, voted_mask, evidence_norm)
    """
    device = logits.device
    dtype = logits.dtype
    N, NT = logits.shape
    C = num_classes
    eps = 1e-8

    inv_map_t = torch.as_tensor(inv_map, dtype=torch.long, device=device)
    c0 = inv_map_t[:, 0].long()          # (NT,) min 色
    c1 = inv_map_t[:, 1].long()          # (NT,) max 色 (c0<=c1 已保证)

    # ---- 1. 色对空间去重 (c0,c1) → pair id ----
    pair_key = c0 * C + c1
    uniq_pk, pair_id_of_t = torch.unique(pair_key, return_inverse=True)
    num_pairs = uniq_pk.numel()
    pc0 = uniq_pk // C
    pc1 = uniq_pk % C

    # ---- 2. 色对边缘概率 P_i[{A,B}] (N, NP) ----
    probs = F.softmax(logits, dim=1)
    pair_probs = torch.zeros(N, num_pairs, device=device, dtype=dtype)
    pair_probs.scatter_add_(1, pair_id_of_t.unsqueeze(0).expand(N, NT), probs)

    w = line_pixel_count.to(device).clamp(min=1.0)
    # 注意: w 只用于返回的 vote_weight (投票长度统计), **不进入** MRF 能量 —— unary 用
    #   conf(联合后验) 加权, pairwise 只用 λ·log(Q/maxQ)。
    ra = region_a.to(device).long()
    rb = region_b.to(device).long()

    # ---- 3. unary: 方向感知的无序色对边际聚合到 region ----
    #   (低 |logit| 回退对称交给 MRF 多线消歧, 高 |logit| 才用模型方向)。
    # in0/in1 = c0/c1 的 one-hot (num_pairs, C)。同色 pair 时 in0==in1, d 自然抵消 → 全归 c。
    in0 = torch.zeros(num_pairs, C, device=device, dtype=dtype)
    in1 = torch.zeros(num_pairs, C, device=device, dtype=dtype)
    pair_col = torch.arange(num_pairs, device=device)
    in0[pair_col, pc0] = 1.0
    in1[pair_col, pc1] = 1.0

    # 方向: 不做拆分, 与训练侧 _joint_region_nll 完全同口径 (乘性协同的联合后验)。
    #   triple 预测的是无序色对 {c0,c1}, c0/c1 只是编号大小、没有语义; 用 side 后验去反推
    #   "哪个是较小色" 会把两个任务耦合 (见 DESIGN_NOTES 11.2)。因此两侧共用同一份无序边际:
    #       prob_a = prob_b = 0.5 · ( P(该色是 c0) + P(该色是 c1) )
    #   侧的区分交给 side 的逐侧似然 (在联合后验里以 q_a/q_b 的乘性因子进入) 与 MRF 多线一致性。
    pair_in0 = pair_probs @ in0                              # (N, C) c0 边际
    pair_in1 = pair_probs @ in1                              # (N, C) c1 边际

    # ---- 乘性协同: triple 给"组合先验", side 给"两侧似然", 相乘得联合后验 ----
    #   对每个候选色对 p = (c0, c1) 有两种朝向:
    #       A_p = P_triple(p) · q_a(c0)^β · q_b(c1)^β      # c0 在 a 侧
    #       B_p = P_triple(p) · q_a(c1)^β · q_b(c0)^β      # 交换
    #   然后分别聚合到两侧的颜色边际 (再各自归一化):
    #       prob_a = (Σ_p A_p·1[c=c0_p] + Σ_p B_p·1[c=c1_p]) / Σ_p (A_p + B_p)
    #       prob_b = (Σ_p A_p·1[c=c1_p] + Σ_p B_p·1[c=c0_p]) / Σ_p (A_p + B_p)
    #
    #   与"加权平均两个分布"的本质区别: 这里每个头只提供自己**能估的那个因子** ——
    #   triple 负责"这两个颜色的组合合不合理", side 负责"每一侧和分配给它的颜色多匹配";
    #   相乘才构成完整证据, 不需要人为定 1:1 之类的配比。
    #
    #   β = SYNERGY_BETA < 1 会软化 side 似然, 降低"side 自信地给 0"掐死 triple 正确证据的风险。
    pair_joint = None   # side 关闭时无联合后验 -> joint 势自动回退到 triple 势
    if side_logits is not None and SIDE_INFER_WEIGHT > 0.0:
        q_a = torch.softmax(side_logits[:, 0], dim=-1).clamp(min=eps)
        q_b = torch.softmax(side_logits[:, 1], dim=-1).clamp(min=eps)
        if UNARY_MODE == "side":
            prob_a, prob_b = q_a, q_b
        else:
            _beta = None
            if UNARY_MODE == "adaptive":
                # 逐线软化: beta_i = conf(q_a)·conf(q_b) ∈ [0,1]; side 越不确定越靠近 triple
                _beta = (_conf_from_probs(q_a) * _conf_from_probs(q_b)).unsqueeze(1)
            prob_a, prob_b, pair_joint = _synergy_posterior(
                pair_probs, q_a, q_b, pc0, pc1, in0, in1, eps, beta=_beta,
                mode=("factorized" if UNARY_MODE == "factorized"
                          else "agree" if UNARY_MODE == "agree" else "poe"))
    else:
        # 没有 side 时退化为 triple 的无序边际 (两侧相同)
        prob_a = 0.5 * (pair_in0 + pair_in1)
        prob_b = prob_a
    if UNARY_MODE == "triple":
        # 仅 triple 边际 (两侧相同), 用于消融 side 的贡献
        prob_a = 0.5 * (pair_in0 + pair_in1)
        prob_b = prob_a

    log_prob_a = torch.log(prob_a.clamp(min=eps))
    log_prob_b = torch.log(prob_b.clamp(min=eps))
    # 逐线逐侧权重 = 该侧 j o i n t 后验自身的置信度 (已同时反映两路的确定性)
    w_a = _conf_from_probs(prob_a).clamp(min=eps)
    w_b = _conf_from_probs(prob_b).clamp(min=eps)

    unary = torch.zeros(num_regions, C, device=device, dtype=dtype)
    region_mass = torch.zeros(num_regions, device=device, dtype=dtype)
    # 路由: prob_a 是 +n 侧颜色分布, 应聚合到真正在 +n 侧的 region (几何法向侧 canonicalization)。
    if line_plus_region_a is not None:
        plus_is_a = line_plus_region_a.to(torch.bool)
        plus_region = torch.where(plus_is_a, ra, rb)
        minus_region = torch.where(plus_is_a, rb, ra)
    else:
        plus_region = ra
        minus_region = rb
    unary.index_add_(0, plus_region, log_prob_a * w_a.unsqueeze(1))
    unary.index_add_(0, minus_region, log_prob_b * w_b.unsqueeze(1))
    region_mass.index_add_(0, plus_region, w_a)
    region_mass.index_add_(0, minus_region, w_b)
    # 真正的加权平均: 除以实际证据质量 mass (而不是 clamp 到 1)。
    #   clamp(min=1.0) 会让"总置信度不足 1"的 region 只保留 mass 倍的对数证据
    #   (例如单条 conf=0.05 的线只保留 5% 强度), 而 pairwise 没有相应缩弱,
    #   使低支持度 region 更容易被 pairwise 主导。mass=0 的 region 由 valid 掩码排除。
    if UNARY_MASS_MODE == "true":
        unary = unary / region_mass.clamp(min=eps).unsqueeze(1)
    else:
        unary = unary / region_mass.clamp(min=1.0).unsqueeze(1)

    # 同侧线 (ra == rb) 不构成 pairwise 边: 否则会生成自环 (er==es), 把该线证据在
    # unary 之外再算一遍并自我强化。这类线只通过上面的 unary 投票 (允许投票),
    # 不参与 region 间的 pairwise 约束。
    _non_self = (ra != rb)                                  # (N,) bool
    e_key = torch.where(ra < rb, ra * num_regions + rb, rb * num_regions + ra)
    if _non_self.any():
        uniq_ek, edge_id_of_line = torch.unique(e_key[_non_self], return_inverse=True)
    else:
        uniq_ek = torch.zeros(0, dtype=torch.long, device=device)
        edge_id_of_line = torch.zeros(0, dtype=torch.long, device=device)
    num_edges = uniq_ek.numel()
    er = uniq_ek // num_regions
    es = uniq_ek % num_regions

    # ---- 4. region 无向邻接边 pairwise potential (NE, C, C) ----
    # 未见色对 (pair_id_mat == -1) 使用有限的惩罚，而非 Q=0 导致接近 -MRF_SOFT_M。
    # E1 (MRF_SOFT_PAIR): ψ_i(A,B) = λ·clip(log(Q_i(A,B)/maxQ_i), -m, 0), 软 pair 势.
    #   保留 Top-K 候选的生存空间 (log pair_probs 比例), 而非只对 argmax pair 不惩罚。
    # 旧 (MRF_SOFT_PAIR=False): 仅 argmax pair 不惩罚, 其余 pair 扣有限常数分。
    pair_id_mat = torch.full((C, C), -1, dtype=torch.long, device=device)
    pair_id_mat[pc0, pc1] = torch.arange(num_pairs, device=device)
    pair_id_mat[pc1, pc0] = torch.arange(num_pairs, device=device)

    if not PAIRWISE_ENABLE:
        # 只保留 unary: 边势恒 0 -> BP 消息恒为常数 -> 每个 region 独立取 unary 最大。
        line_pot = torch.zeros(N, C * C, device=device, dtype=dtype)
    elif MRF_SOFT_PAIR and PAIRWISE_POT in ("joint", "hybrid") and pair_joint is not None:
        # 联合势: 用同一份 Triple×Side 有序联合后验 (保留两侧朝向), 而非仅 triple 边际。
        #   "joint": 所有边都用 (已知会过约束, 见 §18.4);
        #   "hybrid": **只在歧义边**用定向势 —— 即两端 region 的 unary 首选颜色相同的边。
        #     这类边是无序势的盲区: 它只要求(3,10)合法, 无法决定哪侧拿 3, BP 只能按 mass 破平局,
        #     实测会把 unary 置信度 0.99 的大区域翻转 (§18 之后的分析)。
        _is_plus_es = (es[edge_id_of_line] == plus_region[_non_self]).view(-1, 1, 1)
        _pj = torch.where(_is_plus_es,
                          pair_joint[_non_self],
                          pair_joint[_non_self].transpose(1, 2))
        _mx = _pj.flatten(1).max(dim=1).values.view(-1, 1, 1)
        _lr = torch.log((_pj + eps) / (_mx + eps)).clamp(min=-MRF_SOFT_M)
        joint_pot = torch.zeros(N, C * C, device=device, dtype=dtype)
        _seen_pair = (pair_id_mat >= 0).view(1, C, C)
        _lr = torch.where(_seen_pair, _lr, _lr.new_full((), -PAIRWISE_UNSEEN_PENALTY))
        joint_pot[_non_self] = MRF_MISMATCH_WEIGHT * _lr.flatten(1)
        if PAIRWISE_POT == "joint":
            line_pot = joint_pot
        else:
            pid_flat = pair_id_mat.flatten()
            _valid = pid_flat >= 0
            Q = torch.zeros(N, C * C, device=device, dtype=dtype)
            Q[:, _valid] = pair_probs[:, pid_flat[_valid]]
            maxQ = pair_probs.max(dim=1).values.unsqueeze(1)
            _tri_log = torch.log((Q + eps) / (maxQ + eps)).clamp(min=-MRF_SOFT_M)
            _tri_log[:, ~_valid] = -PAIRWISE_UNSEEN_PENALTY
            tri_pot = MRF_MISMATCH_WEIGHT * _tri_log
            _arg_r = unary.argmax(dim=1)
            _amb_edge = (_arg_r[es] == _arg_r[er])                     # (E,) 歧义边
            _m = _amb_edge[edge_id_of_line].unsqueeze(1).to(dtype)     # (n_ns,1)
            line_pot = tri_pot.clone()
            line_pot[_non_self] = (_m * joint_pot[_non_self]
                                   + (1.0 - _m) * tri_pot[_non_self])
    elif MRF_SOFT_PAIR:
        # 配对势 = triple 的无序色对分布。
        #   已测试并否决: 把配对势换成 Triple×Side 的有序联合后验 (即与 region 单点
        #   损失用同一份概率模型)。该联合后验远比 triple 边际尖锐, 又按 incident line
        #   求和, 使得线多的边 (input_B 4.4 线/region) 的 pairwise 项压过 unary:
        #     B 0.8214 -> 0.5238 (n=3), V 0.5594 -> 0.5258 (n=3)。
        #   再改成"按边求平均"(消除线数放大) 后 B 回到 0.8190 (seed42) 但 seed43 为
        #   0.7333, n=2 未定论。结论记录在 DESIGN_NOTES.md §12, 配对势仍用 triple。
        pid_flat = pair_id_mat.flatten()
        _valid = pid_flat >= 0
        Q = torch.zeros(N, C * C, device=device, dtype=dtype)
        Q[:, _valid] = pair_probs[:, pid_flat[_valid]]
        maxQ = pair_probs.max(dim=1).values.unsqueeze(1)
        log_ratio = torch.log((Q + eps) / (maxQ + eps)).clamp(min=-MRF_SOFT_M)
        log_ratio[:, ~_valid] = -PAIRWISE_UNSEEN_PENALTY
        line_pot = MRF_MISMATCH_WEIGHT * log_ratio
    else:
        best_pid = pair_probs.argmax(dim=1)
        pair_conf = pair_probs.gather(1, best_pid.unsqueeze(1)).squeeze(1)
        mismatch = (MRF_MISMATCH_WEIGHT * pair_conf).unsqueeze(1)
        hit = (pair_id_mat.unsqueeze(0) == best_pid.view(-1, 1, 1))  # (N,C,C)
        line_pot = (-mismatch.unsqueeze(2) * (~hit).to(dtype)).reshape(N, C * C)
        # 硬 argmax 模式也不能将未知色对视为绝对禁用，最多施加有限退让代价。
        _seen_flat = (pair_id_mat.flatten() >= 0)
        line_pot[:, ~_seen_flat] = -MRF_MISMATCH_WEIGHT * PAIRWISE_UNSEEN_PENALTY

    pot_flat = torch.zeros(num_edges, C * C, device=device, dtype=dtype)
    if _non_self.any():
        pot_flat.scatter_add_(0,
                              edge_id_of_line.unsqueeze(1).expand(-1, C * C),
                              line_pot[_non_self])
        # 边内证据归一化: 同一条 region 边上的多条线, 其势按 1/sqrt(n_rs) 缩放后再求和。
        #   理由: 一条物理笔画被骨架关键点切成 k 段, 并不等于得到 k 份独立证据; 直接求和会让
        #   线数多的边 (J 这类细碎图) 的 pairwise 强度被放大, 从而需要为每个数据集单独调 λ。
        #   取 sqrt 而非 n: 既抑制线性放大, 又保留"多线支持更强"的单调性; n_rs 由数据决定, 非超参。
        #   两种数据驱动的聚合: "inv_sqrt" (1/sqrt(n)) / "inv_n" (1/n, 即边内平均)。
        if PAIRWISE_AGG in ("inv_sqrt", "inv_n"):
            _n_rs = torch.zeros(num_edges, device=device, dtype=dtype)
            _n_rs.index_add_(0, edge_id_of_line,
                             torch.ones_like(edge_id_of_line, dtype=dtype))
            _denom = (_n_rs.clamp_min(1.0).sqrt() if PAIRWISE_AGG == "inv_sqrt"
                      else _n_rs.clamp_min(1.0))
            pot_flat = pot_flat / _denom.unsqueeze(1)
    pot_mat = pot_flat.view(num_edges, C, C)

    # ---- 4b. unary 边际闸门 ----
    #   当一条边两端的 unary 首选色相同 (argmax 一致) 时, 无序边势只会强迫其中一方改色,
    #   而"改哪一方"只能按 unary 边际决定 —— 这正是 G 的 r3/r4 颜色互换的产生方式。
    #   闸门: 若两端的 unary 边际都超过该边对 (c,c) 的惩罚力度, 说明双方各自证据都比
    #   "必须不同色"这条约束更硬 -> 允许它们同色 (把该 (c,c) 惩罚清零), 不做无依据的强拆。
    if PAIRWISE_GATE != "none" and num_edges > 0:
        _arg = unary.argmax(dim=1)
        _top2 = unary.topk(min(2, C), dim=1).values
        _margin = (_top2[:, 0] - _top2[:, 1]) if C > 1 else torch.zeros_like(_top2[:, 0])
        _same = (_arg[es] == _arg[er])
        if _same.any():
            _idx = _same.nonzero(as_tuple=True)[0]
            _c = _arg[es[_idx]]
            _pen = pot_mat[_idx, _c, _c]                      # (k,) 负值
            if PAIRWISE_GATE == "same_color":
                _ok = torch.ones_like(_pen, dtype=torch.bool)
            else:
                _lim = torch.minimum(_margin[es[_idx]], _margin[er[_idx]])
                _ok = _lim > (-_pen)
            if _ok.any():
                pot_mat[_idx[_ok], _c[_ok], _c[_ok]] = 0.0

    # ---- 5. max-product loopy BP 求 MAP (全边批量向量化, 无 Python 逐边循环) ----
    msg_rs = torch.zeros(num_edges, C, device=device, dtype=dtype)  # r->s
    msg_sr = torch.zeros(num_edges, C, device=device, dtype=dtype)  # s->r
    for _it in range(rounds):
        incoming = torch.zeros(num_regions, C, device=device, dtype=dtype)
        incoming.index_add_(0, es, msg_rs)
        incoming.index_add_(0, er, msg_sr)
        node = unary + incoming                                    # (R, C) 节点势能+入射消息
        cav_r = node[er] - msg_sr                                   # (NE, C)
        new_rs = (pot_mat + cav_r.unsqueeze(2)).max(dim=1).values   # (NE, C) max over y_r
        new_rs = new_rs - new_rs.max(dim=1, keepdim=True).values
        cav_s = node[es] - msg_rs                                   # (NE, C)
        new_sr = (pot_mat + cav_s.unsqueeze(1)).max(dim=2).values   # (NE, C) max over y_s
        new_sr = new_sr - new_sr.max(dim=1, keepdim=True).values
        msg_rs = (1 - damping) * msg_rs + damping * new_rs
        msg_sr = (1 - damping) * msg_sr + damping * new_sr

    log_belief = unary.clone()
    log_belief.index_add_(0, es, msg_rs)
    log_belief.index_add_(0, er, msg_sr)
    y_bp = log_belief.argmax(dim=1)
    # max-product 的节点 max-belief 各自 argmax 并不保证形成有效的联合 MAP；
    # 典型例子是对称的红/蓝边，两端各为 50/50，独立 argmax 会选成红/红。
    # 用联合能量修正，且允许训练集中未出现过的色对。
    y, _energy_bp, _energy_final, _n_map_changed = _joint_mrf_map_refine(
        unary, pot_mat, er, es, y_bp)

    # ---- 6. 输出 ----
    region_probs = F.softmax(log_belief, dim=1)
    region_preds = y

    voted_mask = torch.zeros(num_regions, dtype=torch.bool, device=device)
    voted_mask[ra] = True
    voted_mask[rb] = True

    # ---- 诊断转储 (仅 MRF_DUMP_PATH 设置时): 逐 region 的 unary top-k + 逐线证据 ----
    _dump_path = os.environ.get("MRF_DUMP_PATH", "").strip()
    if _dump_path:
        import json as _json
        with torch.no_grad():
            _uk = unary.topk(min(3, C), dim=1)
            _pk = pair_probs.topk(min(3, pair_probs.size(1)), dim=1)
            _sa = prob_a.topk(min(3, C), dim=1)
            _sb = prob_b.topk(min(3, C), dim=1)
            _rec = {
                "num_regions": int(num_regions),
                "regions": [{
                    "r": int(i),
                    "pred": int(region_preds[i]),
                    "mass": float(region_mass[i]),
                    "unary_top": [[int(c), float(p)] for c, p in
                                  zip(_uk.indices[i].tolist(), _uk.values[i].tolist())],
                } for i in range(num_regions)],
                "lines": [{
                    "l": int(i),
                    "plus_r": int(plus_region[i]), "minus_r": int(minus_region[i]),
                    "w_a": float(w_a[i]), "w_b": float(w_b[i]),
                    "triple_top": [[int(pc0[t]), int(pc1[t]), float(p)] for t, p in
                                   zip(_pk.indices[i].tolist(), _pk.values[i].tolist())],
                    "side_a_top": [[int(c), float(p)] for c, p in
                                   zip(_sa.indices[i].tolist(), _sa.values[i].tolist())],
                    "side_b_top": [[int(c), float(p)] for c, p in
                                   zip(_sb.indices[i].tolist(), _sb.values[i].tolist())],
                } for i in range(N)],
            }
        with open(_dump_path, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_rec) + "\n")

    vote_weight = torch.zeros(num_regions, device=device, dtype=dtype)
    vote_weight.index_add_(0, ra, w)
    # 同侧线 (ra==rb) 已在 ra 侧计过一次, 不在 rb 侧重复累加 (每条线权重只计一次)
    vote_weight.index_add_(0, rb, w * _non_self.to(dtype))

    # ---- 决策支持度 (不是校准正确率) + 周边 Line 分歧 ----
    # 投票权重与 MRF Unary 完全相同：各线侧联合边际的归一化熵权重。
    # 支持度 = Σ_i w_i p_i(y_final) / Σ_i w_i；方向冲突会直接降低支持度。
    # JSD = H(Σ w_i p_i/Σ w_i) - Σ w_i H(p_i)/Σ w_i, 可以区分
    # "各条线都犹豫"(低支持、低分歧) 与 "各条线很自信但互相冲突"(低支持、高分歧)。
    mean_probs = torch.zeros(num_regions, C, device=device, dtype=dtype)
    entropy_mass = torch.zeros(num_regions, device=device, dtype=dtype)
    incident_count = torch.zeros(num_regions, device=device, dtype=dtype)
    _ones = torch.ones_like(w_a)
    _Ha = -(prob_a * torch.log(prob_a.clamp_min(eps))).sum(dim=1)
    _Hb = -(prob_b * torch.log(prob_b.clamp_min(eps))).sum(dim=1)
    mean_probs.index_add_(0, plus_region, prob_a * w_a.unsqueeze(1))
    mean_probs.index_add_(0, minus_region, prob_b * w_b.unsqueeze(1))
    entropy_mass.index_add_(0, plus_region, w_a * _Ha)
    entropy_mass.index_add_(0, minus_region, w_b * _Hb)
    incident_count.index_add_(0, plus_region, _ones)
    incident_count.index_add_(0, minus_region, _ones)
    mean_probs = mean_probs / region_mass.clamp_min(eps).unsqueeze(1)
    entropy_mean = -(mean_probs * torch.log(mean_probs.clamp_min(eps))).sum(dim=1)
    js_div = (entropy_mean - entropy_mass / region_mass.clamp_min(eps)).clamp_min(0)
    # 上界为 log(min(有效投票侧数, 类别数)); 两条完全冲突的确定投票应展示 100% 分歧。
    js_denom = torch.log(torch.minimum(incident_count, incident_count.new_full((), float(C))).clamp_min(2))
    line_disagreement = (js_div / js_denom).clamp(0, 1)
    line_disagreement = torch.where(incident_count > 1, line_disagreement, torch.zeros_like(line_disagreement))
    region_confidence = mean_probs.gather(1, region_preds.unsqueeze(1)).squeeze(1).clamp(0, 1)
    region_confidence = region_confidence.masked_fill(~voted_mask, 0.0)
    line_disagreement = line_disagreement.masked_fill(~voted_mask, 0.0)
    region_probs = region_probs.masked_fill(~voted_mask.unsqueeze(1), 0.0)
    region_preds = region_preds.masked_fill(~voted_mask, 0)

    if diagnostic is not None:
        # 非概率的 BP belief：禁止将 max softmax(belief) 解释为已校准正确率。
        diagnostic.update({
            'unary': unary.detach().clone(),
            'pair_probs': pair_probs.detach().clone(),
            'pair_colors': torch.stack([pc0, pc1], dim=1).detach().clone(),
            'pairwise': pot_mat.detach().clone(),
            'pairwise_edges': torch.stack([er, es], dim=0).detach().clone(),
            'belief': log_belief.detach().clone(),
            'bp_messages_rs': msg_rs.detach().clone(),
            'bp_messages_sr': msg_sr.detach().clone(),
            'region_mass': region_mass.detach().clone(),
            'unary_probs': torch.softmax(unary, dim=1).detach().clone(),
            'decision_support': region_confidence.detach().clone(),
            'line_disagreement': line_disagreement.detach().clone(),
            'incident_count': incident_count.detach().clone(),
            'bp_independent_labels': y_bp.detach().clone(),
            'final_labels': region_preds.detach().clone(),
            'energy_before': _energy_bp,
            'energy_after': _energy_final,
            'joint_map_changes': _n_map_changed,
            'unknown_color_pairs_possible': bool((pair_id_mat < 0).any().item()),
        })
    evidence_norm = log_belief
    # 逐线**实际参与投票**的证据分布: 乘性协同后的方向感知边际 (无 side 时为 triple
    # 无序边际, 两侧相同)。HTML 侧栏 "A侧/B侧 投票" 由此得到 —— 之前外部拿到的是
    # triple 无序边际 (prob_a = prob_b), 与投票用的分布不是同一份。
    return (region_probs, vote_weight, region_preds, region_confidence,
            voted_mask, evidence_norm, prob_a, prob_b)


def predict_regions_from_logits(
    logits,
    inv_map,
    region_a,
    region_b,
    line_pixel_count,
    num_regions,
    num_classes,
    line_c_allowed=None,
    line_plus_region_a=None,
    side_logits=None,
    diagnostic=None,
):
    """
    Triple logits
        -> A/B marginal (方向感知)
        -> Region evidence aggregation

    line_c_allowed: (N, num_classes) bool, 每条线允许的 triple 第三维线色 mask (可选);
        None 表示不启用线色一致性硬屏蔽。(MRF 求解只用色对, 线色约束在线色预测阶段生效)
    """
    # 逐线投票分布由 solve_region_colors_mrf 内部给出: 乘性协同后的**方向感知**边际
    # (无 side 时退化为 triple 无序边际, 两侧相同)。不在这里另算一份 triple 边际 ——
    # 那份 A/B 两侧恒相同, 不是实际参与投票的分布。
    #
    # 软势区域颜色分配：优先匹配 Triple 色对，但未见色对允许由 Unary 证据选择；
    # max-product BP 后用联合能量修正配置，并非硬性限定训练中的 Triple 表。
    (region_probs, vote_weight, region_preds, region_confidence, voted_mask, evidence_norm,
     line_prob_a, line_prob_b) = solve_region_colors_mrf(
        logits, inv_map, region_a, region_b, line_pixel_count,
        num_regions, num_classes,
        line_plus_region_a=line_plus_region_a, side_logits=side_logits,
        diagnostic=diagnostic)

    vote_result = (
        region_probs,
        vote_weight,
        region_preds,
        region_confidence,
        voted_mask,
        evidence_norm,
        None,                        # scores: 旧投票末轮得分, MRF 下不再需要 (线色改由 region 颜色锁定)
    )
    return line_prob_a, line_prob_b, vote_result


def evaluate_model(model, val_loader, device, inv_triple_map, line_color_to_fill=None):
    """
    对验证集计算各项指标（复用训练循环中的验证代码）。
    返回验证集 Triple 主损失、线级无序色对准确率、图节点 RegionAcc、
    仅投票 RegionAcc、投票覆盖率、逐侧 SideAcc。
    不返回 DirAcc：Direction Head 已删除。
    """
    model.eval()
    val_loss_sum = 0.0
    val_correct_pair = val_total = 0
    val_region_correct = val_region_total = 0
    val_region_voted_correct = val_region_voted_total = val_region_unvoted = 0
    # Side-head 逐侧线级准确率 (用于判断 side 是否值得进投票: 与 triple 的 PairAcc 对照)
    side_correct_plus = side_total_plus = 0
    side_correct_minus = side_total_minus = 0
    side_correct_both = side_total_both = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            h_l, h_r = model(
                batch["X_l"],
                batch["edge_index_ll"],
                batch["edge_attr_ll"],
                batch["color_index"],
                batch["edge_index_rr"],
                batch["line_region_a"],
                batch["line_region_b"],
                                region_line_color_geo=batch["region_line_color_geo"],
                edge_relation_mask=batch["edge_relation_mask"],
                stroke_id=batch.get("stroke_id"),
                edge_index_stroke=batch.get("edge_index_stroke"),
                edge_index_intra_stroke=batch.get("edge_index_intra_stroke"),
                cont_stroke_id=batch.get("cont_stroke_id"),
                edge_index_cont_stroke=batch.get("edge_index_cont_stroke"),
                edge_index_intra_cont=batch.get("edge_index_intra_cont"),
                line_pos=batch.get("line_pos"),
                region_pos=batch.get("region_pos"), region_shape_feat=batch.get("region_shape_feat"),
            )

            logits, side_logits, _, region_a, region_b, label_a, label_b = model.classify_lines(
                h_l,
                h_r,
                batch["line_region_a"],
                batch["line_region_b"],
                batch["line_label_a"],
                batch["line_label_b"],
                batch["line_plus_region_a"],
            )

            # 主损失（用于输出，不参与反向传播）
            losses_val = compute_loss(
                logits=logits,
                triple_label=batch["triple_label"],
                region_a=region_a,
                region_b=region_b,
                line_pixel_count=batch["line_pixel_count"],
                num_regions=batch["y"].size(0),
                num_classes=model.num_classes,
                inv_triple_map=inv_triple_map,
                mode='val',
            )
            val_loss_sum += losses_val['main'].item()

            # ---- Side-head 逐侧准确率 (判断 side 是否有资格进投票) ----
            #   口径与 side 的 CE 训练一致: +n 侧对应 side_logits[:,0], 真值按 line_plus_region_a 重排。
            if side_logits is not None and label_a is not None and label_b is not None:
                if batch.get("line_plus_region_a") is not None:
                    _pa = batch["line_plus_region_a"].to(torch.bool)
                    _y_plus = torch.where(_pa, label_a, label_b)
                    _y_minus = torch.where(_pa, label_b, label_a)
                else:
                    _y_plus, _y_minus = label_a, label_b
                _ok_plus = (side_logits[:, 0].argmax(dim=-1) == _y_plus)
                _ok_minus = (side_logits[:, 1].argmax(dim=-1) == _y_minus)
                side_correct_plus += _ok_plus.sum().item()
                side_total_plus += int(_y_plus.numel())
                side_correct_minus += _ok_minus.sum().item()
                side_total_minus += int(_y_minus.numel())
                side_correct_both += (_ok_plus & _ok_minus).sum().item()
                side_total_both += int(_y_plus.numel())

            # ---- 三元组准确率 ----
            if inv_triple_map is not None:
                inv_map = build_inv_map_array(inv_triple_map, logits.size(1))
                _, pred_a, pred_b, _ = decode_triple_predictions(logits, inv_map)
                valid = batch["triple_label"] >= 0
                if valid.any():
                    # 统一"无序色对"口径: 真值对与预测对均规范化为 (min,max), 再逐分量匹配。
                    # (无序 triple 下 pred_a/pred_b 已是 min/max, 此 min/max 化是无操作;
                    #   有序 triple 下 pred_a/pred_b 为 (a,b), 需 min/max 化才能跨方案可比)
                    _la, _lb = label_a[valid], label_b[valid]
                    tmin, tmax = torch.minimum(_la, _lb), torch.maximum(_la, _lb)
                    pa, pb = pred_a[valid], pred_b[valid]
                    pmin, pmax = torch.minimum(pa, pb), torch.maximum(pa, pb)
                    val_correct_pair += ((pmin == tmin) & (pmax == tmax)).sum().item()
                    val_total += valid.sum().item()

            # ---- Region 准确率 ----
            if inv_triple_map is not None:
                inv_map = build_inv_map_array(inv_triple_map, logits.size(1))
                line_c_allowed = build_line_c_allowed(
                    batch["color_index"], batch["is_color_line"],
                    line_color_to_fill, model.num_classes, device)
                _, _, vote_result = predict_regions_from_logits(
                    logits=logits,
                    inv_map=inv_map,
                    region_a=region_a,
                    region_b=region_b,
                    line_pixel_count=batch["line_pixel_count"],
                    num_regions=batch["y"].size(0),
                    num_classes=model.num_classes,
                    line_c_allowed=line_c_allowed,
                            line_plus_region_a=batch["line_plus_region_a"],
                    side_logits=side_logits,
                )
                region_preds = vote_result[2]
                voted_mask = vote_result[4]
                counts = region_accuracy_counts(region_preds, batch["y"], voted_mask)
                val_region_correct += counts['correct']
                val_region_total += counts['total']
                val_region_voted_correct += counts['voted_correct']
                val_region_voted_total += counts['voted_total']
                val_region_unvoted += counts['unvoted']

    num_batches = max(len(val_loader), 1)
    avg_val_loss = val_loss_sum / num_batches
    val_pair_acc = val_correct_pair / max(val_total, 1)
    val_region_acc = val_region_correct / max(val_region_total, 1)
    val_side_acc_plus = side_correct_plus / max(side_total_plus, 1)
    val_side_acc_minus = side_correct_minus / max(side_total_minus, 1)
    val_side_acc_both = side_correct_both / max(side_total_both, 1)

    return {
        'val_loss': avg_val_loss,
        'val_pair_acc': val_pair_acc,
        'val_region_acc': val_region_acc,
        'val_region_correct': int(val_region_correct),
        'val_region_total': int(val_region_total),
        'val_region_voted_acc': val_region_voted_correct / max(val_region_voted_total, 1),
        'val_region_voted_correct': int(val_region_voted_correct),
        'val_region_voted_total': int(val_region_voted_total),
        'val_region_coverage': val_region_voted_total / max(val_region_total, 1),
        'val_region_unvoted': int(val_region_unvoted),
        'val_side_acc_plus': val_side_acc_plus,
        'val_side_acc_minus': val_side_acc_minus,
        'val_side_acc_both': val_side_acc_both,
    }

def compute_line_confusion_matrices(model, val_loader, device, inv_triple_map, line_palette, color_palette,
                                    line_color_to_fill=None):
    """对验证集做一次前向, 收集线级 (真值, 预测) 对, 输出三维拆分混淆矩阵.

    每条线对应一个 triple (a_color, b_color, line_color), 其中:
      - a_color / b_color = 线两侧 region 的颜色 (预测 = decode 的 pred_a/pred_b)
      - line_color       = 线自身墨色 (预测 = decode 的 pred_line)
    真值侧: label_a/label_b 来自 classify_lines, 线色真值从 triple_label 反查 inv_triple_map 第三维.

    Returns:
        dict: { 'confusion_a': (C,C) list, 'confusion_b': (C,C), 'confusion_line': (C,C),
                'top_confusions': [...] }  (C = len(color_palette))
    """
    model.eval()
    num_colors = len(color_palette)
    num_triples = model.num_triples

    # 累计四个混淆矩阵 (真值行, 预测列)
    cm_a = np.zeros((num_colors, num_colors), dtype=np.int64)
    cm_b = np.zeros((num_colors, num_colors), dtype=np.int64)
    cm_line = np.zeros((num_colors, num_colors), dtype=np.int64)
    cm_triple = np.zeros((num_triples, num_triples), dtype=np.int64)

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            h_l, h_r = model(
                batch["X_l"], batch["edge_index_ll"], batch["edge_attr_ll"],
                batch["color_index"], batch["edge_index_rr"],
                batch["line_region_a"], batch["line_region_b"],
                region_line_color_geo=batch["region_line_color_geo"],
                edge_relation_mask=batch["edge_relation_mask"],
                line_pos=batch.get("line_pos"),
                region_pos=batch.get("region_pos"), region_shape_feat=batch.get("region_shape_feat"),
                            )
            logits, side_logits, _, region_a, region_b, label_a, label_b = model.classify_lines(
                h_l, h_r,
                batch["line_region_a"], batch["line_region_b"],
                batch["line_label_a"], batch["line_label_b"],
            )

            inv_map = build_inv_map_array(inv_triple_map, logits.size(1))
            pred_ids, pred_a, pred_b, _ = decode_triple_predictions(logits, inv_map)

            # 线色预测改为在 region 投票完成后进行 (与 colorize_image_ll 口径一致)
            line_c_allowed = build_line_c_allowed(
                batch["color_index"], batch["is_color_line"],
                line_color_to_fill, model.num_classes, device)
            _, _, vote_result = predict_regions_from_logits(
                logits=logits,
                inv_map=inv_map,
                region_a=region_a,
                region_b=region_b,
                line_pixel_count=batch["line_pixel_count"],
                num_regions=batch["y"].size(0),
                num_classes=model.num_classes,
                line_c_allowed=line_c_allowed,
            )
            pred_line = predict_line_colors_after_vote(
                logits=logits,
                inv_map=inv_map,
                region_a=region_a,
                region_b=region_b,
                region_preds=vote_result[2],
                voted_mask=vote_result[4],
            )

            valid = batch["triple_label"] >= 0
            if not valid.any():
                continue
            valid = valid.cpu().numpy()
            la = label_a.cpu().numpy()[valid]
            lb = label_b.cpu().numpy()[valid]
            pa = pred_a.cpu().numpy()[valid]
            pb = pred_b.cpu().numpy()[valid]
            pl = pred_line.cpu().numpy()[valid]
            pid = pred_ids.cpu().numpy()[valid]

            # 线色真值: 从 triple_label 反查 inv_triple_map 第三维
            tl = batch["triple_label"].cpu().numpy()[valid]
            inv_map_np = np.asarray(inv_map)
            line_true = inv_map_np[tl, 2]   # 第三维 = line_color

            for i in range(len(la)):
                # 无序目标: 真值对规范化为 (较小色, 较大色), 与预测候选色 (min,max) 同序, 避免 a/b 顺序错配
                la_i, lb_i = int(la[i]), int(lb[i])
                tm_, tx_ = (la_i, lb_i) if la_i <= lb_i else (lb_i, la_i)
                cm_a[tm_, int(pa[i])] += 1
                cm_b[tx_, int(pb[i])] += 1
                cm_line[int(line_true[i]), int(pl[i])] += 1
                cm_triple[int(tl[i]), int(pid[i])] += 1

    # Top-K 错分对 (按频次), 用颜色 hex 便于人读
    def _hex(idx):
        if 0 <= idx < len(color_palette):
            c = color_palette[idx]
            return "#%02X%02X%02X" % (int(c[0]), int(c[1]), int(c[2]))
        return f"<{idx}>"

    def _top_confusions(cm, dim_name):
        rows, cols = np.nonzero(cm)
        pairs = []
        for r, c in zip(rows, cols):
            if r == c:
                continue
            pairs.append((int(cm[r, c]), int(r), int(c)))
        pairs.sort(reverse=True)
        return [{"dim": dim_name, "count": cnt, "true": _hex(r), "pred": _hex(c)}
                for cnt, r, c in pairs[:20]]

    top = (_top_confusions(cm_a, "a_color")
           + _top_confusions(cm_b, "b_color")
           + _top_confusions(cm_line, "line_color"))
    top.sort(key=lambda x: -x["count"])

    html = _build_confusion_html(cm_a, cm_b, cm_line, cm_triple, _hex, inv_triple_map, top[:30])

    return {
        "num_colors": num_colors,
        "num_triples": num_triples,
        "color_palette": [_hex(i) for i in range(num_colors)],
        "confusion_a": cm_a.tolist(),
        "confusion_b": cm_b.tolist(),
        "confusion_line": cm_line.tolist(),
        "confusion_triple": cm_triple.tolist(),
        "top_confusions": top[:30],
        "html": html,
    }

def _build_confusion_html(cm_a, cm_b, cm_line, cm_triple, hex_fn, inv_triple_map, top_confusions):
    """生成自包含 HTML 热力图: a/b/line 三矩阵 + triple 全矩阵并排 + Top 错分对表.

    行列头用真实颜色色块 (背景=该颜色) 标注, 便于直接对号分辨深色/浅色/邻近色.
    triple 矩阵行列头显示 (a,b,c) 三色块.
    """
    def _text_color(hexcode):
        # 根据颜色亮度选择黑/白文字, 保证色块上文字可读
        try:
            r = int(hexcode[1:3], 16); g = int(hexcode[3:5], 16); b = int(hexcode[5:7], 16)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            return "#000" if lum > 150 else "#fff"
        except Exception:
            return "#000"

    def _chip(hexcode, small=False):
        tc = _text_color(hexcode)
        fs = "9px" if small else "11px"
        return (f'<span style="display:inline-block;background:{hexcode};color:{tc};'
                f'padding:1px 4px;border-radius:3px;font-family:monospace;font-size:{fs};'
                f'border:1px solid #bbb;min-width:34px;text-align:center;">{hexcode}</span>')

    def _matrix_table(cm, title):
        n = cm.shape[0]
        hexs = [hex_fn(i) for i in range(n)]
        # 列头: 用色块
        head = "<tr><th></th>" + "".join(
            f'<th style="padding:3px 4px;">{_chip(h, small=True)}</th>' for h in hexs
        ) + "</tr>"
        rows_html = []
        for r in range(n):
            cells = [f'<th style="padding:2px 4px;text-align:right;">{_chip(hexs[r], small=True)}</th>']
            for c in range(n):
                v = int(cm[r, c])
                if r == c:
                    bg = "#e6f4ea"  # 对角线(正确)淡绿
                    txt = f'<b>{v}</b>' if v > 0 else ''
                elif v > 0:
                    # 错误: 按频次深浅红
                    intensity = min(v / max(cm.max(), 1), 1.0)
                    alpha = int(40 + 200 * intensity)
                    bg = f"rgba(220,60,60,{alpha/255:.2f})"
                    txt = str(v)
                else:
                    bg = "#fafafa"
                    txt = ''
                cells.append(f'<td style="background:{bg};text-align:center;font-size:10px;">{txt}</td>')
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        return f'<div class="mat"><h3>{title}</h3><table>{head}{"".join(rows_html)}</table></div>'

    # triple 矩阵: 行列头显示 (a,b,c) 三色块
    def _triple_head(idx):
        if inv_triple_map is None or idx not in inv_triple_map:
            return f"<{idx}>"
        a, b, c = inv_triple_map[idx]
        return f'{_chip(hex_fn(a), small=True)}{_chip(hex_fn(b), small=True)}{_chip(hex_fn(c), small=True)}'

    def _matrix_table_triple(cm, title):
        n = cm.shape[0]
        head = "<tr><th></th>" + "".join(
            f'<th style="padding:3px 2px;">{_triple_head(c)}</th>' for c in range(n)
        ) + "</tr>"
        rows_html = []
        for r in range(n):
            cells = [f'<th style="padding:2px 2px;text-align:right;">{_triple_head(r)}</th>']
            for c in range(n):
                v = int(cm[r, c])
                if r == c:
                    bg = "#e6f4ea"
                    txt = f'<b>{v}</b>' if v > 0 else ''
                elif v > 0:
                    intensity = min(v / max(cm.max(), 1), 1.0)
                    alpha = int(40 + 200 * intensity)
                    bg = f"rgba(220,60,60,{alpha/255:.2f})"
                    txt = str(v)
                else:
                    bg = "#fafafa"
                    txt = ''
                cells.append(f'<td style="background:{bg};text-align:center;font-size:9px;">{txt}</td>')
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        return (f'<div class="mat triple"><h3>{title}</h3>'
                f'<div class="scroll"><table>{head}{"".join(rows_html)}</table></div></div>')

    mats = (_matrix_table(cm_a, "a_color (线左侧区域)"),
            _matrix_table(cm_b, "b_color (线右侧区域)"),
            _matrix_table(cm_line, "line_color (线自身墨色)"),
            _matrix_table_triple(cm_triple, "triple (a_color,b_color,line_color)"))

    top_rows = "".join(
        f'<tr><td>{it["dim"]}</td><td>{_chip(it["true"])} → {_chip(it["pred"])}</td>'
        f'<td>{it["count"]}</td></tr>'
        for it in top_confusions
    )

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>线级混淆矩阵</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif; background:#f5f6f8; margin:20px; color:#222; }}
  h1 {{ font-size:18px; }}
  .mats {{ display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }}
  .mat {{ background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:10px; }}
  .mat h3 {{ font-size:13px; margin:0 0 8px; }}
  .mat.triple {{ max-width:100%; }}
  .scroll {{ max-width:900px; max-height:560px; overflow:auto; }}
  table {{ border-collapse:collapse; }}
  td, th {{ border:1px solid #e8e8e8; padding:2px 4px; }}
  .top {{ margin-top:20px; background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:10px; max-width:560px; }}
  .top h3 {{ font-size:13px; margin:0 0 8px; }}
  .top td {{ padding:4px 8px; font-size:12px; }}
  .legend {{ font-size:11px; color:#666; margin-top:8px; }}
</style></head><body>
<h1>线级混淆矩阵 (真值=行, 预测=列)</h1>
<div class="mats">{mats[0]}{mats[1]}{mats[2]}</div>
<div class="top"><h3>Top 错分对 (真值 → 预测)</h3>
<table><tr><th>维度</th><th>颜色</th><th>次数</th></tr>{top_rows}</table></div>
<div class="legend">绿色=对角线(判对); 红色深浅=错分频次; 单元值=样本数; 行列头=真实颜色</div>
</body></html>"""

def _rank_neg_weight(y, num_classes):
    """负类频率权重 (在出现过的类上均值为 1, 未出现的类为 0)。

    训练是 full-batch (`train_batch_size = len(train_dataset)`), 因此 batch 内计数即数据集频率。
    power=0 或关闭时返回 None (等价于原来的均匀权重)。
    """
    p = float(RANK_SVM_FREQ_POWER)
    if p <= 0.0:
        return None
    freq = torch.bincount(y.clamp(min=0), minlength=num_classes).to(torch.float32)
    nz = freq > 0
    w = torch.zeros_like(freq)
    if not bool(nz.any()):
        return None
    w[nz] = (freq[nz] / freq[nz].mean().clamp_min(1e-8)).pow(p)
    return w.clamp(max=float(RANK_SVM_FREQ_CLAMP))


def _triple_rank_svm(logits, triple_label, margin=None):
    """Rank-SVM 多标签 hinge (Elisseeff & Weston 2001)。

    对每条线 i, 真色对 y_i 为正标签, **只与训练中出现过的其它色对**比较:
        L_i = Σ_{n ∈ Present, n ≠ y_i} relu(m − (s[y_i] − s[n])) / |Present − 1|
    总损失 = 对有效线求均值。

    负类集合用**二值包含**而非频率加权 —— 频率加权只是把稀有负类的权重压到 0.049,
    它仍在竞争; 直接排除才符合"这些类没有样本、不该参与判别"的语义, 且与类分布无关。

    Args:
        logits: (N, NT) triple 分数。
        triple_label: (N,) 真 triple id, <0 表示无效。
        margin: 标量或 0 维 tensor; None 时用固定常量 RANK_SVM_MARGIN。
    """
    tl = triple_label.long()
    valid = tl >= 0
    if not valid.any():
        return logits.sum() * 0.0
    m = RANK_SVM_MARGIN if margin is None else margin
    lg = logits[valid]
    y = tl[valid]
    nt = lg.size(1)

    # 训练中出现过的类 (二值), 未出现的类完全不参与排序竞争
    present = torch.zeros(nt, dtype=torch.bool, device=lg.device)
    present.scatter_(0, torch.unique(y), True)

    pos = lg.gather(1, y.unsqueeze(1))
    viol = F.relu(m - (pos - lg))
    viol = viol.masked_fill(~present.unsqueeze(0), 0.0)
    viol = viol.scatter(1, y.unsqueeze(1), torch.zeros_like(pos))

    n_neg = (present.sum() - 1).clamp(min=1).to(lg.dtype)
    return viol.sum(dim=1).mean() / n_neg


def _cs_topk_hinge(logits, target, sample_weight=None, margin=None, topk=None,
                   reduction='mean'):
    """Crammer–Singer 式 margin 排序损失 (只约束最难的 top-K 竞争者)。

        L_i = (1/K) · Σ_{k ∈ TopK(z, 排除 y_i)} relu(m − (z[y_i] − z[k]))

    m = ln(1.2): 若满足, 则 softmax 下 p[y]/p[k] ≥ 1.2, 即"正确类概率高过竞争者 20%"。
    满足即零梯度 —— 这是它与 CE 的本质区别: CE 会持续把正确类概率推向 1, 于是不得不
    靠温度锐化; margin 是下界约束, 够用就停, 因此不需要大温度。

    只取 top-K (默认 5) 个竞争者: 真正需要被拉开的就是这 K 个; 对 101 类的长尾来说,
    约束远端的低分负类既无信息又会让梯度被长尾主导。

    target: (N,) long, <0 表示无标签 (该行不计损失)。
    sample_weight: (N,) 逐样本权重 (side 头用结构支持度)。
    """
    # 默认 margin: 相对口径 —— ln(C) / support, 由 _triple_loss / compute_loss 显式传入;
    #   未传时退化为 ln(C) (要求全分离, 即最严格的相对要求)。
    m = float(np.log(max(int(logits.size(1)), 2))) if margin is None else float(margin)
    K = CS_TOPK if topk is None else int(topk)
    tgt = target.long()
    valid = tgt >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0

    lg = logits[valid]
    tv = tgt[valid]
    z_y = lg.gather(1, tv.unsqueeze(1))                       # (n,1) 正确类
    neg = lg.scatter(1, tv.unsqueeze(1), float('-inf'))       # 排除正确类
    k = min(K, neg.size(1) - 1)
    z_comp = neg.topk(k, dim=1).values                        # (n,k) 最难的竞争者
    hinge = F.relu(m - (z_y - z_comp)).mean(dim=1)            # (n,)
    if sample_weight is not None:
        w = sample_weight[valid].to(hinge.dtype)
        if reduction == 'none':
            out = hinge * w
        else:
            out = (hinge * w).sum() / w.sum().clamp_min(1e-8)
        return out
    return hinge if reduction == 'none' else hinge.mean()


def _cs_mass_hinge(logits, target, delta=None, sample_weight=None, reduction='mean'):
    """概率质量 margin (替代 logit margin): 正确类须占住 (1+delta)/2 以上的质量。

        L_i = relu(delta − ( p[y_i] − Σ_{k≠y_i} p[k] ))
            = relu(1 + delta − 2·p[y_i])

    delta = 0.2 ⟺ 要求 p[y] ≥ 0.6: 正确类拿 60% 质量、错误类合计不超过 40%,
    两者之差至少 0.2。与 logit-margin 相比:
      - 直接用概率质量表达, 就是 unary / MRF 消费的那个量, 不需要 support 换算;
      - 类别数自动进入 (同样的 logit 差, C 越大 p 越小), 不必再按数据集调 support;
      - 有上界: 满足即零梯度, 不会像 CE 那样把概率一路推向 1 (0.6 就够, 不许过度锐化)。

    target: (N,) long, <0 表示无标签; sample_weight: (N,) 逐样本权重 (side 结构支持度)。
    """
    d = CS_MASS_DELTA if delta is None else float(delta)
    tgt = target.long()
    valid = tgt >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0

    p = F.softmax(logits[valid], dim=1)
    p_y = p.gather(1, tgt[valid].unsqueeze(1)).squeeze(1)      # (n,) 正确类质量
    hinge = F.relu(d - (2.0 * p_y - 1.0))                      # p_y >= (1+d)/2 时为零
    if sample_weight is not None:
        w = sample_weight[valid].to(hinge.dtype)
        if reduction == 'none':
            out = hinge * w
        else:
            out = (hinge * w).sum() / w.sum().clamp_min(1e-8)
        return out
    return hinge if reduction == 'none' else hinge.mean()


def _triple_ce(logits, triple_label):
    """主损失 (对照用): 普通 One-Hot 交叉熵 (line-level triple 分类)。"""
    tl = triple_label.long()
    valid = tl >= 0
    if valid.any():
        return F.cross_entropy(logits[valid], tl[valid],
                               label_smoothing=float(LABEL_SMOOTHING))
    return logits.sum() * 0.0


def _side_rank_svm(side_logits, y_plus, y_minus, w_plus=None, w_minus=None, margin=None):
    """逐侧 Rank-SVM 排序损失。真颜色分数应领先其它所有颜色至少一个 margin。

    L = 0.5 * mean_i [ w⁺·Σ_{n≠y⁺} relu(m - (s⁺[y⁺] - s⁺[n]))/(C-1)
                     + w⁻·Σ_{n≠y⁻} relu(m - (s⁻[y⁻] - s⁻[n]))/(C-1) ]

    w_plus / w_minus: 结构支持度权重。**必须与 CE 项用同一套权重** —— 否则两个目标
    会对同一批线给出不同的相对重要性, 互相拉扯 (实测无权版本 B/V 各 −0.6pp)。
    """
    m = SIDE_RANK_MARGIN if margin is None else float(margin)
    tot = None
    for k, y, w in ((0, y_plus, w_plus), (1, y_minus, w_minus)):
        if y is None:
            continue
        y = y.long()
        valid = y >= 0
        if not valid.any():
            continue
        lg = side_logits[:, k]
        c = lg.size(1)
        yc = y.clamp(min=0)
        pos = lg.gather(1, yc.unsqueeze(1))
        viol = F.relu(m - (pos - lg))
        viol = viol.scatter(1, yc.unsqueeze(1), torch.zeros_like(pos))
        nw = _rank_neg_weight(yc[valid], c)
        if nw is not None:
            viol = viol * nw.to(viol.dtype).unsqueeze(0)
        term = viol.sum(dim=1) / float(max(c - 1, 1))
        term = torch.where(valid, term, torch.zeros_like(term))
        if w is not None:
            term = term * w
        tot = term if tot is None else tot + term
    if tot is None:
        return side_logits.sum() * 0.0
    return (tot * 0.5).mean()


def _triple_loss(logits, triple_label, margin=None, cs_margin=None):
    """triple 主损失。

    - "ce"          : 交叉熵 (原行为)
    - "rank_svm"    : 纯 Rank-SVM 多标签 hinge (关闭 CE)
    - "ce_plus_rank": CE + λ·(排序损失 / 负类数) (对照用)

    margin 由调用方传入可学习张量 (model.head.rank_margin())。
    """
    t = str(TRIPLE_LOSS_TYPE).lower()
    if t in ("rank_svm", "rank", "svm"):
        return _triple_rank_svm(logits, triple_label, margin=margin)
    if t in ("ce_plus_rank", "ce_rank", "mixed"):
        n_neg = max(int(logits.size(1)) - 1, 1)
        # 注意: _triple_rank_svm 内部已 `return viol.sum(dim=1).mean() / n_neg`,
        # 即已按负类数归约; 这里不能再除一次 n_neg, 否则 rank 项被再压 1/(NT-1)≈1/100,
        # 使 "CE + λ·rank" 实际退化为纯 CE (清单第 5 项怀疑的"权重被意外削弱"即此)。
        rank = _triple_rank_svm(logits, triple_label, margin=margin)
        return _triple_ce(logits, triple_label) + float(RANK_SVM_AUX_WEIGHT) * rank
    if t in ("cs_mass", "mass"):
        return _cs_mass_hinge(logits, triple_label)
    if t in ("cs_topk", "cs", "margin"):
        return _cs_topk_hinge(logits, triple_label, margin=cs_margin)
    return _triple_ce(logits, triple_label)


def _synergy_posterior(pair_probs, q_a, q_b, pc0, pc1, in0, in1, eps=1e-8, beta=None,
                      mode="poe"):
    """乘性协同的联合后验 —— **训练与推理共用的唯一实现**。

    对每个候选色对 p = (c0, c1) 有两种朝向:

        A_p = P_triple(p) · q_a(c0)^β · q_b(c1)^β       # c0 在 a 侧
        B_p = P_triple(p) · q_a(c1)^β · q_b(c0)^β       # 交换

        prob_a = (Σ_p A_p·1[c=c0] + Σ_p B_p·1[c=c1]) / Σ_p (A_p + B_p)
        prob_b = (Σ_p A_p·1[c=c1] + Σ_p B_p·1[c=c0]) / Σ_p (A_p + B_p)

    分工: triple 提供"组合先验"(这两个颜色能否同时出现), side 提供"两侧似然"
    (每一侧与分配给它的颜色多匹配)。β = SYNERGY_BETA 软化 side 似然。

    抽成一个函数是为了让 `_joint_region_nll`(训练) 与 `solve_region_colors_mrf`(推理)
    **不可能漂移** —— 它们是同一个公式的两种消费方式。

    Args:
        pair_probs: (N, NP) triple 的无序色对分布
        q_a, q_b:   (N, C)  side 的逐侧颜色分布 (已是概率, 内部会做 q^β)
        pc0, pc1:   (NP,)   每个色对的较小色 / 较大色
        in0, in1:   (NP, C) 把 pair 质量映射到 c0 / c1 颜色上的 one-hot
    Returns:
        (prob_a, prob_b) 各 (N, C), 每行和为 1
    """
    if beta is None:
        _b = float(SYNERGY_BETA)
        if _b != 1.0:
            q_a = q_a.pow(_b)
            q_b = q_b.pow(_b)
    else:
        # beta: (N,1) 逐线软化系数 (自适应模式) —— side 不可靠的线自动退回 triple 先验
        q_a = torch.exp(beta * torch.log(q_a.clamp_min(eps)))
        q_b = torch.exp(beta * torch.log(q_b.clamp_min(eps)))
    if mode == "agree":
        # side 定朝向, triple 按"side 对朝向有多确定"加强/减弱色对权重:
        #   pi_p = sA/(sA+sB) (朝向),  kappa_p = 2|pi_p - 0.5| in [0,1] (一致性强度)
        #   weight_p = P_T(p) * kappa_p -> side 越确定越强, 越犹豫越弱; 完全犹豫时该色对权重归零。
        sA = (q_a[:, pc0] * q_b[:, pc1]).clamp_min(eps)
        sB = (q_a[:, pc1] * q_b[:, pc0]).clamp_min(eps)
        pi = sA / (sA + sB)
        kappa = (2.0 * pi - 1.0).abs()                      # (N, NP) in [0,1]
        wgt = pair_probs * kappa
        den = wgt.sum(dim=-1, keepdim=True)
        _fb = 0.5 * ((pair_probs @ in0) + (pair_probs @ in1))
        den_safe = den.clamp_min(eps)
        prob_a = ((wgt * pi) @ in0 + (wgt * (1.0 - pi)) @ in1) / den_safe
        prob_b = ((wgt * pi) @ in1 + (wgt * (1.0 - pi)) @ in0) / den_safe
        _ok = (den.squeeze(-1) > eps).unsqueeze(-1)
        prob_a = torch.where(_ok, prob_a, _fb)
        prob_b = torch.where(_ok, prob_b, _fb)
        return prob_a, prob_b, None

    if mode == "factorized":
        # P(p, o) = P_T(p) · P_S(o|p): triple 只决定"色对及其概率", side 只决定色对内的朝向。
        #   与 poe 的差别: poe 的聚合权重是 P_T(p)·(s_A+s_B), side 可借此**否决整个色对**;
        #   这里权重固定为 P_T(p), side 没有否决权, 只分配朝向。
        sA = (q_a[:, pc0] * q_b[:, pc1]).clamp_min(eps)     # (N, NP) c0 在 a 侧
        sB = (q_a[:, pc1] * q_b[:, pc0]).clamp_min(eps)     # (N, NP) 交换
        pi = sA / (sA + sB)                                 # 朝向概率 (P_T 已约掉)
        mass_p = pair_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        wa_ = pair_probs * pi
        wb_ = pair_probs * (1.0 - pi)
        prob_a = (wa_ @ in0 + wb_ @ in1) / mass_p
        prob_b = (wa_ @ in1 + wb_ @ in0) / mass_p
        _C = prob_a.size(1)
        pair_joint = None
        return prob_a, prob_b, pair_joint

    A = pair_probs * q_a[:, pc0] * q_b[:, pc1]              # (N, NP)
    B = pair_probs * q_a[:, pc1] * q_b[:, pc0]              # (N, NP)
    prob_a = A @ in0 + B @ in1                              # (N, C)
    prob_b = A @ in1 + B @ in0                              # (N, C)
    _mass = (A + B).sum(dim=-1, keepdim=True).clamp_min(eps)
    prob_a = prob_a / _mass
    prob_b = prob_b / _mass
    # 有序联合后验 P(+n 侧=i, -n 侧=j), 供 PAIRWISE_POT="joint" 使用。
    _C = prob_a.size(1)
    pair_joint = A.new_zeros((A.size(0), _C, _C))
    _r = torch.arange(A.size(0), device=A.device).unsqueeze(1).expand_as(A)
    pair_joint.index_put_((_r, pc0.view(1, -1).expand_as(A), pc1.view(1, -1).expand_as(A)),
                          A, accumulate=True)
    pair_joint.index_put_((_r, pc1.view(1, -1).expand_as(A), pc0.view(1, -1).expand_as(A)),
                          B, accumulate=True)
    pair_joint = pair_joint / pair_joint.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    return prob_a, prob_b, pair_joint


def _joint_region_nll(logits, side_logits, inv_map, region_a, region_b, region_y,
                      num_regions, num_classes, line_plus_region_a=None,
                      return_unary=False):
    """**单个** region NLL: 用乘性协同的联合后验, 梯度同时喂给两个分类器。

    与推理阶段 `solve_region_colors_mrf` 完全同口径。对每个候选色对 p = (c0, c1):

        A_p = P_triple(p) · q_a(c0)^β · q_b(c1)^β       # c0 在 a 侧
        B_p = P_triple(p) · q_a(c1)^β · q_b(c0)^β       # 交换

        prob_a = (Σ_p A_p·1[c=c0] + Σ_p B_p·1[c=c1]) / Σ_p (A_p + B_p)
        prob_b = (Σ_p A_p·1[c=c1] + Σ_p B_p·1[c=c0]) / Σ_p (A_p + B_p)

    triple 提供"组合先验"(这两个颜色能否同时出现), side 提供"两侧似然"(每侧与分配给它的
    颜色多匹配), 相乘构成完整证据。梯度经 A/B 的乘积**同时**回传两个头 —— 这是与
    "两个独立损失各训各的"最本质的区别: 两个头被同一个目标联合优化, 必须协同才能降低损失。

    逐线逐侧权重 = conf(该侧联合后验) = 1 - H/logC。
    聚合 = 加权平均 log 概率 → region → NLL。
    """
    device, dtype = logits.device, logits.dtype
    eps = 1e-8
    C = num_classes
    N, NT = logits.shape

    inv_map_t = torch.as_tensor(inv_map, dtype=torch.long, device=device)
    c0 = inv_map_t[:, 0].long()
    c1 = inv_map_t[:, 1].long()

    pair_key = c0 * C + c1
    uniq_pk, pair_id_of_t = torch.unique(pair_key, return_inverse=True)
    num_pairs = uniq_pk.numel()
    pc0 = uniq_pk // C
    pc1 = uniq_pk % C

    probs = F.softmax(logits, dim=1)
    pair_probs = torch.zeros(N, num_pairs, device=device, dtype=dtype)
    pair_probs.scatter_add_(1, pair_id_of_t.unsqueeze(0).expand(N, NT), probs)

    in0 = torch.zeros(num_pairs, C, device=device, dtype=dtype)
    in1 = torch.zeros(num_pairs, C, device=device, dtype=dtype)
    pair_col = torch.arange(num_pairs, device=device)
    in0[pair_col, pc0] = 1.0
    in1[pair_col, pc1] = 1.0

    # ---- 乘性协同的联合后验 (与推理共用同一实现) ----
    q_a = torch.softmax(side_logits[:, 0], dim=-1).clamp(min=eps)
    q_b = torch.softmax(side_logits[:, 1], dim=-1).clamp(min=eps)
    if UNARY_MODE == "side":
        prob_a, prob_b = q_a, q_b
    elif UNARY_MODE == "triple":
        # 训练侧没有 pair_in0/pair_in1 变量, 直接用 pair_probs @ in0/in1 取边际
        prob_a = 0.5 * (pair_probs @ in0 + pair_probs @ in1)
        prob_b = prob_a
    else:
        _beta = None
        if UNARY_MODE == "adaptive":
            _beta = (_conf_from_probs(q_a) * _conf_from_probs(q_b)).unsqueeze(1)
        prob_a, prob_b, _ = _synergy_posterior(
            pair_probs, q_a, q_b, pc0, pc1, in0, in1, eps, beta=_beta,
            mode=("factorized" if UNARY_MODE == "factorized"
                          else "agree" if UNARY_MODE == "agree" else "poe"))

    w_a = _conf_from_probs(prob_a).clamp(min=eps)           # (N,)
    w_b = _conf_from_probs(prob_b).clamp(min=eps)
    log_prob_a = torch.log(prob_a.clamp(min=eps))
    log_prob_b = torch.log(prob_b.clamp(min=eps))

    ra = region_a.to(device).long()
    rb = region_b.to(device).long()
    if line_plus_region_a is not None:
        plus_is_a = line_plus_region_a.to(torch.bool)
        plus_region = torch.where(plus_is_a, ra, rb)
        minus_region = torch.where(plus_is_a, rb, ra)
    else:
        plus_region = ra
        minus_region = rb

    unary = torch.zeros(num_regions, C, device=device, dtype=dtype)
    mass = torch.zeros(num_regions, device=device, dtype=dtype)
    unary.index_add_(0, plus_region, log_prob_a * w_a.unsqueeze(1))
    unary.index_add_(0, minus_region, log_prob_b * w_b.unsqueeze(1))
    mass.index_add_(0, plus_region, w_a)
    mass.index_add_(0, minus_region, w_b)
    valid = mass > 0
    # 与推理侧 solve_region_colors_mrf 保持完全同口径: 除以真实 mass (不用 clamp(min=1.0))。
    if UNARY_MASS_MODE == "true":
        unary = unary / mass.clamp(min=eps).unsqueeze(1)
    else:
        unary = unary / mass.clamp(min=1.0).unsqueeze(1)

    log_p = F.log_softmax(unary, dim=1)
    region_y = region_y.to(device).long()
    nll = -log_p.gather(1, region_y.unsqueeze(1)).squeeze(1)
    if valid.any():
        nll_out = nll[valid].mean()
    else:
        nll_out = unary.sum() * 0.0
    if return_unary:
        return nll_out, unary
    return nll_out


def _region_support_rank_loss(unary, region_y, margin=0.1, eps=1e-8, mode=None):
    """Region 级 support 排序损失 (方案 A)。

    对同一张图内的 region 集合:
        support_i = softmax(unary_i)[argmax]     # 与推理侧 decision_support 同口径

    mode="logistic" (默认, **无任何超参**):
        L = mean_{j in wrong, i in correct} softplus(support_j - support_i) / ln2
        —— 即 RankNet / AUC 的成对 logistic 代理: 两端 support 相等时代价恰为 1.0,
           权重可以恒取 1.0, 不需要 margin, 也不是被截断的 hinge (不会饱和成 0 梯度)。
    mode="hinge" (旧版, 需要 margin):
        L = mean relu(support_j - support_i + margin)

    只用 region_y > 0 的 region (0 = 无法判定/背景回落, 不参与排序)。
    没有同时存在"对"和"错"时返回 0 (保持可微路径)。
    """
    _mode = str(REGION_RANK_MODE if mode is None else mode).lower()
    p = F.softmax(unary, dim=1)
    pred = p.argmax(dim=1)
    sup = p.gather(1, pred.unsqueeze(1)).squeeze(1)
    y = region_y.to(unary.device).long()
    valid = y > 0
    correct = (pred == y) & valid
    wrong = (pred != y) & valid
    if not (bool(correct.any()) and bool(wrong.any())):
        return unary.sum() * 0.0
    s_c = sup[correct]
    s_w = sup[wrong]
    gap = s_w.unsqueeze(1) - s_c.unsqueeze(0)          # (W, C)
    if _mode == "hinge":
        return torch.relu(gap + float(margin)).mean()
    if _mode == "auc":
        # 光滑 (1 - AUC): 每对 (错, 对) 记 σ(s_w - s_c) ∈ (0,1);
        #   排序完全正确 -> 0, 完全反了 -> 1, 随机 -> 0.5。本身就无量纲、无需 margin,
        #   权重恒取 1 时含义就是"错区域被错误地排在正确区域之上的比例"。
        return torch.sigmoid(gap).mean()
    # logistic: 与 support 同量纲、无量纲超参; 除以 ln2 使"并列"代价恰为 1
    return (F.softplus(gap) / math.log(2.0)).mean()



def _side_support_count(edge_index_ll, region_a, region_b, stroke_id, cont_stroke_id, line_plus_region_a):
    """统计每条线 +n/−n 两侧的「同粗化组同侧邻居」支持分数。

    对每条线 ℓ: 看经 L-L 边相邻的节点 dst, 若 dst 也以 ℓ 的 +n 侧 region 为邻, 则
    按同组匹配计分计入 sup_plus; 同理 sup_minus。计分: 每个同侧邻居在 stroke_id 同组
    记 1 分、在 cont_stroke_id 同组记 1 分 → 每个邻居贡献 0/1/2 分, 累加为支持度。
    返回 (sup_plus, sup_minus), 均为 (N,) float。
    """
    N = region_a.size(0)
    device = region_a.device
    plus_is_a = line_plus_region_a.to(torch.bool)
    plus_region = torch.where(plus_is_a, region_a, region_b)    # (N,) +n 侧 region
    minus_region = torch.where(plus_is_a, region_b, region_a)   # (N,) -n 侧 region
    # 无向 L-L 边在 batch 里只存一次 (src→dst), 必须先双向展开, 否则只有 src 端累计到支持度。
    src0 = edge_index_ll[0]
    dst0 = edge_index_ll[1]
    src = torch.cat([src0, dst0], dim=0)                        # 双向: 每条无向边贡献两端
    dst = torch.cat([dst0, src0], dim=0)
    # 每个邻居的计分: stroke 同组 +1, cont 同组 +1 → 0/1/2
    same_stroke = (stroke_id[src] == stroke_id[dst]).to(torch.float32)      # (4E,)
    if cont_stroke_id is not None:
        same_cont = (cont_stroke_id[src] == cont_stroke_id[dst]).to(torch.float32)
    else:
        same_cont = same_stroke
    group_score = same_stroke + same_cont                                   # ∈ {0,1,2}
    ra_dst = region_a[dst]
    rb_dst = region_b[dst]
    share_plus = (ra_dst == plus_region[src]) | (rb_dst == plus_region[src])
    share_minus = (ra_dst == minus_region[src]) | (rb_dst == minus_region[src])
    sup_plus = torch.zeros(N, device=device, dtype=torch.float32)
    sup_minus = torch.zeros(N, device=device, dtype=torch.float32)
    sup_plus.scatter_add_(0, src, group_score * share_plus.to(torch.float32))
    sup_minus.scatter_add_(0, src, group_score * share_minus.to(torch.float32))
    return sup_plus, sup_minus


def _support_weight(sup):
    """结构支持度 → 侧 loss 权重 (单调递增, 由 SIDE_SUPPORT_MODE 决定)。"""
    if SIDE_SUPPORT_MODE == "linear":
        return 1.0 + sup
    elif SIDE_SUPPORT_MODE == "log":
        return 1.0 + torch.log1p(sup)
    else:  # saturating
        return 1.0 + sup / (1.0 + sup)


# ===== 消融层：loss 计算与加权分离 =====
# weights 运行时取全局开关；不修改 ClassifierHead 的参数、梯度连接或 Region NLL 公式。
LOSS_TERM_ORDER = ('triple', 'region', 'side', 'rank')


def get_loss_weights():
    """返回本步真实系数。"""
    return {
        'triple': float(TRIPLE_LOSS_WEIGHT),
        'region': float(REGION_LOSS_WEIGHT),
        'side': float(SIDE_LOSS_WEIGHT),
        'rank': float(REGION_RANK_WEIGHT),
    }


def combine_loss_terms(raw_terms, weights):
    """将独立 loss 组合成反向目标；缺失项为 None，不参与求和。

    返回 total, weighted_terms；保留 0×loss 的可微路径以支持单项消融。
    """
    weighted = {}
    total = None
    for name in LOSS_TERM_ORDER:
        value = raw_terms.get(name)
        if value is None:
            continue
        term = weights[name] * value
        weighted[name] = term
        total = term if total is None else total + term
    if total is None:
        raise ValueError('没有可计算的 loss 分量')
    return total, weighted


def compute_loss(
    logits,
    triple_label,
    region_a,
    region_b,
    line_pixel_count,
    num_regions,
    num_classes,
    inv_triple_map,
    mode='train',
    line_c_allowed=None,
    edge_index_ll=None,
    pair_transition=None,
    region_y=None,
    label_a=None,
    label_b=None,
    side_logits=None,
    line_plus_region_a=None,
    stroke_id=None,
    cont_stroke_id=None,
    rank_margin=None,
):
    """Triple 主损失 + Side 两侧监督 + Triple×Side 联合 Region NLL。

    Triple 损失由 TRIPLE_LOSS_TYPE 决定（默认带 label smoothing 的 CE）。
    mode='train' 时联合 Region NLL 和 Side Loss 会反向传播；MRF 求解只用于
    Graph RegionAcc 监控，运行于 no_grad，不通过 BP 反传。
    pair_transition 保留向后兼容，但当前损失未使用该参数。
    已无独立 Direction Head / dir_loss。
    """
    device = logits.device
    # 相对 margin: 单线证据需要多强的分离, 取决于下游要聚合几条线才能分辨 C 个候选。
    #   一条 region 的决定由 support ≈ (线数/region 数) 条线的证据叠加而成, 故单线的
    #   概率比只需 ~C^(1/support) 即可在聚合后分辨全部候选 -> margin = ln(C)/support。
    #   两个量都直接来自本 batch 的图结构, 不是扫描出来的超参; support 下界取 1
    #   (等价于"一条线就要独立决策", 此时要求完整分离 ln(C))。
    _cs_support = 1.0
    if region_a is not None and num_regions is not None and int(num_regions) > 0:
        _cs_support = max(float(region_a.numel()) / float(num_regions), 1.0)
    _cs_margin_triple = float(np.log(max(int(logits.size(1)), 2))) / _cs_support
    _cs_margin_side = float(np.log(max(int(num_classes), 2))) / _cs_support
    loss_main = _triple_loss(logits, triple_label, margin=rank_margin,
                             cs_margin=_cs_margin_triple)

    inv_map = None
    if mode == 'train' and inv_triple_map is not None:
        inv_map = build_inv_map_array(inv_triple_map, logits.size(1))

    # --- Region 颜色监督: **单个联合损失**, 梯度同时喂给两个分类器 ---
    #   用乘性协同的联合后验 (与推理同口径), triple 的组合先验 × side 的两侧似然。
    #   两个头被同一个目标联合优化, 必须协同才能降低损失 (不再"各训各的")。
    region_loss = None
    region_unary = None
    if (mode == 'train' and inv_map is not None and region_y is not None
            and side_logits is not None):
        region_loss, region_unary = _joint_region_nll(
            logits, side_logits, inv_map, region_a, region_b, region_y,
            num_regions, num_classes,
            line_plus_region_a=line_plus_region_a, return_unary=True)

    # --- Region 级 support 排序 (方案 A): 只惩罚"错 region 的 support 高于对 region" ---
    rank_loss = None
    if (mode == 'train' and REGION_RANK_WEIGHT > 0.0
            and region_unary is not None and region_y is not None):
        rank_loss = _region_support_rank_loss(
            region_unary, region_y, margin=float(REGION_RANK_MARGIN))

    # --- Side-head 两侧颜色直接监督 (P1): 解耦 triple 联合分类目标 ---
    # 几何法向侧监督: side_logits[:,0]=+n侧, [:,1]=-n侧, 与 encode_sides 重排一致。
    # y_+/y_- 由 line_plus_region_a 决定 (region_a 在 +n 侧时 y_+=label_a), 而非大/小区域顺序。
    side_loss = None
    if mode == 'train' and side_logits is not None and label_a is not None and label_b is not None:
        if line_plus_region_a is not None:
            plus_a = line_plus_region_a.to(torch.bool)
            y_plus = torch.where(plus_a, label_a, label_b)
            y_minus = torch.where(plus_a, label_b, label_a)
        else:
            y_plus = label_a
            y_minus = label_b
        # 结构支持度加权: 同粗化组同侧邻居越多, 该侧 loss 权重越大。
        w_plus = torch.ones(side_logits.size(0), device=device)
        w_minus = torch.ones(side_logits.size(0), device=device)
        if edge_index_ll is not None and stroke_id is not None and line_plus_region_a is not None:
            sup_plus, sup_minus = _side_support_count(
                edge_index_ll, region_a, region_b, stroke_id, cont_stroke_id, line_plus_region_a)
            w_plus = _support_weight(sup_plus)
            w_minus = _support_weight(sup_minus)
        # 主损失: 由 SIDE_LOSS_TYPE 选择 (默认 "ce" = 基准行为)
        if str(SIDE_LOSS_TYPE).lower() in ("cs_mass", "mass"):
            ce_plus = _cs_mass_hinge(side_logits[:, 0], y_plus, reduction='none')    # (N,)
            ce_minus = _cs_mass_hinge(side_logits[:, 1], y_minus, reduction='none')  # (N,)
        elif str(SIDE_LOSS_TYPE).lower() in ("cs_topk", "cs", "margin"):
            ce_plus = _cs_topk_hinge(side_logits[:, 0], y_plus, reduction='none')    # (N,)
            ce_minus = _cs_topk_hinge(side_logits[:, 1], y_minus, reduction='none')  # (N,)
        else:
            ce_plus = F.cross_entropy(side_logits[:, 0], y_plus, reduction='none',
                                      label_smoothing=float(LABEL_SMOOTHING))   # (N,)
            ce_minus = F.cross_entropy(side_logits[:, 1], y_minus, reduction='none',
                                       label_smoothing=float(LABEL_SMOOTHING))  # (N,)
        if SIDE_OTHER_MODE == "reward":
            side_loss = (ce_plus * w_plus + ce_minus * w_minus).mean()
        elif SIDE_OTHER_MODE == "zero":
            # 非主导侧 (支持度低的一侧) 不奖励: CE 权重置零
            dom_plus = (sup_plus >= sup_minus)
            w_plus = torch.where(dom_plus, w_plus, torch.zeros_like(w_plus))
            w_minus = torch.where(~dom_plus, w_minus, torch.zeros_like(w_minus))
            side_loss = (ce_plus * w_plus + ce_minus * w_minus).mean()
        elif SIDE_OTHER_MODE == "entropy":
            # 非主导侧鼓励不预测: CE 换成负熵 (推均匀); 主导侧仍用 CE 奖励
            dom_plus = (sup_plus >= sup_minus)
            p_plus = F.softmax(side_logits[:, 0], dim=-1)
            p_minus = F.softmax(side_logits[:, 1], dim=-1)
            neg_entropy_plus = (p_plus * torch.log(p_plus.clamp(min=1e-8))).sum(dim=-1)   # -H ∈ [-log C, 0]
            neg_entropy_minus = (p_minus * torch.log(p_minus.clamp(min=1e-8))).sum(dim=-1)
            term_plus = torch.where(dom_plus, ce_plus * w_plus, neg_entropy_plus)
            term_minus = torch.where(~dom_plus, ce_minus * w_minus, neg_entropy_minus)
            side_loss = (term_plus + term_minus).mean()
        elif SIDE_OTHER_MODE == "soft":
            # 双侧都监督, 按支持度相对强度缩放并归一化到均值 1 (w_++w_-=2)。
            scale = ((w_plus + w_minus) * 0.5).clamp_min(1e-8)
            w_plus = w_plus / scale
            w_minus = w_minus / scale
            side_loss = (ce_plus * w_plus + ce_minus * w_minus).mean()
        else:
            side_loss = (ce_plus * w_plus + ce_minus * w_minus).mean()

        # --- side 头的 Rank-SVM 排序项 (多标签) ---
        #   side 只有 C 个类, 长尾远轻于 triple (B 的 101 个闭集色对), 排序压力可承受;
        #   "ce_plus_rank" 保留 CE 的概率归一化, 叠加排序 margin。
        _st = str(SIDE_LOSS_TYPE).lower()
        if _st in ("rank_svm", "rank", "svm"):
            side_loss = _side_rank_svm(side_logits, y_plus, y_minus, w_plus, w_minus)
        elif _st in ("ce_plus_rank", "ce_rank", "mixed") and SIDE_RANK_WEIGHT > 0.0:
            side_loss = side_loss + float(SIDE_RANK_WEIGHT) * _side_rank_svm(
                side_logits, y_plus, y_minus, w_plus, w_minus)

    # --- RegionAcc 监控 (no_grad, 仅统计不反向) ---
    region_preds_out = None
    voted_mask_out = None
    if mode == 'train' and inv_triple_map is not None:
        with torch.no_grad():
            _, _, vote_result = predict_regions_from_logits(
                logits=logits,
                inv_map=inv_map,
                region_a=region_a,
                region_b=region_b,
                line_pixel_count=line_pixel_count,
                num_regions=num_regions,
                num_classes=num_classes,
                line_c_allowed=line_c_allowed,
                line_plus_region_a=line_plus_region_a,
                side_logits=side_logits,
            )
        _, _, region_preds_out, _, voted_mask_out, _, _ = vote_result

    raw_terms = {
        'triple': loss_main,
        'region': region_loss,
        'side': side_loss,
        'rank': rank_loss,
    }
    loss_weights = get_loss_weights()
    total, weighted_terms = combine_loss_terms(raw_terms, loss_weights)

    return {
        'total': total,
        'main': loss_main,
        'region_loss': region_loss,
        'side_loss': side_loss,
        'region_preds': region_preds_out,
        'voted_mask': voted_mask_out,
        'raw_terms': raw_terms,
        'weighted_terms': weighted_terms,
        'loss_weights': loss_weights,
    }
    

def train_multitask_pretrain(backbone, heads, bundles, device, epochs, lr, save_path=None,
                             stop_event=None):
    """阶段一: 多数据集联合训练共享判别器 backbone + 每数据集独立分类头。

    所有数据集的 train 数据共享同一 backbone, 各自 head 分类 → backbone 学到跨数据集通用
    判别特征; 各 head 学到数据集特定的 triple 映射。训练结束后保存 backbone 供阶段二微调。

    Args:
        backbone: LineColorizerBackbone (共享判别器)。
        heads:    dict {name: ClassifierHead}, 每个数据集一个独立分类头。
        bundles:  dict {name: bundle}, 每个 bundle 含 train_loader/num_classes/inv_triple_map/
                  line_color_to_fill 等。
    """
    backbone = backbone.to(device)
    for name in heads:
        heads[name] = heads[name].to(device)

    opt_params = list(backbone.parameters())
    for name in heads:
        opt_params += list(heads[name].parameters())
    optimizer = AdamW(opt_params, lr=lr, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-4)
    def _stop_requested():
        return stop_event is not None and stop_event.is_set()

    print(f"[阶段一] 多任务联合训练判别器 backbone | Epochs: {epochs} | 数据集: {list(bundles.keys())}")
    for epoch in range(epochs):
        if _stop_requested():
            print("[阶段一] 用户停止")
            break
        backbone.train()
        for name in heads:
            heads[name].train()

        epoch_loss = 0.0
        n_batches = 0
        per_name_loss = {name: 0.0 for name in bundles}
        per_name_cnt = {name: 0 for name in bundles}

        for name, bundle in bundles.items():
            head = heads[name]
            for batch in bundle['train_loader']:
                if _stop_requested():
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                h_l, h_r = backbone(
                    batch['X_l'], batch['edge_index_ll'], batch['edge_attr_ll'],
                    batch['color_index'], batch['edge_index_rr'],
                    batch['line_region_a'], batch['line_region_b'],
                    region_line_color_geo=batch['region_line_color_geo'],
                    edge_relation_mask=batch['edge_relation_mask'],
                    stroke_id=batch.get('stroke_id'),
                    edge_index_stroke=batch.get('edge_index_stroke'),
                    edge_index_intra_stroke=batch.get('edge_index_intra_stroke'),
                    cont_stroke_id=batch.get('cont_stroke_id'),
                    edge_index_cont_stroke=batch.get('edge_index_cont_stroke'),
                    edge_index_intra_cont=batch.get('edge_index_intra_cont'),
                    line_pos=batch.get('line_pos'),
                    region_pos=batch.get('region_pos'), region_shape_feat=batch.get('region_shape_feat'))

                region_a = batch['line_region_a']
                region_b = batch['line_region_b']
                h_line, side_sum, z_plus, z_minus = backbone.encode_sides(
                    h_l, h_r, region_a, region_b, batch['line_plus_region_a'])
                logits, side_logits = head(h_line, side_sum, z_plus, z_minus)

                line_c_allowed = build_line_c_allowed(
                    batch['color_index'], batch['is_color_line'],
                    bundle['line_color_to_fill'], bundle['num_classes'], device)
                losses = compute_loss(
                    logits=logits,
                    triple_label=batch['triple_label'],
                    region_a=region_a,
                    region_b=region_b,
                    line_pixel_count=batch['line_pixel_count'],
                    num_regions=batch['y'].size(0),
                    num_classes=bundle['num_classes'],
                    inv_triple_map=bundle['inv_triple_map'],
                    mode='train',
                    line_c_allowed=line_c_allowed,
                    edge_index_ll=batch['edge_index_ll'],
                    pair_transition=bundle['pair_transition'],
                    region_y=batch['y'],
                    label_a=batch['line_label_a'],
                    label_b=batch['line_label_b'],
                    side_logits=side_logits,
                    line_plus_region_a=batch['line_plus_region_a'],
                    stroke_id=batch.get('stroke_id'),
                    cont_stroke_id=batch.get('cont_stroke_id'),
                    rank_margin=head.rank_margin(),
                )
                loss = losses['total']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=3.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                per_name_loss[name] += loss.item()
                per_name_cnt[name] += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        parts = " ".join(f"{n}={per_name_loss[n]/max(per_name_cnt[n],1):.3f}" for n in bundles)
        print(f"  [阶段一] Epoch {epoch+1:03d} | total_loss={avg:.4f} | {parts}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({'backbone_state': backbone.state_dict()}, save_path)
        print(f"  --> [阶段一] 判别器 backbone 已保存: {save_path}")
    return backbone


def train_ll_gnn(
    model,
    train_loader,
    val_loader,
    val_lines,
    val_colors,
    line_palette,
    color_palette,
    epochs,
    lr,
    device,
    save_path,
    patience,
    save_weights=True,
    progress_cb=None,
    stop_event=None,
    inv_triple_map=None,
    line_color_to_fill=None,
    pair_transition=None,
    freeze_backbone=False,
    save_meta=None,
    allow_overwrite=False,
):
    """
    训练循环，使用三元组分类。

    freeze_backbone=True 时冻结 backbone (判别器), 仅训练分类头 (head), 用于阶段二微调。
    """
    model = model.to(device)

    # X_l 标准化统计量: 仅端到端训练 (freeze_backbone=False) 时从训练集注入。
    #   阶段二 (freeze_backbone=True) 沿用阶段一 pretrained backbone 已保存的统计量,
    #   不重新计算, 避免与阶段一分布不一致。
    if not freeze_backbone:
        _mean, _std = compute_x_l_stats(train_loader, device)
        model.backbone.set_x_l_stats(_mean, _std)
        print(f"[标准化] X_l z-score 统计量已注入 (训练集):\n"
              f"  mean={_mean.tolist()}\n  std ={_std.tolist()}")

    swa_window = deque(maxlen=SWA_WINDOW)   # 滚动窗口: (epoch, 参数快照), 内存有界

    def _report(phase, **kw):
        if progress_cb is not None:
            try:
                progress_cb(phase, kw)
            except Exception:
                pass

    def _stop_requested():
        return stop_event is not None and stop_event.is_set()

    # --- 训练进度实时刷新渲染: train/val 两行对齐 (表格样式) ---
    # 终端 (tty) 下用 ANSI 控制序列覆盖刷新, 避免每 epoch 刷屏; 重定向到文件时降级为逐行打印。
    _tty = sys.stdout.isatty()
    _HEADER = (
        f"{'Epoch':>5} | {'Split':>5} | {'TripleLoss':>10} | {'PairAcc':>7} | "
        f"{'Graph RegionAcc (correct/total/wrong)':>40} | {'VoteCov':>7} | {'Time':>6}"
    )
    _SEP = "-" * len(_HEADER)
    _progress_on = False

    def _cell(v, w):
        """数值单元格: None → '-'，否则固定宽度右对齐。"""
        return f"{'-':>{w}}" if v is None else f"{v:{w}.4f}"

    def _render_epoch(epoch_no, tr_loss, tr_pair, tr_reg, tr_rc, tr_rt, tr_cov,
                      va_loss, va_pair, va_reg, va_rc, va_rt, va_cov, etime):
        nonlocal _progress_on
        def _reg(v, rc, rt):
            if v is None:
                return f"{'-':>40}"
            return f"{v:9.4f} ({rc}/{rt}/{rt - rc})".rjust(40)
        train_line = (
            f"{epoch_no:>5} | {'Train':>5} | {_cell(tr_loss, 10)} | {_cell(tr_pair, 7)} | "
            f"{_reg(tr_reg, tr_rc, tr_rt)} | {_cell(tr_cov, 7)} | {etime:5.1f}s"
        )
        val_line = (
            f"{'':>5} | {'Val':>5} | {_cell(va_loss, 10)} | {_cell(va_pair, 7)} | "
            f"{_reg(va_reg, va_rc, va_rt)} | {_cell(va_cov, 7)} |"
        )
        if _tty:
            if not _progress_on:
                print(_HEADER)
                print(_SEP)
                _progress_on = True
            else:
                sys.stdout.write("\x1b[1A\r")
            sys.stdout.write("\x1b[K" + train_line + "\n")
            sys.stdout.write("\x1b[K" + val_line)
            sys.stdout.flush()
        else:
            if not _progress_on:
                print(_HEADER)
                print(_SEP)
                _progress_on = True
            print(train_line)
            print(val_line)

    def _end_progress():
        nonlocal _progress_on
        if _progress_on and _tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _progress_on = False

    if freeze_backbone:
        # 冻结 backbone: 只优化 head 的参数; backbone 走 eval 模式 (关 dropout/DropEdge)。
        for p in model.backbone.parameters():
            p.requires_grad = False
        opt_params = list(model.head.parameters())
    else:
        opt_params = list(model.parameters())

    optimizer = AdamW(
        opt_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY
    )
    # 基于 epoch 数的余弦退火: 不依赖验证集, 符合"只用训练集判断"的约束。
    # 原 ReduceLROnPlateau 监控训练损失, 而训练损失单调递减 → 几乎永不触发衰减,
    # 全程以固定高 LR 训练、难以精细收敛, 且 SWA 平均的是高 LR 噪声区的参数。
    # 改为余弦退火让 LR 随训练推进平滑下降 (末期接近 eta_min), 提升收敛与 SWA 质量。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=1e-4,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    best_train_loss = float('inf')
    best_epoch = 0
    best_state = None   # total_loss 最低 epoch 的参数快照 (best 被挤出滚动窗口时兜底)
    loss_hist = deque(maxlen=max(EARLY_STOP_SLOPE_WINDOW, 1))   # 早停斜率观察窗口
    train_start_time = time.time()

    history = {
        "train_loss": [],
        "train_total_loss": [],
        "val_loss": [],
        "train_region_acc": [],
        "train_region_voted_acc": [],
        "train_region_coverage": [],
        "val_pair_acc": [],
        "val_region_acc": [],
        "val_region_voted_acc": [],
        "val_region_coverage": [],
        "epochs": [],
        "loss_components": [],   # [{triple,region,side,weighted_*}, ...]
    }

    print(f"开始训练 v6 Line-Centric L-L GNN | Epochs: {epochs} | hidden={HIDDEN_DIM}")
    _report("train_start", epochs=epochs)

    val_every = 1
    interrupted = False

    for epoch in range(epochs):
        if _stop_requested():
            interrupted = True
            _report("stopped", reason="user", epoch=epoch)
            break

        _report("epoch_start", epoch=epoch + 1, epochs=epochs)
        epoch_start = time.time()
        if freeze_backbone:
            model.backbone.eval()   # 冻结的 backbone 关闭 dropout/DropEdge, 确定性前向
            model.head.train()
        else:
            model.train()
        total_loss_main = 0.0
        total_loss_total = 0.0
        _component_sums = {}
        _component_counts = {}
        correct_pair = total_lines = 0
        train_region_correct = train_region_total = 0
        train_region_voted_correct = train_region_voted_total = 0
        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            if _stop_requested():
                interrupted = True
                _report("stopped", reason="user", epoch=epoch + 1)
                break

            # 批次搬到 device (与验证循环一致; 缺失时 CUDA 下模型在 GPU 而输入在 CPU 会直接报错)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            h_l, h_r = model(
                batch["X_l"],
                batch["edge_index_ll"],
                batch["edge_attr_ll"],
                batch["color_index"],
                batch["edge_index_rr"],
                batch["line_region_a"],
                batch["line_region_b"],
                                region_line_color_geo=batch["region_line_color_geo"],
                edge_relation_mask=batch["edge_relation_mask"],
                stroke_id=batch.get("stroke_id"),
                edge_index_stroke=batch.get("edge_index_stroke"),
                edge_index_intra_stroke=batch.get("edge_index_intra_stroke"),
                cont_stroke_id=batch.get("cont_stroke_id"),
                edge_index_cont_stroke=batch.get("edge_index_cont_stroke"),
                edge_index_intra_cont=batch.get("edge_index_intra_cont"),
                line_pos=batch.get("line_pos"),
                region_pos=batch.get("region_pos"), region_shape_feat=batch.get("region_shape_feat"),
            )

            (logits, side_logits,
             _, region_a, region_b, label_a, label_b) = model.classify_lines(
                h_l,
                h_r,
                batch["line_region_a"],
                batch["line_region_b"],
                batch["line_label_a"],
                batch["line_label_b"],
                batch["line_plus_region_a"],
            )

            # 训练阶段
            line_c_allowed = build_line_c_allowed(
                batch["color_index"], batch["is_color_line"],
                line_color_to_fill, model.num_classes, device)
            losses = compute_loss(
                logits=logits,
                triple_label=batch["triple_label"],
                region_a=region_a,
                region_b=region_b,
                line_pixel_count=batch["line_pixel_count"],
                num_regions=batch["y"].size(0),
                num_classes=model.num_classes,
                inv_triple_map=inv_triple_map,
                mode='train',
                line_c_allowed=line_c_allowed,
                edge_index_ll=batch["edge_index_ll"],
                pair_transition=pair_transition,
                region_y=batch["y"],
                label_a=label_a,
                label_b=label_b,
                side_logits=side_logits,
                line_plus_region_a=batch["line_plus_region_a"],
                stroke_id=batch.get("stroke_id"),
                cont_stroke_id=batch.get("cont_stroke_id"),
                rank_margin=model.head.rank_margin(),
            )
            loss = losses['total']
            total_loss_main += losses['main'].item()
            total_loss_total += losses['total'].item()
            for _kind, _terms in (('raw', losses['raw_terms']),
                                   ('weighted', losses['weighted_terms'])):
                for _name, _value in _terms.items():
                    if _value is None:
                        continue
                    _key = f'{_kind}_{_name}'
                    _component_sums[_key] = _component_sums.get(_key, 0.0) + float(_value.detach().item())
                    _component_counts[_key] = _component_counts.get(_key, 0) + 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            if inv_triple_map is not None:
                inv_map = build_inv_map_array(inv_triple_map, logits.size(1))
                _, pred_a, pred_b, _ = decode_triple_predictions(logits, inv_map)
                valid = batch["triple_label"] >= 0

                if valid.any():
                    # 统一"无序色对"口径: 真值对与预测对均规范化为 (min,max), 再逐分量匹配。
                    # (无序 triple 下 pred_a/pred_b 已是 min/max, 此 min/max 化是无操作;
                    #   有序 triple 下 pred_a/pred_b 为 (a,b), 需 min/max 化才能跨方案可比)
                    _la, _lb = label_a[valid], label_b[valid]
                    tmin, tmax = torch.minimum(_la, _lb), torch.maximum(_la, _lb)
                    pa, pb = pred_a[valid], pred_b[valid]
                    pmin, pmax = torch.minimum(pa, pb), torch.maximum(pa, pb)
                    correct_pair += ((pmin == tmin) & (pmax == tmax)).sum().item()
                    total_lines += valid.sum().item()

            # ---- 训练集 Region 准确率 (复用 compute_loss 内的 vote_result, 不二次传播, 无额外建图) ----
            # region_preds/voted_mask 来自 vote_result 的 argmax/bool 分量 (无梯度), backward 后仍可用。
            region_preds = losses.get('region_preds')
            voted_mask = losses.get('voted_mask')
            if region_preds is not None and voted_mask is not None:
                counts = region_accuracy_counts(region_preds, batch["y"], voted_mask)
                train_region_correct += counts['correct']
                train_region_total += counts['total']
                train_region_voted_correct += counts['voted_correct']
                train_region_voted_total += counts['voted_total']

            _report(
                "batch",
                epoch=epoch + 1,
                epochs=epochs,
                batch=batch_idx + 1,
                batches=num_batches,
                loss=loss.item(),
            )

        if interrupted:
            break
        avg_main_loss = total_loss_main / max(len(train_loader), 1)       # 主损失 (仅记录/展示)
        avg_train_loss = total_loss_total / max(len(train_loader), 1)     # 总损失 (模型选择/早停/SWA 依据)
        scheduler.step()   # 余弦退火: 按 epoch 数推进, 不依赖任何验证指标

        train_pair_acc = correct_pair / max(total_lines, 1)
        train_region_acc = train_region_correct / max(train_region_total, 1)
        train_region_voted_acc = train_region_voted_correct / max(train_region_voted_total, 1)
        train_region_coverage = train_region_voted_total / max(train_region_total, 1)
        epoch_time = time.time() - epoch_start

        do_val = (epoch + 1) % val_every == 0 or epoch == 0 or epoch == epochs - 1

        avg_val_loss = None
        val_pair_acc = val_region_acc = val_region_coverage = val_region_voted_acc = None
        val_region_correct = val_region_total = 0

        if do_val and val_loader:
            val_metrics = evaluate_model(model, val_loader, device, inv_triple_map,
                                         line_color_to_fill=line_color_to_fill)
            avg_val_loss = val_metrics['val_loss']
            val_pair_acc = val_metrics['val_pair_acc']
            val_region_acc = val_metrics['val_region_acc']
            val_region_correct = val_metrics.get('val_region_correct', 0)
            val_region_total = val_metrics.get('val_region_total', 0)
            val_region_coverage = val_metrics['val_region_coverage']
            val_region_voted_acc = val_metrics['val_region_voted_acc']

        _render_epoch(
            epoch + 1,
            avg_main_loss, train_pair_acc, train_region_acc,
            train_region_correct, train_region_total, train_region_coverage,
            avg_val_loss, val_pair_acc, val_region_acc,
            val_region_correct, val_region_total, val_region_coverage,
            epoch_time,
        )

        history["loss_components"].append({
            k: _component_sums[k] / _component_counts[k] for k in _component_sums
        })
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_main_loss)        # 主损失 (展示口径不变)
        history["train_total_loss"].append(avg_train_loss) # 总损失 (模型选择口径)
        history["train_region_acc"].append(train_region_acc)
        history["train_region_voted_acc"].append(train_region_voted_acc)
        history["train_region_coverage"].append(train_region_coverage)

        if val_loader is not None and do_val:
            history["val_loss"].append(avg_val_loss)
            history["val_pair_acc"].append(val_pair_acc)
            history["val_region_acc"].append(val_region_acc)
            history["val_region_voted_acc"].append(val_region_voted_acc)
            history["val_region_coverage"].append(val_region_coverage)
        else:
            history["val_loss"].append(None)
            history["val_pair_acc"].append(None)
            history["val_region_acc"].append(None)
            history["val_region_voted_acc"].append(None)
            history["val_region_coverage"].append(None)

        _report(
            "epoch_end",
            epoch=epoch + 1,
            epochs=epochs,
            train_loss=avg_main_loss,
            val_loss=avg_val_loss,
            train_pair_acc=train_pair_acc,
            train_region_acc=train_region_acc,
            train_region_coverage=train_region_coverage,
            val_region_acc=val_region_acc,
            val_region_coverage=val_region_coverage,
            epoch_time=epoch_time,
        )

        # 保存当前 epoch 快照到 SWA 滚动窗口 (deque, 内存有界)。
        snapshot = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
        swa_window.append((epoch + 1, snapshot))

        # --- 早停: 看 total_loss 在一段 epoch 窗口内的斜率 (每 epoch 平均下降) ---
        loss_hist.append(avg_train_loss)
        if EARLY_STOP_SLOPE_TOL > 0.0 and len(loss_hist) == loss_hist.maxlen:
            slope = (loss_hist[0] - loss_hist[-1]) / (len(loss_hist) - 1)   # 平均每 epoch 下降
            if slope < EARLY_STOP_SLOPE_TOL:
                _end_progress()
                print(f"[早停] total_loss 斜率 {slope:.6f}/epoch < {EARLY_STOP_SLOPE_TOL} (窗口 {len(loss_hist)})")
                break

        # 模型选择: 记录 total_loss 最低的 snapshot (供 SWA 窗口补回, 不用于早停)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_state = snapshot

    total_time = time.time() - train_start_time
    minutes, seconds = divmod(int(total_time), 60)
    hours, minutes = divmod(minutes, 60)

    time_str = (
        f"{hours}h {minutes}m {seconds}s"
        if hours > 0
        else f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    )

    _end_progress()
    print(f"[训练完成] 共 {epoch+1} epochs | 总耗时: {time_str}")

    # --- 训练结束: 对 SWA 滚动窗口内 (最近 SWA_WINDOW 个 epoch) 的参数快照做等权平均 ---
    if swa_window:
        swa_states = [state for _, state in swa_window]
        lo_epoch = swa_window[0][0]
        hi_epoch = swa_window[-1][0]
        # 早停 patience==SWA_WINDOW, total_loss 最低点常恰好被挤出窗口 → 单独补回,
        # 保证"最低点附近"参与平均。
        if best_state is not None and best_epoch < lo_epoch:
            swa_states.append(best_state)

        print(
            f"[SWA] 平均最近 {len(swa_window)} 个 epoch 的权重 "
            f"(epoch {lo_epoch}~{hi_epoch}; total_loss 最低在 epoch {best_epoch})"
        )

        with torch.no_grad():
            param_dict = dict(model.named_parameters())

            for name, param in param_dict.items():
                avg_param = torch.stack(
                    [state[name] for state in swa_states],
                    dim=0
                ).mean(dim=0)

                param.copy_(
                    avg_param.to(
                        device=param.device,
                        dtype=param.dtype
                    )
                )
    else:
        print("[SWA] 没有可用的参数快照，保留当前模型权重")

    if save_weights:
        payload = {
            "model_state": model.state_dict(),
            "line_palette": {int(k): tuple(int(x) for x in v) for k, v in line_palette.items()},
            "color_palette": {int(k): tuple(int(x) for x in v) for k, v in color_palette.items()},
        }
        if isinstance(save_meta, dict):
            payload.update(save_meta)
        save_checkpoint_atomic(save_path, payload,
                               allow_overwrite=bool(allow_overwrite))
        print(f"  --> [保存] SWA 最终权重: {save_path}")



    _report(
        "train_done",
        epochs_run=epoch + 1,
        time_str=time_str,
        interrupted=interrupted,
        save_path=save_path,
    )

    # --- SWA 权重的最终验证指标（与 epoch 曲线的中间权重指标分开）。---
    final_metrics = None
    if val_loader is not None and inv_triple_map is not None:
        final_metrics = evaluate_model(model, val_loader, device, inv_triple_map,
                                       line_color_to_fill=line_color_to_fill)
        print("\n[最终 SWA 模型] 验证集指标:")
        print(f"  Loss:      {final_metrics['val_loss']:.4f}")
        print(f"  PairAcc:   {final_metrics['val_pair_acc']:.4f}")
        print(f"  SideAcc:   +n {final_metrics.get('val_side_acc_plus', 0):.4f} | "
              f"-n {final_metrics.get('val_side_acc_minus', 0):.4f} | "
              f"两侧都对 {final_metrics.get('val_side_acc_both', 0):.4f}   "
              f"(对照 PairAcc = triple 的线级色对准确率)")
        rc = final_metrics.get('val_region_correct', 0)
        rt = final_metrics.get('val_region_total', 0)
        print(f"  Graph RegionAcc (全部区域): {final_metrics['val_region_acc']:.4f}  ({rc}/{rt}/{rt - rc})")
        print(f"  Graph RegionAcc (仅投票): {final_metrics['val_region_voted_acc']:.4f}  "
              f"({final_metrics['val_region_voted_correct']}/{final_metrics['val_region_voted_total']})")
        print(f"  Region 投票覆盖率: {final_metrics['val_region_coverage']:.4f} "
              f"(未投票 {final_metrics['val_region_unvoted']} 个)")
    else:
        print("\n[最终 SWA 模型] 无验证集或 inv_triple_map，跳过最终评估")

    final_pixel_metrics = None
    if val_lines and val_colors:
        pixel_acc, pixel_miou, pixel_cm = evaluate_pixel_level(
            model,
            val_lines,
            val_colors,
            line_palette,
            color_palette,
            device,
            inv_triple_map,
            fill_lineart=True,
        )
        final_pixel_metrics = {'pixel_acc': pixel_acc, 'pixel_miou': pixel_miou,
                               'pixel_metric_defined': bool(pixel_cm.sum()),
                               'evaluated_pixels': int(pixel_cm.sum())}

    # 指标日志在 SWA 最终验证 / 最终成图评价之后保存，避免只记录训练曲线而漏记最终结果。
    if save_weights:
        with open(os.path.splitext(save_path)[0] + '_ablation_log.json', 'w', encoding='utf-8') as _f:
            json.dump({
                'metrics_schema': 'v2_graph_region_all_plus_image_rgba',
                'metric_scope': {'train_region_acc': 'Graph RegionAcc (all GT regions)',
                                 'val_region_acc': 'Graph RegionAcc (all GT regions)',
                                 'pixel_acc': 'known GT palette pixels',
                                 'pixel_miou': 'macro IoU (present palette classes)'},
                'loss_weights': get_loss_weights(),
                'sideinferw': SIDE_INFER_WEIGHT,
                'mrf_lambda': MRF_MISMATCH_WEIGHT,
                'seed_note': '见调用命令 --seed',
                'history': history,
                'final_swa_val_metrics': final_metrics,
                'final_image_metrics': final_pixel_metrics,
            }, _f, ensure_ascii=False, indent=2)

    _plot_training_curves(history)

    return model

def evaluate_pixel_level(
    model, val_lines, val_colors, line_palette, color_palette,
    device, inv_triple_map, fill_lineart=True,
):
    """独立成图评价入口：让 gnn_common 直接比较线稿/最终 RGBA 输出/GT。

    返回 pixel_acc, pixel_miou, confusion；额外打印最终成图的 RegionAcc，
    与训练/验证阶段骨架图节点的 Graph RegionAcc 严格区分。
    """
    if not val_lines or not val_colors:
        print("\n[像素级评估] 无验证图像，跳过")
        return 0.0, 0.0, None
    # 此 API 的两份列表已由调用者显式配对；数据集入口另行检查文件名对应关系。
    if len(val_lines) != len(val_colors):
        raise ValueError('独立成图评价: 线稿与 GT 数量不一致')

    model.eval()
    num_classes = len(color_palette)
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    totals = dict(region_correct=0, region_total=0, pixel_total=0,
                  pixel_correct_exact_rgba=0, pixel_total_exact_rgba=0,
                  excluded_line_pixels=0, excluded_unknown_gt_pixels=0,
                  unknown_prediction_pixels=0, total_fill_pixels=0)
    processed = 0
    per_image = []
    for idx, (line_path, gt_path) in enumerate(zip(val_lines, val_colors), 1):
        start = time.time()
        print(f"[成图评估] {idx}/{len(val_lines)} {os.path.basename(line_path)} ...", flush=True)
        # 直接调用现有推理 API 获取含原线稿的最终成图；不要求先保存到磁盘。
        rendered_result = colorize_image_ll(
            model, line_path, line_palette, color_palette, device,
            inv_triple_map=inv_triple_map, fill_lineart=fill_lineart,
        )
        rendered = (rendered_result[0] if isinstance(rendered_result, tuple)
                    else rendered_result)
        if rendered is None or not isinstance(rendered, np.ndarray):
            raise ValueError(f'推理结果不是图像: {line_path}')
        base = os.path.splitext(os.path.basename(line_path))[0]
        metrics = evaluate_coloring_images(
            line_path, rendered, gt_path, color_palette=color_palette,
            output_prefix=os.path.join('Debug', 'metrics', f'{idx:03d}_{base}'),
        )
        confusion += metrics['confusion']
        processed += 1
        per_image.append({
            'lineart': os.fspath(line_path), 'ground_truth': os.fspath(gt_path),
            'pixel_acc': metrics['pixel_acc'], 'pixel_miou': metrics['pixel_miou'],
            'pixel_acc_exact_rgba': metrics['pixel_acc_exact_rgba'],
            'region_acc': metrics['region_acc'],
            'region_correct': metrics['region_correct'], 'region_total': metrics['region_total'],
            'pixel_coverage': metrics['pixel_coverage'],
            'excluded_unknown_gt_pixels': metrics['excluded_unknown_gt_pixels'],
            'metrics_json': os.path.join('Debug', 'metrics', f'{idx:03d}_{base}_metrics.json'),
        })
        for key in totals:
            totals[key] += metrics[key]
        def _f(v, defined=True):
            return f'{v:.4f}' if defined else 'N/A'
        _pixel_defined = metrics['pixel_metric_defined']
        _region_defined = metrics['region_metric_defined']
        print(f"  成图 RegionAcc={_f(metrics['region_acc'],_region_defined)} "
              f"({metrics['region_correct']}/{metrics['region_total']}); "
              f"PixelAcc(调色板)={_f(metrics['pixel_acc'],_pixel_defined)}; "
              f"Pixel mIoU={_f(metrics['pixel_miou'],_pixel_defined)}; "
              f"严格RGBA PixelAcc={_f(metrics['pixel_acc_exact_rgba'],metrics['total_fill_pixels']>0)}; "
              f"像素覆盖率={metrics['pixel_coverage']:.3f}; "
              f"耗时={time.time()-start:.1f}s", flush=True)

    acc, miou, class_iou = pixel_scores_from_confusion(confusion, num_classes)
    rc, rt = totals['region_correct'], totals['region_total']
    print(f"\n[成图评估汇总] 图片 {processed}/{len(val_lines)}")
    _pix_defined = totals['pixel_total'] > 0
    _exact_defined = totals['pixel_total_exact_rgba'] > 0
    _reg_defined = rt > 0
    def _f(v, defined=True):
        return f'{v:.4f}' if defined else 'N/A'
    print(f"  PixelAcc (调色板匹配): {_f(acc,_pix_defined)}    "
          f"Pixel mIoU: {_f(miou,_pix_defined)} (有效类别 {len(class_iou)})")
    print(f"  PixelAcc (严格RGBA，无漏评): "
          f"{_f(totals['pixel_correct_exact_rgba']/max(totals['pixel_total_exact_rgba'],1),_exact_defined)} "
          f"({totals['pixel_correct_exact_rgba']:,}/{totals['pixel_total_exact_rgba']:,})")
    print(f"  成图 RegionAcc: {_f(rc/max(rt,1),_reg_defined)}  ({rc}/{rt})")
    print(f"  有效像素: {totals['pixel_total']:,}/{totals['total_fill_pixels']:,} "
          f"(排除线稿 {totals['excluded_line_pixels']:,} px；"
          f"未知 GT {totals['excluded_unknown_gt_pixels']:,} px；"
          f"未知预测 {totals['unknown_prediction_pixels']:,} px)")
    if totals['excluded_unknown_gt_pixels']:
        print("  [注意] GT 有未收录在调色板里的颜色，已排除于像素指标；"
              "请检查 *_metrics.json 的覆盖率，避免指标虚高。")
    summary = {
        'metrics_schema': 'v2_graph_region_all_plus_image_rgba',
        'image_count': processed,
        'pixel_acc': acc,
        'pixel_miou': miou,
        'class_ious': {str(k): float(v) for k, v in class_iou.items()},
        'pixel_acc_exact_rgba': (totals['pixel_correct_exact_rgba'] /
                                 max(totals['pixel_total_exact_rgba'], 1)),
        'region_acc': rc / max(rt, 1),
        'region_correct': rc, 'region_total': rt,
        'pixel_metric_defined': _pix_defined,
        'pixel_exact_metric_defined': _exact_defined,
        'region_metric_defined': _reg_defined,
        'pixel_coverage': totals['pixel_total'] / max(totals['total_fill_pixels'], 1),
        **totals,
        'per_image': per_image,
        'metric_scope': {
            'region_acc': '成图 RegionAcc：原线稿分区的 RGBA 众数准确率',
            'pixel_acc': '调色板已知 GT 像素上的准确率（未知 GT 排除）',
            'pixel_miou': '汇总混淆矩阵中出现过的有效颜色类 macro IoU',
            'pixel_acc_exact_rgba': '所有非线条像素 RGBA 严格匹配准确率',
        },
    }
    os.makedirs(os.path.join('Debug', 'metrics'), exist_ok=True)
    summary_path = os.path.join('Debug', 'metrics', 'summary_metrics.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [汇总日志] {summary_path}")
    print("  成图 Region 对错图见 Debug/metrics/*_errors.png")
    return acc, miou, confusion

def colorize_image_ll(
    model,
    line_path,
    line_palette,
    color_palette,
    device,
    inv_triple_map,
    edge_diag=True,
    fill_lineart=True,
    line_color_to_fill=None,
    mc_samples=None,
    out_dir=None,
    out_stem=None,
    gt_color_path=None,
    gt_graph=None,
    gt_line_path=None,
):
    """
    推理流程，使用三元组分类。

    mc_samples: MC Dropout 采样次数 (>1 = 开启, 推理期保持 dropout 开启并 K 次前向平均 logits)。
       None = 用模块全局 MC_DROPOUT_SAMPLES。
    out_dir / out_stem: 给定则把调试图 (<stem>_confidence.png / <stem>_target.png) 写到该目录,
       与彩色结果图同名对应; 未给定则回退到旧的 Debug/ 固定路径。
    gt_color_path: 给定则额外输出 <stem>_errors.png（众数级 Region 对错）
       与 <stem>_metrics.json (判对=绿, 判错=红, 线稿=黑)。region 的 GT 颜色取该 region
       内 GT 彩稿像素的众数, 与 extract_ll_graph 里构造 region_labels 的口径一致 (剔除线稿像素)。
    """
    if mc_samples is None:
        mc_samples = MC_DROPOUT_SAMPLES

    l_raw = cv2.imdecode(
        np.fromfile(line_path, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if l_raw is None:
        raise ValueError(f"无法读取线稿图像: {line_path}")

    l_rgba = process_alpha(l_raw)
    l_rgba = prepare_lineart_for_graph(l_rgba)  # 仅删除内封闭小孔，不删除触边 Region
    h_img, w_img = l_rgba.shape[:2]

    graph = extract_ll_graph(
        l_rgba,
        np.zeros_like(l_rgba),
        line_palette,
        color_palette,
        debug_prefix="test_v6",
    )

    if graph is None:
        print("[推理] 图提取失败, 返回透明图")
        return np.zeros((h_img, w_img, 4), dtype=np.uint8)

    num_regions = graph["num_regions"]
    markers = graph["markers"]
    skeleton_bin = graph["skeleton_bin"]
    N_l = graph["N_l"]

    X_l = graph["X_l"].to(device)
    edge_index_ll = graph["edge_index_ll"].to(device)
    edge_attr_ll = graph["edge_attr_ll"].to(device)
    edge_relation_mask = graph["edge_relation_mask"].to(device)
    color_index = graph["color_index"].to(device)
    edge_index_rr = graph["edge_index_rr"].to(device)

    line_region_a = graph["line_region_a"].to(device)
    line_region_b = graph["line_region_b"].to(device)

    # MC Dropout: 逐次 Softmax 后对概率做平均，而不是先平均 logits 再 Softmax。
    #   后者会掩盖少数采样明确反对当前预测的事实。
    #   模型全 LayerNorm (无 BatchNorm), train/eval 仅差在 dropout 开关, 切换安全。
    was_training = model.training
    do_mc = mc_samples is not None and mc_samples > 1
    if do_mc:
        model.train()   # 保持 dropout + DropEdge 开启
    else:
        model.eval()

    # 前向输入预提取 (不随 dropout 变化), 只 .to(device) 一次, K 次前向复用。
    _geo = graph["region_line_color_geo"].to(device) if "region_line_color_geo" in graph else None
    _stroke_id = graph["stroke_id"].to(device) if graph.get("stroke_id") is not None else None
    _edge_index_stroke = graph["edge_index_stroke"].to(device) if graph.get("edge_index_stroke") is not None else None
    _edge_index_intra_stroke = graph["edge_index_intra_stroke"].to(device) if graph.get("edge_index_intra_stroke") is not None else None
    _cont_stroke_id = graph["cont_stroke_id"].to(device) if graph.get("cont_stroke_id") is not None else None
    _edge_index_cont_stroke = graph["edge_index_cont_stroke"].to(device) if graph.get("edge_index_cont_stroke") is not None else None
    _edge_index_intra_cont = graph["edge_index_intra_cont"].to(device) if graph.get("edge_index_intra_cont") is not None else None
    _label_a = graph["line_label_a"].to(device)
    _label_b = graph["line_label_b"].to(device)
    _plus_a = graph["line_plus_region_a"].to(device) if "line_plus_region_a" in graph else None
    _line_pos = graph["line_pos"].to(device) if graph.get("line_pos") is not None else None
    _region_pos = graph["region_pos"].to(device) if graph.get("region_pos") is not None else None
    _region_shape = graph["region_shape_feat"].to(device) if graph.get("region_shape_feat") is not None else None

    def _forward_once():
        h_l, h_r = model(
            X_l,
            edge_index_ll,
            edge_attr_ll,
            color_index,
            edge_index_rr,
            line_region_a,
            line_region_b,
            region_line_color_geo=_geo,
            edge_relation_mask=edge_relation_mask,
            stroke_id=_stroke_id,
            edge_index_stroke=_edge_index_stroke,
            edge_index_intra_stroke=_edge_index_intra_stroke,
            cont_stroke_id=_cont_stroke_id,
            edge_index_cont_stroke=_edge_index_cont_stroke,
            edge_index_intra_cont=_edge_index_intra_cont,
            line_pos=_line_pos,
            region_pos=_region_pos,
            region_shape_feat=_region_shape,
        )
        logits, side_logits, _, region_a, region_b, _, _ = model.classify_lines(
            h_l,
            h_r,
            line_region_a,
            line_region_b,
            _label_a,
            _label_b,
            _plus_a,
        )
        return logits, side_logits, region_a, region_b

    with torch.no_grad():
        side_logits = None
        if do_mc:
            triple_prob_sum = side_prob_sum = None
            for _ in range(mc_samples):
                _logits, _side, region_a, region_b = _forward_once()
                _pt = F.softmax(_logits, dim=1)
                _ps = F.softmax(_side, dim=-1)
                triple_prob_sum = _pt if triple_prob_sum is None else triple_prob_sum + _pt
                side_prob_sum = _ps if side_prob_sum is None else side_prob_sum + _ps
            # log(mean p) 的 softmax 正好是 mean p；保留已有 logits 消费接口。
            logits = torch.log((triple_prob_sum / float(mc_samples)).clamp_min(1e-12))
            side_logits = torch.log((side_prob_sum / float(mc_samples)).clamp_min(1e-12))
        else:
            logits, side_logits, region_a, region_b = _forward_once()

        inv_map = build_inv_map_array(inv_triple_map, logits.size(1))

        _, pred_a_hard_t, pred_b_hard_t, _ = (
            decode_triple_predictions(logits, inv_map)
        )

        pred_a_hard = pred_a_hard_t.cpu().numpy()
        pred_b_hard = pred_b_hard_t.cpu().numpy()

        # top-3 triple 概率 (供 HTML 预测 Triple 栏展示): 每条线前 3 个 triple + 概率
        _probs = torch.softmax(logits, dim=1)               # (N_l, num_triples)
        _topk = torch.topk(_probs, k=min(3, _probs.size(1)), dim=1)
        top3_triples = []                                   # list of [{c0,c1,c2,prob}]
        for i in range(_probs.size(0)):
            row = []
            for k in range(_topk.indices.size(1)):
                tid = int(_topk.indices[i, k].item())
                t = inv_triple_map.get(tid, (0, 0, 0))      # (min色, max色, line色)
                row.append({
                    "c0": int(t[0]), "c1": int(t[1]), "c2": int(t[2]),
                    "prob": float(_topk.values[i, k].item()),
                })
            top3_triples.append(row)

        # ---- 乘性协同实际使用的两路 Top-4 (供 HTML 单节点面板) ----
        #   triple: 色对分布 P_triple(p) = 对线色求和 (与 _synergy_posterior 的输入一致;
        #           带线色的 triple 粒度不是投票用的那组数据)。
        #   side:   逐侧颜色分布 q_a / q_b。
        _C = int(model.num_classes)
        _inv_t = torch.as_tensor(inv_map, dtype=torch.long, device=_probs.device)
        _pkey = _inv_t[:, 0].long() * _C + _inv_t[:, 1].long()
        _upk, _pinv = torch.unique(_pkey, return_inverse=True)
        _pp = torch.zeros(_probs.size(0), _upk.numel(),
                          device=_probs.device, dtype=_probs.dtype)
        _pp.scatter_add_(1, _pinv.unsqueeze(0).expand(_probs.size(0), -1), _probs)
        _k4 = min(4, _upk.numel())
        _t4 = torch.topk(_pp, k=_k4, dim=1)
        top4_pairs = []
        for i in range(_pp.size(0)):
            row = []
            for k in range(_k4):
                pk = int(_upk[_t4.indices[i, k]].item())
                row.append({"c0": pk // _C, "c1": pk % _C,
                            "prob": float(_t4.values[i, k].item())})
            top4_pairs.append(row)

        if side_logits is not None:
            def _top4(dist):
                d = torch.softmax(dist, dim=-1)
                kk = min(4, d.size(1))
                tv, ti = torch.topk(d, k=kk, dim=1)
                return [[{"c": int(ti[i, k].item()), "prob": float(tv[i, k].item())}
                         for k in range(kk)] for i in range(d.size(0))]
            top4_side_a = _top4(side_logits[:, 0])
            top4_side_b = _top4(side_logits[:, 1])
        else:
            top4_side_a = top4_side_b = None

        if "line_pixel_count" in graph:
            line_pixel_count = graph["line_pixel_count"]

            if not torch.is_tensor(line_pixel_count):
                line_pixel_count = torch.as_tensor(
                    line_pixel_count,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                line_pixel_count = line_pixel_count.to(device).float()
        else:
            line_pixel_count = torch.tensor(
                [info["pixel_count"] for info in graph["line_info"]],
                dtype=torch.float32,
                device=device,
            )

        is_color_line = graph["is_color_line"].to(device)

        line_c_allowed = build_line_c_allowed(
            color_index, is_color_line, line_color_to_fill, model.num_classes, device)

        _mrf_diagnostic = {}
        prob_a_t, prob_b_t, vote_result = predict_regions_from_logits(
            logits=logits,
            inv_map=inv_map,
            region_a=region_a,
            region_b=region_b,
            line_pixel_count=line_pixel_count,
            num_regions=num_regions,
            num_classes=model.num_classes,
            line_c_allowed=line_c_allowed,
                        line_plus_region_a=_plus_a,
            side_logits=side_logits,
            diagnostic=_mrf_diagnostic,
        )

        (
            region_probs_t,
            region_weight_t,
            region_preds_t,
            region_confidence_t,
            voted_mask_t,
            _,
            _,
        ) = vote_result

        # 线色: region 颜色已由 MRF 求解确定, 由两侧 region 最终颜色锁定色对再选线色
        # (与用户约束"线最终投票与两边 region 颜色一致"直接对应: 色对 = {y_a, y_b})。
        pred_line_hard_t = predict_line_colors_after_vote(
            logits=logits,
            inv_map=inv_map,
            region_a=region_a,
            region_b=region_b,
            region_preds=region_preds_t,
            voted_mask=voted_mask_t,
        )
        pred_line_hard = pred_line_hard_t.cpu().numpy()
        # “未见 Triple”与“未见 Region 色对”必须分开计数：Triple 的第三维是线色。
        # 即使一条完整 Triple 不在表中，它的 Region 色对可能已在其他 Triple 中出现。
        _seen_triples = {tuple(map(int, triple[:3])) for triple in inv_map.tolist()}
        _seen_pairs = {(a, b) for a, b, _ in _seen_triples}
        _ra_np = region_a.detach().cpu().numpy()
        _rb_np = region_b.detach().cpu().numpy()
        _pred_np = region_preds_t.detach().cpu().numpy()
        _new_pairs = []
        _new_triples = []
        for _line_i in range(N_l):
            _ca = int(_pred_np[int(_ra_np[_line_i])])
            _cb = int(_pred_np[int(_rb_np[_line_i])])
            _a, _b = min(_ca, _cb), max(_ca, _cb)
            _new_pairs.append((_a, _b) not in _seen_pairs)
            _new_triples.append((_a, _b, int(pred_line_hard[_line_i])) not in _seen_triples)
        print(f"  最终颜色组合: 未见完整Triple {sum(_new_triples)}/{N_l}, "
              f"未见Region色对 {sum(_new_pairs)}/{N_l}；仅做诊断，不硬性拒绝")

        prob_a_lines = prob_a_t.cpu().numpy()
        prob_b_lines = prob_b_t.cpu().numpy()
        region_preds = region_preds_t.cpu().numpy()
        region_confidence = region_confidence_t.cpu().numpy()
        voted_mask = voted_mask_t.cpu().numpy()
        region_weight = region_weight_t.cpu().numpy()
        # BP belief 的 softmax 仅作相对得分诊断，不是经过校准的正确概率。
        consistency_all = region_probs_t.max(dim=1).values.cpu().numpy()
        region_disagreement = _mrf_diagnostic['line_disagreement'].cpu().numpy()
        region_unary_probs = _mrf_diagnostic['unary_probs'].cpu().numpy()
        region_incidents = _mrf_diagnostic['incident_count'].cpu().numpy()
        region_bp_independent = _mrf_diagnostic['bp_independent_labels'].cpu().numpy()

        no_vote_regions = np.where(~voted_mask)[0].tolist()
        CONFIDENCE_THRESHOLD = 0.4  # 仅用于统计打印"低置信区域数"; 热力图已改为渐变, 此值不再代表色阶界限
        n_voted = int(np.sum(voted_mask))

        if n_voted > 0:
            conf_voted = region_confidence[voted_mask]
            mean_conf = float(np.mean(conf_voted))
            min_conf = float(np.min(conf_voted))
        else:
            mean_conf = 0.0
            min_conf = 0.0

        low_conf_count = int(
            np.sum(voted_mask & (region_confidence < CONFIDENCE_THRESHOLD))
        )

        print(f"[v6 推理] {n_voted}/{num_regions} 区域获得投票, Lines={N_l}")
        print(
            f"  最终颜色支持度(非正确率): mean={mean_conf:.3f}, min={min_conf:.3f}, "
            f"<{CONFIDENCE_THRESHOLD}={low_conf_count}个, "
            f"无投票={len(no_vote_regions)}个"
        )

        # 诊断: 分位数分布, 辅助判断是"一致性虚高"还是"证据权重误伤"
        if n_voted > 0:
            def _pct(a, nd=3):
                q = np.percentile(a, [10, 25, 50, 75, 90])
                return np.round(q, nd).tolist()

            print(
                f"  BP得分(非概率) p10..90={_pct(consistency_all[voted_mask])}, "
                f"vote_weight p10..90={_pct(region_weight[voted_mask], 1)}, "
                f"决策支持度 p10..90={_pct(conf_voted)}, "
                f"线间分歧 p10..90={_pct(region_disagreement[voted_mask])}"
            )
            print(f"  联合MAP修正={_mrf_diagnostic['joint_map_changes']}个Region; "
                  f"MRF能量 {_mrf_diagnostic['energy_before']:.3f} → "
                  f"{_mrf_diagnostic['energy_after']:.3f}")

        result_img = np.zeros((h_img, w_img, 4), dtype=np.uint8)

        for r_id in range(num_regions):
            mask = markers == r_id + 1
            if not voted_mask[r_id]:
                # 无投票区域保持透明
                continue
            pred_color_id = int(region_preds[r_id])
            if pred_color_id in color_palette:
                color = np.asarray(color_palette[pred_color_id], dtype=np.float64)
                result_img[mask] = np.clip(color, 0, 255).astype(np.uint8)

        line_mask = l_rgba[..., 3] > 0
        result_img[line_mask] = l_rgba[line_mask]

        line_comp_id_np = graph["line_comp_id"].cpu().numpy()
        num_line_comps = int(graph["num_line_comps"])

        if fill_lineart and num_line_comps > 0:
            comp_votes = np.zeros(
                (num_line_comps, model.num_classes),
                dtype=np.float64,
            )

            skel_comp_map = np.full(
                (h_img, w_img),
                -1,
                dtype=np.int64,
            )

            for l_idx in range(N_l):
                if not is_color_line[l_idx].item():
                    continue

                g = int(line_comp_id_np[l_idx])
                if g < 0:
                    continue

                weight = float(graph["line_info"][l_idx]["pixel_count"])
                pred_line = int(pred_line_hard[l_idx])

                if 0 <= pred_line < model.num_classes:
                    comp_votes[g, pred_line] += weight

                bfs_px, bfs_py = graph["line_info"][l_idx]["bfs_order"]
                skel_comp_map[bfs_py, bfs_px] = g

            comp_pred = np.argmax(comp_votes, axis=1)

            line_rgb_array = np.array(list(line_palette.values()), dtype=np.float32)
            line_id_array = np.array(list(line_palette.keys()), dtype=np.int64)
            main_lc = int(graph["main_line_color"])

            pixel_line_color = np.full(
                (h_img, w_img),
                -1,
                dtype=np.int64,
            )

            nw_ys, nw_xs = np.where(line_mask)

            if len(nw_ys) > 0 and line_rgb_array.size > 0:
                nw_colors = l_rgba[nw_ys, nw_xs].astype(np.float32)
                d = np.sum(
                    (
                        nw_colors[:, None, :] - line_rgb_array[None, :, :]
                    ) ** 2,
                    axis=2,
                )
                pixel_line_color[nw_ys, nw_xs] = line_id_array[np.argmin(d, axis=1)]

            filled_id_map = np.full(
                (h_img, w_img),
                -1,
                dtype=np.int64,
            )
            filled_colors = []

            for C in np.unique(pixel_line_color):
                C = int(C)

                if C < 0 or C == main_lc:
                    continue

                lbl = sk_label(pixel_line_color == C, connectivity=2)

                for k in range(1, int(lbl.max()) + 1):
                    comp_pix = lbl == k
                    g_vals = skel_comp_map[comp_pix]
                    g_vals = g_vals[g_vals >= 0]

                    if g_vals.size == 0:
                        continue

                    vals, cnts = np.unique(g_vals, return_counts=True)
                    g = int(vals[np.argmax(cnts)])
                    cid = int(comp_pred[g])

                    if cid in color_palette:
                        result_img[comp_pix] = np.clip(
                            np.asarray(color_palette[cid], dtype=np.float64),
                            0,
                            255,
                        ).astype(np.uint8)

                        filled_id_map[comp_pix] = len(filled_colors)
                        filled_colors.append(
                            np.asarray(color_palette[cid], dtype=np.float64)
                        )

            if filled_colors:
                bg = (filled_id_map < 0).astype(np.float64)
                dist, inds = distance_transform_edt(
                    bg,
                    return_indices=True,
                )
                fc_colors = np.array(filled_colors, dtype=np.float64)

                for C in np.unique(pixel_line_color):
                    C = int(C)

                    if C < 0 or C == main_lc:
                        continue

                    lbl = sk_label(pixel_line_color == C, connectivity=2)

                    for k in range(1, int(lbl.max()) + 1):
                        comp_pix = lbl == k
                        g_vals = skel_comp_map[comp_pix]
                        g_vals = g_vals[g_vals >= 0]

                        if g_vals.size > 0:
                            continue

                        d_comp = dist[comp_pix]
                        bi = int(np.argmin(d_comp))
                        ny = inds[0][comp_pix][bi]
                        nx = inds[1][comp_pix][bi]
                        best = int(filled_id_map[ny, nx])

                        if best >= 0:
                            result_img[comp_pix] = np.clip(
                                fc_colors[best],
                                0,
                                255,
                            ).astype(np.uint8)

            unfilled_mask = line_mask & (filled_id_map < 0)

            if np.any(unfilled_mask):
                non_line_mask = ~line_mask

                if np.any(non_line_mask):
                    dt_nl_result = distance_transform_edt(
                        ~non_line_mask,
                        return_indices=True,
                    )
                    inds_nl = dt_nl_result[1]

                    for C in np.unique(pixel_line_color):
                        C = int(C)

                        if C < 0 or C == main_lc:
                            continue

                        lbl = sk_label(
                            (pixel_line_color == C) & unfilled_mask,
                            connectivity=2,
                        )

                        for k in range(1, int(lbl.max()) + 1):
                            comp_pix = lbl == k
                            near_ys = inds_nl[0][comp_pix]
                            near_xs = inds_nl[1][comp_pix]
                            near_colors = result_img[near_ys, near_xs]

                            non_transparent = near_colors[..., 3] > 0
                            if not np.any(non_transparent):
                                continue

                            near_colors = near_colors[non_transparent]
                            color_tuples = [tuple(c) for c in near_colors]
                            dominant = Counter(color_tuples).most_common(1)[0][0]
                            result_img[comp_pix] = np.array(
                                dominant,
                                dtype=np.uint8,
                            )

        conf_img = np.full((h_img, w_img, 3), 255, dtype=np.uint8)

        for r_id in range(num_regions):
            mask = markers == r_id + 1
            conf = float(region_confidence[r_id])

            if conf > 0:
                # 渐变热力图: conf∈(0,1] 沿 红→黄→绿 连续插值 (取代旧固定阈值三段)
                t = min(max(conf, 0.0), 1.0)
                c = np.array(
                    [np.interp(t, CONF_GRAD_T, CONF_GRAD_RGB[:, k]) for k in range(3)],
                    dtype=np.uint8,
                )
                conf_img[mask] = c
            else:
                conf_img[mask] = [180, 180, 180]

        conf_img[skeleton_bin] = [0, 0, 0]

        disagreement_img = np.full((h_img, w_img, 3), 255, dtype=np.uint8)
        for r_id in range(num_regions):
            mask = markers == r_id + 1
            if not voted_mask[r_id]:
                disagreement_img[mask] = (180, 180, 180)
            else:
                # 低分歧=浅色，高分歧=红色；与支持度单独展示。
                d = float(np.clip(region_disagreement[r_id], 0.0, 1.0))
                disagreement_img[mask] = (255, int(255 * (1.0 - d)), int(255 * (1.0 - d)))
        disagreement_img[skeleton_bin] = (0, 0, 0)

        os.makedirs("Debug", exist_ok=True)

        pred_target_vis = np.full(
            (h_img, w_img, 3),
            255,
            dtype=np.uint8,
        )

        for r_id in range(num_regions):
            mask = markers == r_id + 1
            pred_color_id = int(region_preds[r_id])

            if pred_color_id in color_palette:
                pred_target_vis[mask] = np.asarray(color_palette[pred_color_id], dtype=np.uint8)[:3]

        # 调试图输出路径: 优先写到调用方指定的 output 目录, 与彩色结果图同名对应;
        #   未指定时回退旧的 Debug/ 固定路径 (多图会互相覆盖, 仅作临时调试用)。
        if out_dir is not None and out_stem is not None:
            os.makedirs(out_dir, exist_ok=True)
            _tgt_path = os.path.join(out_dir, f"{out_stem}_target.png")
            _conf_path = os.path.join(out_dir, f"{out_stem}_confidence.png")
            _disagreement_path = os.path.join(out_dir, f"{out_stem}_disagreement.png")
            _diag_path = os.path.join(out_dir, f"{out_stem}_confidence_diagnostics.json")
        else:
            _tgt_path = "Debug/test_v6_3_target.png"
            _conf_path = "Debug/test_v6_3_confidence.png"
            _disagreement_path = "Debug/test_v6_3_disagreement.png"
            _diag_path = "Debug/test_v6_3_confidence_diagnostics.json"

        cv2.imencode(
            ".png",
            cv2.cvtColor(pred_target_vis, cv2.COLOR_RGB2BGR),
        )[1].tofile(_tgt_path)

        cv2.imencode(
            ".png",
            cv2.cvtColor(conf_img, cv2.COLOR_RGB2BGR),
        )[1].tofile(_conf_path)
        cv2.imencode(
            ".png",
            cv2.cvtColor(disagreement_img, cv2.COLOR_RGB2BGR),
        )[1].tofile(_disagreement_path)
        # 方向归属风险是独立的几何诊断：不把未校准的几何分数混成“正确概率”。
        _geometry_scores = graph['line_plus_region_a_confidence'].cpu().numpy()
        _geom_risk_sum = np.zeros(num_regions, dtype=np.float64)
        _geom_risk_count = np.zeros(num_regions, dtype=np.float64)
        for _li in range(N_l):
            _a = int(_ra_np[_li]); _b = int(_rb_np[_li])
            if _a == _b:  # 同一 Region 两侧不需要方向判别
                continue
            _unc = float(np.clip(1.0 - _geometry_scores[_li], 0.0, 1.0))
            _geom_risk_sum[_a] += _unc
            _geom_risk_sum[_b] += _unc
            _geom_risk_count[_a] += 1
            _geom_risk_count[_b] += 1
        _geometry_risk = np.divide(_geom_risk_sum, np.maximum(_geom_risk_count, 1.0))
        # --- 逐图提取 Region 的 GT 颜色 / 面积 / 是否邻接未见 Triple (供置信度验收) ---
        #   口径必须与 extract_ll_graph 构造 region_labels **完全一致**, 否则这里的"区域对错"
        #   会和 Graph RegionAcc 给出两套互相矛盾的标签 (实测小区域几乎全被判错):
        #     1) 众数只在"非线稿像素"上统计 —— 粗笔触边缘的线色会污染众数;
        #     2) 众数颜色必须与调色板**精确相等** —— 抗锯齿/混色像素不能被最近色硬归入某类。
        #   gt_color       : 精确匹配的调色板 id; -1 = 无法判定 (排序类指标应剔除)。
        #   gt_color_label : 未命中时回落 0, 与 region_labels 完全相同, 用于与 Graph RegionAcc 对账。
        #   注意: 这里用的是**图提取 Region** (markers)，与 evaluate_coloring_images 的图像级
        #   连通分区不是同一套; 两者不能按 id 配对 (分区指纹不同)。
        _gt_region = [-1] * int(num_regions)
        _gt_label = [0] * int(num_regions)
        _region_area = [int((markers == (_r + 1)).sum()) for _r in range(int(num_regions))]
        if gt_color_path is not None and os.path.exists(gt_color_path):
            try:
                _gt_rgba = process_alpha(cv2.imdecode(
                    np.fromfile(gt_color_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED))
                if _gt_rgba.shape[:2] == markers.shape:
                    _ink = l_rgba[..., 3] > 0
                    _color_to_id = {tuple(int(x) for x in v[:4]): int(k)
                                    for k, v in color_palette.items()}
                    for _r in range(int(num_regions)):
                        _m = (markers == (_r + 1)) & (~_ink)
                        if not _m.any():
                            continue
                        _px = _gt_rgba[_m].astype(np.int64)
                        _px = _px[_px[:, 3] > 0]
                        if _px.size == 0:
                            continue
                        _vals, _cnt = np.unique(_px, axis=0, return_counts=True)
                        _cand = _vals[_cnt == _cnt.max()]
                        if _cand.shape[0] == 1:
                            _mc = _cand[0]
                        else:
                            # 平局取光栅序最早出现的颜色 (与 region_labels 的 tie-break 一致)
                            _firsts = [int(np.argmax((_px == _c).all(axis=1))) for _c in _cand]
                            _mc = _cand[int(np.argmin(_firsts))]
                        _key = tuple(int(x) for x in _mc)
                        if _key in _color_to_id:
                            _gt_region[_r] = _color_to_id[_key]
                            _gt_label[_r] = _color_to_id[_key]
                        else:
                            _gt_region[_r] = -1
                            _gt_label[_r] = 0
            except Exception as _e:
                print(f'[诊断] GT 颜色提取失败: {_e}', flush=True)
        # 邻接至少一条"完整 Triple 未见"的 Line 的 Region (基于**预测**组合, 见文档 S0 审查 §1:
        #   这只是 predicted_unseen_*, 不能当作"真实未见组合的泛化评价")
        _adj_unseen = [False] * int(num_regions)
        if _new_triples:
            for _li, _un in enumerate(_new_triples):
                if not _un:
                    continue
                _a = int(_ra_np[_li]); _b = int(_rb_np[_li])
                if 0 <= _a < num_regions:
                    _adj_unseen[_a] = True
                if 0 <= _b < num_regions:
                    _adj_unseen[_b] = True

        # --- GT 真值未见分组 (S0 审查 §1): 与 predicted_* 严格分开, 绝不混用 -------------
        #   训练期 Triple 表构造公式 (build_dataset_bundle → triple_set):
        #       a = line_label_a, b = line_label_b (val 图**真实 region 标签**)
        #       c = line_color_label if is_color_line else main_line_color
        #       truth = (min(a,b), max(a,b), c)
        #   这里直接复用 val 图 (带 GT 构建的那份 graph) 的标签数组, 而不是自己重算,
        #   以免类别空间或 tie-break 与训练期漂移。只有当两份 markers 指纹完全相同
        #   (同一张图、同一套分区) 才允许配对, 否则标记 unavailable 并跳过。
        _markers_sha = hashlib.sha256(np.ascontiguousarray(markers).tobytes()).hexdigest()
        _gt_status = 'unavailable'
        _gt_line_valid = [False] * int(N_l)
        _gt_triple_flags = [None] * int(N_l)
        _gt_pair_flags = [None] * int(N_l)
        _gt_region_label = None
        _adj_gt_triple = [False] * int(num_regions)
        _adj_gt_pair = [False] * int(num_regions)
        _gt_formula_mismatch = None
        _partition_basis = None
        _gt_region_label_mismatch = None
        _gt_line_triple = None
        if gt_graph is not None:
            try:
                # 分区指纹核对: 优先用 markers 逐像素 sha; 若 val 图在缓存里把 markers 置空
                # (省内存), 则退化为"输入线稿文件字节完全相同 + Region/Line 数相同"
                # —— 图提取对同一份线稿是确定性的, 因此两者等价。
                _gm = gt_graph.get('markers')
                _partition_ok = False
                if _gm is not None and np.shape(_gm) == np.shape(markers):
                    _partition_ok = (hashlib.sha256(np.ascontiguousarray(_gm).tobytes()).hexdigest()
                                     == _markers_sha)
                    _partition_basis = 'markers_sha256'
                else:
                    _src_ok = False
                    if gt_line_path is not None and os.path.exists(gt_line_path):
                        try:
                            _src_ok = (hashlib.sha256(open(line_path, 'rb').read()).hexdigest()
                                       == hashlib.sha256(open(gt_line_path, 'rb').read()).hexdigest())
                        except Exception:
                            _src_ok = False
                    _partition_ok = (_src_ok
                                     and int(gt_graph.get('num_regions', -1)) == int(num_regions)
                                     and int(gt_graph.get('N_l', -1)) == int(N_l))
                    _partition_basis = 'lineart_sha256+shape'
                if _partition_ok:
                    _la = np.asarray(gt_graph['line_label_a'].cpu()
                                     if torch.is_tensor(gt_graph['line_label_a'])
                                     else gt_graph['line_label_a']).astype(int)
                    _lb = np.asarray(gt_graph['line_label_b'].cpu()
                                     if torch.is_tensor(gt_graph['line_label_b'])
                                     else gt_graph['line_label_b']).astype(int)
                    _lc = np.asarray(gt_graph['line_color_label'].cpu()
                                     if torch.is_tensor(gt_graph['line_color_label'])
                                     else gt_graph['line_color_label']).astype(int)
                    _iscl = np.asarray(gt_graph['is_color_line'].cpu()
                                       if torch.is_tensor(gt_graph['is_color_line'])
                                       else gt_graph['is_color_line']).astype(bool)
                    _mainc = int(gt_graph['main_line_color'])
                    if len(_la) == int(N_l):
                        _c = np.where(_iscl, _lc, _mainc)
                        _triple_flags = [(min(int(_la[i]), int(_lb[i])),
                                          max(int(_la[i]), int(_lb[i])),
                                          int(_c[i])) not in _seen_triples
                                         for i in range(int(N_l))]
                        _pair_flags = [(min(int(_la[i]), int(_lb[i])),
                                        max(int(_la[i]), int(_lb[i]))) not in _seen_pairs
                                       for i in range(int(N_l))]
                        # GT 有效性: 两侧 Region 都要"可判定" (gt_color != -1),
                        #   不能把未知回落 0 当作合法的透明色。
                        _gt_line_valid = [
                            (0 <= int(_la[i]) < num_regions and 0 <= int(_lb[i]) < num_regions
                             and _gt_region[int(_la[i])] >= 0 and _gt_region[int(_lb[i])] >= 0)
                            for i in range(int(N_l))]
                        _gt_triple_flags = _triple_flags
                        _gt_pair_flags = _pair_flags
                        _gt_line_triple = [[int(min(_la[i], _lb[i])),
                                            int(max(_la[i], _lb[i])),
                                            int(_c[i])] for i in range(int(N_l))]
                        _y = gt_graph.get('y')
                        if _y is not None:
                            _gt_region_label = np.asarray(
                                _y.cpu() if torch.is_tensor(_y) else _y).astype(int).tolist()
                            # 逐 Region 向量核对: 诊断复算的 gt_color_label 必须与官方
                            # region_labels(=graph['y']) 完全相同 (不能只看整体 RegionAcc 差值为 0)
                            if len(_gt_region_label) == len(_gt_label):
                                _gt_region_label_mismatch = int(sum(
                                    1 for _o, _d in zip(_gt_region_label, _gt_label)
                                    if int(_o) != int(_d)))
                        for _li in range(int(N_l)):
                            if not _gt_line_valid[_li]:
                                continue
                            _a = int(_ra_np[_li]); _b = int(_rb_np[_li])
                            _f = _gt_triple_flags[_li]
                            _p = _gt_pair_flags[_li]
                            if 0 <= _a < num_regions:
                                _adj_gt_triple[_a] = _adj_gt_triple[_a] or bool(_f)
                                _adj_gt_pair[_a] = _adj_gt_pair[_a] or bool(_p)
                            if 0 <= _b < num_regions:
                                _adj_gt_triple[_b] = _adj_gt_triple[_b] or bool(_f)
                                _adj_gt_pair[_b] = _adj_gt_pair[_b] or bool(_p)
                        # 与 val 图自带的 triple_label (-1 = 训练表中不存在) 交叉核对公式
                        _tl = gt_graph.get('triple_label')
                        if _tl is not None:
                            _tl = np.asarray(_tl.cpu() if torch.is_tensor(_tl) else _tl).astype(int)
                            if len(_tl) == int(N_l):
                                _gt_formula_mismatch = int(sum(
                                    bool((_tl[i] < 0) != bool(_gt_triple_flags[i]))
                                    for i in range(int(N_l))))
                        _gt_status = 'ok'
            except Exception as _e:
                print(f'[诊断] GT 未见分组计算失败: {_e}', flush=True)

        _n_valid = int(sum(_gt_line_valid))
        _gt_triple_frac = (float(sum(bool(x) for x, v in zip(_gt_triple_flags, _gt_line_valid) if v))
                           / _n_valid) if _n_valid else None
        _gt_pair_frac = (float(sum(bool(x) for x, v in zip(_gt_pair_flags, _gt_line_valid) if v))
                         / _n_valid) if _n_valid else None

        _conf_diagnostics = {
            'scope': '图提取 Region（0-based）；与本次输出线稿的 graph.markers 对应',
            'confidence_meaning': '最终颜色的周边联合边际加权支持度；未经过正确率校准',
            'disagreement_meaning': '周边 Line 颜色分布的加权 JS 分歧，范围 [0,1]',
            'num_regions': int(num_regions), 'num_lines': int(N_l),
            'final_color': region_preds.astype(int).tolist(),
            'unary_color': region_unary_probs.argmax(axis=1).astype(int).tolist(),
            'unary_top2': [{str(int(k)): float(v) for k, v in
                            zip(np.argsort(p)[::-1][:2], p[np.argsort(p)[::-1][:2]])}
                           for p in region_unary_probs],
            'decision_support': region_confidence.astype(float).tolist(),
            'line_disagreement': region_disagreement.astype(float).tolist(),
            'incident_line_sides': region_incidents.astype(int).tolist(),
            'voted_mask': voted_mask.astype(bool).tolist(),
            'bp_independent_color': region_bp_independent.astype(int).tolist(),
            'joint_map_changes': int(_mrf_diagnostic['joint_map_changes']),
            'energy_before': float(_mrf_diagnostic['energy_before']),
            'energy_after': float(_mrf_diagnostic['energy_after']),
            'unseen_pair_penalty': float(PAIRWISE_UNSEEN_PENALTY),
            'pairwise_mode': str(PAIRWISE_POT),
            # --- 预测口径 (模型自己的"未见"组合; 只用于观察解码行为, 不能当泛化评价) ---
            'predicted_unseen_region_pair_line_ids': [i for i, unseen in enumerate(_new_pairs)
                                                      if unseen],
            'predicted_unseen_full_triple_line_ids': [i for i, unseen in enumerate(_new_triples)
                                                      if unseen],
            'predicted_unseen_region_pair_fraction': (float(np.mean(_new_pairs))
                                                      if N_l else 0.0),
            'predicted_unseen_full_triple_fraction': (float(np.mean(_new_triples))
                                                      if N_l else 0.0),
            'predicted_unseen_full_triple_line_flags': [bool(x) for x in _new_triples],
            'predicted_unseen_pair_line_flags': [bool(x) for x in _new_pairs],
            'adjacent_predicted_unseen_triple': _adj_unseen,
            # --- GT 口径 (真值标签 + 训练期冻结 Triple 表; 泛化评价只用这一组) ---
            'gt_unseen_status': _gt_status,
            'gt_partition_basis': _partition_basis,
            'gt_region_label_vs_diag_mismatch': _gt_region_label_mismatch,
            'gt_line_valid_flags': [bool(x) for x in _gt_line_valid],
            'gt_unseen_full_triple_line_flags': ([bool(x) if x is not None else None
                                                  for x in _gt_triple_flags]),
            'gt_unseen_pair_line_flags': ([bool(x) if x is not None else None
                                           for x in _gt_pair_flags]),
            'gt_unseen_full_triple_fraction': _gt_triple_frac,
            'gt_unseen_pair_fraction': _gt_pair_frac,
            'gt_valid_line_count': _n_valid,
            'gt_unseen_full_triple_line_count': (int(sum(bool(x) for x in _gt_triple_flags))
                                                 if _gt_status == 'ok' else None),
            'gt_unseen_pair_line_count': (int(sum(bool(x) for x in _gt_pair_flags))
                                          if _gt_status == 'ok' else None),
            'gt_unseen_formula_vs_triple_label_mismatch': _gt_formula_mismatch,
            'gt_line_triple': _gt_line_triple,
            'adjacent_gt_unseen_triple': _adj_gt_triple,
            'adjacent_gt_unseen_pair': _adj_gt_pair,
            'region_label_official': _gt_region_label,
            'line_orientation_geometry_confidence': _geometry_scores.astype(float).tolist(),
            'region_orientation_geometry_risk': _geometry_risk.astype(float).tolist(),
            'region_orientation_line_count': _geom_risk_count.astype(int).tolist(),
            'markers_sha256': _markers_sha,
            'gt_color': _gt_region,
            'gt_color_label': _gt_label,
            'line_final_color': pred_line_hard.astype(int).tolist(),
            # 逐 Line 的两侧 Region 索引: 用于离线把 (Region 预测色对 + 线色) 组成预测三元组
            'line_region_a': [int(x) for x in _ra_np],
            'line_region_b': [int(x) for x in _rb_np],
            'region_area': _region_area,
        }
        with open(_diag_path, 'w', encoding='utf-8') as _f:
            json.dump(_conf_diagnostics, _f, indent=2, ensure_ascii=False)

        # --- 逐区域和逐像素对错图：统一委托 gnn_common 独立 RGBA 评价 ---
        if (gt_color_path is not None and os.path.exists(gt_color_path)
                and out_dir is not None and out_stem is not None):
            try:
                _metrics = evaluate_coloring_images(
                    line_path, result_img, gt_color_path, color_palette=color_palette,
                    output_prefix=os.path.join(out_dir, out_stem))
                _f = lambda v, ok: f'{v:.4f}' if ok else 'N/A'
                print(f"[成图对错] {out_stem}: "
                      f"成图 RegionAcc {_f(_metrics['region_acc'],_metrics['region_metric_defined'])} "
                      f"({_metrics['region_correct']}/{_metrics['region_total']}); "
                      f"PixelAcc(调色板) {_f(_metrics['pixel_acc'],_metrics['pixel_metric_defined'])}; "
                      f"Pixel mIoU {_f(_metrics['pixel_miou'],_metrics['pixel_metric_defined'])}; "
                      f"严格RGBA PixelAcc {_f(_metrics['pixel_acc_exact_rgba'],_metrics['total_fill_pixels']>0)}; "
                      f"像素覆盖率 {_metrics['pixel_coverage']:.3f}", flush=True)
            except Exception as _e:
                print(f"[成图对错] 生成失败: {_e}", flush=True)

    model.train(was_training)   # 恢复调用前的训练模式 (MC 分支临时开了 train)

    return (
        result_img,
        prob_a_lines,
        prob_b_lines,
        pred_a_hard,
        pred_b_hard,
        pred_line_hard,
        top3_triples, top4_pairs, top4_side_a, top4_side_b,
    )

# ==================== 6. L-L 图可视化 ====================

def visualize_ll_graph(lineart_path: str, colorized_path: Optional[str] = None,
                       line_palette=None, color_palette=None,
                       output_html: str = "graph_vis_v6.html",
                       pred_a=None, pred_b=None,
                       logits_a=None, logits_b=None,
                       pred_line=None,
                       prob_a_marginal=None, prob_b_marginal=None,
                       top3_triples=None,
                       top4_pairs=None, top4_side_a=None, top4_side_b=None,
                       confidence_path=None, error_path=None,
                       metrics_path=None, train_previews=None,
                       disagreement_path=None, confidence_diagnostics_path=None,
                       pixel_error_path=None):
    """
    增强版：接受几何 +n/-n 的联合后验边际 prob_a_marginal/prob_b_marginal
    (shape: N_l, num_classes)，HTML 转为面积排序的 A/B 侧再显示。
    若 colorized_path 是最终预测图，则这里构造出的 graph 标签来自预测图，非 GT。
    top3_triples: (N_l,) list of [{id, prob}], 每条线前 3 个 triple 及概率。
    confidence_path: 最终颜色支持度热图（不是正确率），disagreement_path: Line 分歧热图。
    confidence_diagnostics_path: 同一次推理的 MRF/Unary/决策支持度 JSON。
    error_path: 可嵌入成图错误背景。
    metrics_path: 若是本次生成的 *_metrics.json，则在 HTML 面板展示成图统计摘要。
    train_previews: [(label, path), ...] 学习用图 (1-shot 训练样本), 在 HTML 右下角
       浮窗展示, 支持放大 / 隐藏; 为空则不显示该浮窗。
    Graph 的节点数与成图 RegionAcc 的分区口径可能不同，HTML 会明确区分。
    """
    import base64

    l_raw = cv2.imdecode(np.fromfile(lineart_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if l_raw is None:
        print(f"[可视化] 无法读取: {lineart_path}")
        return
    l_rgb = process_alpha(l_raw)

    if colorized_path and os.path.exists(colorized_path):
        c_raw = cv2.imdecode(np.fromfile(colorized_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        c_rgb = process_alpha(c_raw) if c_raw is not None else np.zeros_like(l_rgb)
    else:
        c_rgb = np.zeros_like(l_rgb)

    lp = line_palette or {}
    cp = color_palette or {}

    graph = extract_ll_graph(l_rgb, c_rgb, lp, cp, debug_prefix="vis")
    if graph is None:
        print("[可视化] 图提取失败")
        return

    skeleton_rgb = graph["skeleton_rgb"]
    markers = graph["markers"]
    num_regions = graph["num_regions"]
    keypoints = graph["keypoints"]
    h, w = markers.shape
    N_l = graph["N_l"]

    skeleton_bin = skeleton_rgb[..., 3] > 0

    def _draw_keypoints(img, tint, offset):
        """在 img 上按骨架交叉数给关键点着色, 并用骨架像素的淡化色提亮背景。"""
        for x, y in keypoints:
            cn = _ring_crossing_number(skeleton_bin, x, y)
            if cn == 1:
                color = (50, 100, 220)   # 端点: 蓝
            elif cn == 4:
                color = (0, 0, 255)      # 交叉点 (n=4): 红
            elif cn >= 3:
                color = (50, 200, 80)    # T 型等交叉 (n=3 等): 绿
            else:
                color = (230, 120, 30)   # 角点 (n=2): 橙
            cv2.circle(img, (x, y), 3, color, -1)
        img[skeleton_bin] = np.clip(
            skeleton_rgb[skeleton_bin][:, :3].astype(np.float32) * tint + offset, 0, 255
        ).astype(np.uint8)

    # --- 骨架背景图 (原始尺寸) ---
    skel_img = np.full((h, w, 3), 255, dtype=np.uint8)
    _draw_keypoints(skel_img, 0.6, 40)
    _, buf = cv2.imencode('.png', cv2.cvtColor(skel_img, cv2.COLOR_RGB2BGR))
    bg_skel_b64 = base64.b64encode(buf).decode('ascii')

    # --- Region 着色图 ---
    region_img = np.full((h, w, 3), 255, dtype=np.uint8)
    colored_mask = np.zeros((h, w), dtype=bool)   # 实际被着色 (label 在 cp 且非透明) 的 region 像素
    for r_id in range(1, num_regions + 1):
        mask = (markers == r_id)
        if not np.any(mask):
            continue
        label_id = graph["y"][r_id - 1]
        label_int = label_id.item() if hasattr(label_id, 'item') else int(label_id)
        if cp and label_int in cp:
            rgba = cp[label_int]
            # 透明背景 (alpha=0, 如 index 0 的 (0,0,0,0)) 不着色 → 最终透明, 露出页面背景
            if len(rgba) > 3 and rgba[3] == 0:
                continue
            color = np.array(rgba, dtype=np.float32)[:3]
            # 直接显示原始颜色 (不混白), 避免上色结果泛白
            region_img[mask] = color.astype(np.uint8)
            colored_mask |= mask
        # label 不在 cp 的 region 保持白色, 但不标 colored → 最终透明
    _draw_keypoints(region_img, 0.7, 30)
    # 去蒙版: 仅"实际着色"的 region 像素 + 骨架像素不透明, 其余(背景 region)透明。
    alpha = np.where(colored_mask | skeleton_bin, 255, 0).astype(np.uint8)
    region_rgba = np.dstack([region_img, alpha])
    _, buf2 = cv2.imencode('.png', cv2.cvtColor(region_rgba, cv2.COLOR_RGBA2BGRA))
    bg_region_b64 = base64.b64encode(buf2).decode('ascii')

    # --- 准备颜色映射 ---
    color_ids = sorted(cp.keys()) if cp else []
    color_hex_map = {cid: f"#{r:02x}{gg:02x}{b:02x}" for cid, (r, gg, b, _a) in cp.items()}

    # --- 辅助: 获取边际概率 ---
    # 推理返回的两个边际是几何 +n / -n，不是面积排序的 Region A/B！
    # 必须利用 line_plus_region_a 映射，HTML 才能与 MRF 实际 unary 路由一致。
    # 此处仅展示边际概率；下面的诊断聚合不是 MRF 的 log-unary / pairwise / BP 最终后验。
    if prob_a_marginal is not None and prob_b_marginal is not None:
        plus = np.asarray(prob_a_marginal, dtype=np.float64)  # (N_l,C) +n
        minus = np.asarray(prob_b_marginal, dtype=np.float64) # (N_l,C) -n
        if plus.shape != minus.shape or plus.ndim != 2 or plus.shape[0] != N_l:
            raise ValueError(f'HTML 侧向概率维度错误: +n {plus.shape}, -n {minus.shape}; N_l={N_l}')
        plus_is_a = _to_np(graph['line_plus_region_a']).astype(bool).reshape(-1)
        if plus_is_a.shape[0] != N_l:
            raise ValueError('HTML line_plus_region_a 长度与 Line 数不一致')
        prob_a = np.where(plus_is_a[:, None], plus, minus)
        prob_b = np.where(plus_is_a[:, None], minus, plus)
    else:
        if logits_a is not None or logits_b is not None:
            warnings.warn(
                'visualize_ll_graph 只接受经过 Triple×Side 融合的概率边际；'
                '仅有 side logits 无法展示 MRF 输入边际，已跳过投票展示。',
                stacklevel=2,
            )
        prob_a = prob_b = None

    # --- 获取 line_region 信息 ---
    lr_a = _to_np(graph["line_region_a"])
    lr_b = _to_np(graph["line_region_b"])
    line_pixel_counts = _to_np(graph["line_pixel_count"])

    # --- 仅用于诊断的 Region 边界边际均值（不是 MRF 最终信念分布）。---
    region_vote_probs = None
    if prob_a is not None and prob_b is not None:
        num_classes = prob_a.shape[1]
        region_votes = np.zeros((num_regions, num_classes), dtype=np.float64)
        region_weights = np.zeros(num_regions, dtype=np.float64)
        for l_idx in range(N_l):
            r_a = int(lr_a[l_idx])
            r_b = int(lr_b[l_idx])
            vote_w = float(line_pixel_counts[l_idx])  # 注意: 不能用 w, 会覆盖图像宽度变量
            if 0 <= r_a < num_regions:
                region_votes[r_a] += prob_a[l_idx] * vote_w
                region_weights[r_a] += vote_w
            if 0 <= r_b < num_regions:
                region_votes[r_b] += prob_b[l_idx] * vote_w
                region_weights[r_b] += vote_w
        # 每条线按像素长度加权的算术均值；MRF 实际使用置信度加权的 log unary + Pairwise + BP。
        region_vote_probs = region_votes / (region_weights[:, None] + 1e-8)
        # 转换为列表结构 {label, prob}
        region_vote_list = []
        for r_id in range(num_regions):
            probs = region_vote_probs[r_id]
            # 只保留概率 > 0.01 的类别
            entries = [{"label": int(c), "prob": float(probs[c])} for c in range(num_classes) if probs[c] > 0.01]
            entries.sort(key=lambda x: -x["prob"])
            region_vote_list.append(entries)
    else:
        region_vote_list = [[] for _ in range(num_regions)]

    # 只有与本次线稿 Region 数量一致的诊断结果才能接入 HTML，避免旧缓存/旧图片错位。
    _confidence_details = None
    if confidence_diagnostics_path and os.path.isfile(confidence_diagnostics_path):
        try:
            with open(confidence_diagnostics_path, encoding='utf-8') as _f:
                _candidate_diag = json.load(_f)
            if (int(_candidate_diag.get('num_regions', -1)) == num_regions
                    and int(_candidate_diag.get('num_lines', -1)) == N_l
                    and _candidate_diag.get('markers_sha256') ==
                    hashlib.sha256(np.ascontiguousarray(markers).tobytes()).hexdigest()):
                _confidence_details = _candidate_diag
            else:
                print('[可视化] MRF 诊断文件的 Region/Line 数量与本图不同，跳过')
        except (OSError, ValueError, TypeError) as _e:
            print(f'[可视化] 无法读取 MRF 诊断文件: {_e}')

    # --- 构建 Line 节点 ---
    X_l = _to_np(graph["X_l"])
    line_nodes = []
    line_info = graph["line_info"]
    ll_a = _to_np(graph["line_label_a"])
    ll_b = _to_np(graph["line_label_b"])
    # 笔画分组信息 (池化时使用): 展示每条线属于哪个 corner/cont 组。
    stroke_id = _to_np(graph.get("stroke_id")) if graph.get("stroke_id") is not None else None
    cont_stroke_id = _to_np(graph.get("cont_stroke_id")) if graph.get("cont_stroke_id") is not None else None

    for l_idx in range(len(line_info)):
        li = line_info[l_idx]
        mid = li['mid']
        kp1, kp2 = li['kp1'], li['kp2']
        feat = X_l[l_idx].tolist()
        skel_path = li.get('bfs_order', None)
        if skel_path is not None:
            path_px = np.stack([skel_path[0], skel_path[1]], axis=1)
        else:
            path_px = np.array([mid])
        if len(path_px) > 0:
            path_colors = skeleton_rgb[path_px[:, 1], path_px[:, 0]]
            mode_rgb_packed, _ = _mode_rgb_packed(path_colors)
            line_hex = f"#{(mode_rgb_packed >> 16) & 0xFF:02x}{(mode_rgb_packed >> 8) & 0xFF:02x}{mode_rgb_packed & 0xFF:02x}"
        else:
            line_hex = "#000000"

        ra, rb = int(lr_a[l_idx]) + 1, int(lr_b[l_idx]) + 1  # 1-indexed

        # --- 构建投票数据（A/B侧） ---
        prob_a_list = []
        prob_b_list = []
        if prob_a is not None and l_idx < len(prob_a):
            for c in range(prob_a.shape[1]):
                p = prob_a[l_idx, c]
                if p > 0.0:   # 全类保留: 面板按 Top-3 展示并给出"其余合计", 需精确求和
                    prob_a_list.append({"label": int(c), "prob": float(p)})
            prob_a_list.sort(key=lambda x: -x["prob"])
        if prob_b is not None and l_idx < len(prob_b):
            for c in range(prob_b.shape[1]):
                p = prob_b[l_idx, c]
                if p > 0.0:   # 全类保留 (同上)
                    prob_b_list.append({"label": int(c), "prob": float(p)})
            prob_b_list.sort(key=lambda x: -x["prob"])

        # 构建 features 字典
        features = {
            "relative_arc_length": f"{feat[0]:.4f}",
            "straightness": f"{feat[1]:.4f}",
            "frt": f"{feat[2]:.4f}",
            "rel_arc_len": f"{feat[3]:.4f}",
            "color_index": str(li.get('color_index', -1)),
            "pixel_count": str(li['pixel_count']),
            "region_a (大)": f"R{ra}",
            "region_b (小)": f"R{rb}",
            "label_a (显示图)": str(int(ll_a[l_idx])),
            "label_b (显示图)": str(int(ll_b[l_idx])),
        }
        _orient_conf = graph.get('line_plus_region_a_confidence')
        if _orient_conf is not None:
            _orient_val = float(_to_np(_orient_conf)[l_idx])
            features['+n/-n 几何归属可靠度'] = f'{_orient_val:.4f}'
            if _orient_val <= 0.0 and int(lr_a[l_idx]) != int(lr_b[l_idx]):
                features['几何方向提醒'] = '方向证据不足，使用回退规则；Side 方位可能不可靠'
        if _confidence_details is not None:
            _unknown_pair_ids = _confidence_details.get('unseen_region_pair_line_ids', [])
            _unknown_triple_ids = _confidence_details.get('unseen_full_triple_line_ids', [])
            features['最终色对未见于 Triple 表'] = '是' if l_idx in _unknown_pair_ids else '否'
            features['最终完整 Triple 未见于训练'] = '是' if l_idx in _unknown_triple_ids else '否'
        # 笔画分组 (池化时使用)
        if stroke_id is not None and l_idx < len(stroke_id):
            features["corner组"] = f"stroke {int(stroke_id[l_idx])}"
        if cont_stroke_id is not None and l_idx < len(cont_stroke_id):
            features["cont组"] = f"stroke {int(cont_stroke_id[l_idx])}"
        # Triple 头的 pred_a/pred_b 是无序色对 (min/max)，不是面积 A/B 预测；
        # 展示图是最终预测图，不能用其生成的标签冒充 GT、据此计算 a_correct/b_correct。
        if pred_a is not None and pred_b is not None and l_idx < len(pred_a):
            features["Triple 色对 min"] = str(min(int(pred_a[l_idx]), int(pred_b[l_idx])))
            features["Triple 色对 max"] = str(max(int(pred_a[l_idx]), int(pred_b[l_idx])))
            if pred_line is not None:
                features["Triple 线色"] = str(pred_line[l_idx])
        # top-3 triple (预测概率)
        if top3_triples is not None and l_idx < len(top3_triples):
            features["top3_triples"] = top3_triples[l_idx]
        if top4_pairs is not None and l_idx < len(top4_pairs):
            features["top4_pairs"] = top4_pairs[l_idx]
        if top4_side_a is not None and l_idx < len(top4_side_a):
            features["top4_side_a"] = top4_side_a[l_idx]
        if top4_side_b is not None and l_idx < len(top4_side_b):
            features["top4_side_b"] = top4_side_b[l_idx]
        # 添加投票数组（结构化）
        if prob_a_list:
            features["prob_a"] = prob_a_list
        if prob_b_list:
            features["prob_b"] = prob_b_list

        line_nodes.append({
            "id": f"L{l_idx}",
            "pos": {"x": mid[0], "y": mid[1]},
            "endpoints": [list(kp1), list(kp2)],
            "color": line_hex,
            "stroke_id": int(stroke_id[l_idx]) if stroke_id is not None and l_idx < len(stroke_id) else -1,
            "features": features
        })

    # --- 构建 Region 节点（线边际均值 + 本次真实 MRF 诊断） ---
    region_nodes = []
    for r_id in range(1, num_regions + 1):
        ys_r, xs_r = np.where(markers == r_id)
        cx = float(np.mean(xs_r)) if len(xs_r) > 0 else w / 2.0
        cy = float(np.mean(ys_r)) if len(ys_r) > 0 else h / 2.0
        area = len(ys_r)
        radius = max(10, min(40, math.sqrt(area) * 0.3))
        label_id = graph["y"][r_id - 1]
        label_int = label_id.item() if hasattr(label_id, 'item') else int(label_id)
        features = {
            "显示图 label（非 GT）": str(label_int),
            "area_px": str(area),
            "area_norm": f"{area / graph['area_median']:.4f}",
        }
        if _confidence_details is not None:
            _i = r_id - 1
            if _confidence_details['voted_mask'][_i]:
                _fin = int(_confidence_details['final_color'][_i])
                _unc = int(_confidence_details['unary_color'][_i])
                features['MRF 最终颜色（非训练 GT）'] = str(_fin)
                features['Unary 首选颜色（无 Pairwise）'] = str(_unc)
                features['Pairwise 改变颜色'] = '是' if _fin != _unc else '否'
                features['Decision Support（非正确率）'] = f"{float(_confidence_details['decision_support'][_i]):.4f}"
                features['Line 分歧 JS'] = f"{float(_confidence_details['line_disagreement'][_i]):.4f}"
                _risk = float(_confidence_details.get('region_orientation_geometry_risk', [0.0]*num_regions)[_i])
                features['邻线几何方向不确定度'] = f'{_risk:.4f}'
                if _risk > .5:
                    features['几何风险提醒'] = '较多邻线的正反侧判别不稳定；支持度可能存在方向错配'
                features['有效 Line 侧数'] = str(int(_confidence_details['incident_line_sides'][_i]))
                features['Unary Top-2（非校准概率）'] = str(_confidence_details['unary_top2'][_i])
            else:
                features['MRF 状态'] = '无投票（不显示可靠度）'
        # 添加周边线长度加权概率均值（仅用于可视化诊断，不是 MRF 后验）
        if region_vote_list and r_id-1 < len(region_vote_list):
            features["vote_probs"] = region_vote_list[r_id-1]

        # --- 周边线证据: 每条边界线归属的面积 A/B 侧概率（已与 +n/-n 正确映射） ---
        # 用于诊断"小区域周边线都投了什么颜色" (线在 a 侧投 prob_a, b 侧投 prob_b)。
        boundary_lines = []
        for l_idx in range(N_l):
            ra = int(lr_a[l_idx])
            rb = int(lr_b[l_idx])
            side = None
            votes = None
            label_display = None
            if ra == r_id - 1:
                side = "A (大)"
                votes = prob_a[l_idx] if prob_a is not None else None
                label_display = int(ll_a[l_idx])
            elif rb == r_id - 1:
                side = "B (小)"
                votes = prob_b[l_idx] if prob_b is not None else None
                label_display = int(ll_b[l_idx])
            if side is None:
                continue
            vote_list = []
            if votes is not None:
                for c in range(len(votes)):
                    p = float(votes[c])
                    if p > 0.01:
                        vote_list.append({"label": int(c), "prob": p})
                vote_list.sort(key=lambda x: -x["prob"])
            boundary_lines.append({
                "line_id": f"L{l_idx}",
                "side": side,
                "votes": vote_list,
                "label_display": label_display,
                "pixel_count": int(line_pixel_counts[l_idx]),
            })
        if boundary_lines:
            features["boundary_lines"] = boundary_lines

        region_nodes.append({
            "id": f"R{r_id}",
            "pos": {"x": cx, "y": cy},
            "size": radius,
            "area": area,
            "features": features
        })

    # --- L-L 边 (线-线消息传递边, 网络实际使用的边) ---
    ei_ll = _to_np(graph["edge_index_ll"])
    ea_ll = _to_np(graph["edge_attr_ll"])
    er_a = _to_np(graph["edge_region_a"])
    er_b = _to_np(graph["edge_region_b"])

    edges = []
    edge_id = 0
    if ei_ll.size > 0:
        seen = set()
        for i in range(ei_ll.shape[1]):
            s, d = int(ei_ll[0, i]), int(ei_ll[1, i])
            pair = (min(s, d), max(s, d))
            if pair in seen:
                continue
            seen.add(pair)
            feat = ea_ll[i].tolist()
            r_a = int(er_a[i]) + 1
            r_b = int(er_b[i]) + 1
            edges.append({
                "id": f"e{edge_id}", "source": f"L{s}", "target": f"L{d}",
                "edge_type": "Line-Line",
                "features": {
                    "tan_half": f"{feat[0]:.4f}",
                    "len_ratio": f"{feat[1]:.4f}",
                    "region_a": f"R{r_a}", "region_b": f"R{r_b}",
                }
            })
            edge_id += 1

    # --- Line-Region 边 (线 → 两侧 Region 归属, 分类器/解码器实际使用的归属关系) ---
    for l_idx in range(N_l):
        ra = int(lr_a[l_idx])
        rb = int(lr_b[l_idx])
        if 0 <= ra < num_regions:
            edges.append({
                "id": f"e{edge_id}", "source": f"L{l_idx}", "target": f"R{ra + 1}",
                "edge_type": "Line-Region",
                "features": {"side": "A (大)", "label": str(int(ll_a[l_idx]))},
            })
            edge_id += 1
        if 0 <= rb < num_regions and rb != ra:
            edges.append({
                "id": f"e{edge_id}", "source": f"L{l_idx}", "target": f"R{rb + 1}",
                "edge_type": "Line-Region",
                "features": {"side": "B (小)", "label": str(int(ll_b[l_idx]))},
            })
            edge_id += 1

    # --- Region-Region 边 (共享 L-L 边的两侧 region 互为邻居) ---
    ei_rr = _to_np(graph["edge_index_rr"])
    if ei_rr.size > 0:
        seen_rr = set()
        for i in range(ei_rr.shape[1]):
            s, d = int(ei_rr[0, i]), int(ei_rr[1, i])
            pair = (min(s, d), max(s, d))
            if pair in seen_rr:
                continue
            seen_rr.add(pair)
            edges.append({
                "id": f"e{edge_id}", "source": f"R{s + 1}", "target": f"R{d + 1}",
                "edge_type": "Region-Region",
                "features": {}
            })
            edge_id += 1

    n_edges = sum(1 for e in edges if e["edge_type"] == "Line-Line")
    n_line_region = sum(1 for e in edges if e["edge_type"] == "Line-Region")
    n_region_region = sum(1 for e in edges if e["edge_type"] == "Region-Region")

    # 置信度图 / 错误图: 直接按文件字节嵌入 (不重新编码), 缺失则给 None
    def _file_b64(p):
        try:
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("ascii")
        except Exception as _e:
            print(f"[可视化] 读取 {p} 失败: {_e}")
        return None

    bg_conf_b64 = _file_b64(confidence_path)
    bg_disagreement_b64 = _file_b64(disagreement_path)
    bg_err_b64 = _file_b64(error_path)
    bg_pixel_err_b64 = _file_b64(pixel_error_path)
    train_preview_data = []
    for _label, _p in (train_previews or []):
        _b = _file_b64(_p)
        if _b:
            train_preview_data.append({"label": _label, "b64": _b})
    image_metrics = None
    if metrics_path and os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding='utf-8') as _f:
                _m = json.load(_f)
            image_metrics = {key: _m.get(key) for key in (
                'region_acc', 'region_correct', 'region_total',
                'pixel_acc', 'pixel_miou', 'pixel_acc_exact_rgba',
                'pixel_coverage', 'excluded_unknown_gt_pixels',
                'unknown_prediction_pixels', 'pixel_metric_defined', 'region_metric_defined',
                'pixel_total_exact_rgba',
            )}
        except (OSError, ValueError, TypeError) as _e:
            print(f'[可视化] 无法读取本次评价指标 {metrics_path}: {_e}')

    vis_data = {
        "line_nodes": _to_native(line_nodes),
        "region_nodes": _to_native(region_nodes),
        "edges": _to_native(edges),
        "bg_skel_b64": bg_skel_b64,
        "bg_region_b64": bg_region_b64,
        "bg_conf_b64": bg_conf_b64,
        "bg_disagreement_b64": bg_disagreement_b64,
        "bg_err_b64": bg_err_b64,
        "bg_pixel_err_b64": bg_pixel_err_b64,
        "train_previews": train_preview_data,
                "image_metrics": image_metrics,
        "color_palette": color_hex_map,
        "stats": {
            "n_nodes": N_l, "n_regions": num_regions,
            "n_edges": n_edges,
            "n_line_region": n_line_region,
            "n_region_region": n_region_region,
            "img_w": w, "img_h": h
        }
    }

    graph_json = json.dumps(vis_data, ensure_ascii=False)
    html = _VIS_HTML_TEMPLATE.replace("%%GRAPH_DATA%%", graph_json)

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    st = vis_data["stats"]
    print(f"[可视化] 已生成: {output_html}")
    print(f"  Line={st['n_nodes']} | Region={st['n_regions']} | "
          f"L-L={st['n_edges']} | L-R={st['n_line_region']} | R-R={st['n_region_region']}")


def _check_image_pairs(line_paths, color_paths, purpose):
    """杜绝按排序直接 zip 时的线稿/GT 错配（错配会使所有准确率失真）。

    主程序训练集与最终成图评价必须使用相同且一一对应的文件名 stem；
    如果实际项目的文件名规则不同，请先建立显式映射，而不是静默按字典序配对。
    """
    if len(line_paths) != len(color_paths):
        raise ValueError(f'{purpose}: 线稿 {len(line_paths)} 张，GT {len(color_paths)} 张，无法一一配对')
    seen = set()
    for i, (lp, cp) in enumerate(zip(line_paths, color_paths)):
        ls = os.path.splitext(os.path.basename(os.fspath(lp)))[0]
        cs = os.path.splitext(os.path.basename(os.fspath(cp)))[0]
        if ls != cs:
            raise ValueError(f'{purpose}: 第 {i+1} 对文件名不一致：{ls!r} vs {cs!r}；'
                             '不能按排序直接比较，否则会把上色正确的图判错')
        if ls in seen:
            raise ValueError(f'{purpose}: 重复的文件 stem {ls!r}，会导致 GT 对应关系不唯一')
        seen.add(ls)


def build_dataset_bundle(root, num_workers, use_pin_memory, save_cache=True,
                         train_root=None, val_root=None):
    """为单个数据集根目录构建训练/验证 bundle (多数据集共享 backbone 时的基本单元)。

    train_root / val_root: 显式指定训练/验证目录 (含 NotPainted/ 与 Painted/ 子目录)。
        None 时回退到 <root>/train 与 <root>/val。

    Returns:
        dict 含 name/root/train_lines/train_colors/val_lines/val_colors/line_palette/
        color_palette/num_classes/train_loader/val_loader/triple_map/num_triples/
        inv_triple_map/line_color_to_fill; 无训练图时返回 None。
    """
    tr = train_root if train_root is not None else os.path.join(root, "train")
    vr = val_root if val_root is not None else os.path.join(root, "val")
    train_lines = sorted(glob.glob(os.path.join(tr, "NotPainted/*")))
    train_colors = sorted(glob.glob(os.path.join(tr, "Painted/*")))
    val_lines = sorted(glob.glob(os.path.join(vr, "NotPainted/*")))
    val_colors = sorted(glob.glob(os.path.join(vr, "Painted/*")))
    if len(train_lines) == 0:
        return None
    _check_image_pairs(train_lines, train_colors, f'{tr} 训练集')
    _check_image_pairs(val_lines, val_colors, f'{vr} 验证集')

    name = os.path.basename(os.path.normpath(root))
    all_lines = train_lines + val_lines
    all_colors = train_colors + val_colors
    # 关键: 调色板缓存键必须含数据集标识。build_global_palettes 的缓存键只有 version,
    #   若多数据集共用 "v6_sorted" 会串缓存 (后处理的数据集拿到别家的调色板 → label 错乱)。
    #   这里 version=f"v6_sorted_{name}" 使每个数据集有独立缓存文件, 彻底隔离。
    line_palette, color_palette = build_global_palettes(all_lines, all_colors,
                                                        version=f"v6_sorted_transparent_{name}",
                                                        save_cache=save_cache)
    # 线色调色板截断到 MAX_LINE_COLORS: 保证 color_index < MAX_LINE_COLORS, 使共享 backbone 的
    # color_embed / region_color_compress 维度固定 (build_global_palettes 已按像素数降序, 前 8 个最主色)。
    line_palette = {k: v for k, v in line_palette.items() if k < MAX_LINE_COLORS}
    num_classes = len(color_palette)

    train_dataset = LineartLLDataset(train_lines, train_colors, line_palette, color_palette,
                                     f"train_{name}", save_cache=save_cache)
    train_batch_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              collate_fn=ll_collate_fn, num_workers=num_workers,
                              pin_memory=use_pin_memory, persistent_workers=num_workers > 0)
    triple_map = train_dataset.triple_map
    num_triples = train_dataset.num_triples
    inv_triple_map = train_dataset.inv_triple_map
    line_color_to_fill = train_dataset.line_color_to_fill
    pair_transition = train_dataset.pair_transition
    val_loader = None
    if val_lines and val_colors:
        val_dataset = LineartLLDataset(val_lines, val_colors, line_palette, color_palette,
                                       f"val_{name}", save_cache=save_cache,
                                       triple_map=triple_map, line_color_to_fill=line_color_to_fill,
                                       pair_transition=pair_transition)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,
                                collate_fn=ll_collate_fn, num_workers=num_workers,
                                pin_memory=use_pin_memory, persistent_workers=num_workers > 0)
    return {
        'name': name,
        'root': root,
        'train_lines': train_lines, 'train_colors': train_colors,
        'val_lines': val_lines, 'val_colors': val_colors,
        'line_palette': line_palette, 'color_palette': color_palette,
        'num_classes': num_classes,
        'train_loader': train_loader, 'val_loader': val_loader,
        'triple_map': triple_map, 'num_triples': num_triples,
        'inv_triple_map': inv_triple_map,
        'line_color_to_fill': line_color_to_fill,
        'pair_transition': pair_transition,
    }


def compute_x_l_stats(loaders, device='cpu'):
    """从训练集 X_l 计算逐维 mean/std (z-score 标准化统计量)。

    loaders: 单个 DataLoader 或 list (多数据集时联合统计)。所有图 X_l 拼接后统计;
             本任务为小样本 (每图仅数十~数百条线), 直接 concat 内存无压力。
    Returns:
        (mean, std) 两个 1D tensor (shape=(LINE_FEAT_DIM,), dtype=float32)。
        无任何数据时返回 mean=0 / std=1 (恒等标准化, 安全退化)。
    """
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    xs = []
    for loader in loaders:
        for batch in loader:
            x = batch.get("X_l")
            if x is not None and x.numel() > 0:
                xs.append(x.reshape(-1, LINE_FEAT_DIM))
    if not xs:
        return (torch.zeros(LINE_FEAT_DIM),
                torch.ones(LINE_FEAT_DIM))
    all_x = torch.cat(xs, dim=0).float()
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0, unbiased=False)
    # 防零方差维度: 恒为常数的特征 std≈0 → 除零; 用 1.0 兜底
    # (该维标准化后全为 0, 无信息但不爆炸)。
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.to(device), std.to(device)


def _run_multitask_pipeline(bundles, device, args):
    """多数据集两阶段训练 + 推理。

    阶段一: 所有数据集联合训练共享判别器 backbone (LineColorizerBackbone) + 每数据集独立
            分类头 (ClassifierHead) → backbone 学到跨数据集通用判别特征。
    阶段二: 对每个数据集, 冻结 backbone, 仅微调其分类头 → 快速适配该数据集, 并对 predict
            目录上色推理。
    """
    names = list(bundles.keys())
    print(f"\n[多数据集] 共 {len(names)} 套: {names}")

    backbone = LineColorizerBackbone().to(device)
    heads = {name: ClassifierHead(bundles[name]['num_triples'], num_classes=bundles[name]['num_classes'],
                                  hidden_dim=HIDDEN_DIM, cls_side_dim=CLASSIFIER_SIDE_DIM).to(device) for name in names}

    _ck_sfx = ('_' + _TRAIN_LABEL) if _TRAIN_LABEL else ''
    backbone_path = os.path.join("Output", f"gnn_backbone_pretrained{_ck_sfx}.pth")

    # ---- 阶段一: 联合训练判别器 backbone (若已存在则直接加载) ----
    if os.path.exists(backbone_path):
        ckpt = load_checkpoint(backbone_path, device)
        sd = ckpt.get('backbone_state', ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            backbone.load_state_dict(sd)
            print(f"已加载预训练 backbone: {backbone_path}")
        except RuntimeError:
            print("[警告] 预训练 backbone 不兼容, 重新联合训练")
            os.remove(backbone_path)
            backbone = LineColorizerBackbone().to(device)

    if not os.path.exists(backbone_path):
        # X_l 标准化统计量: 多数据集联合统计 (共享 backbone 用同一组统计量), 注入后随
        #   backbone_state 保存, 阶段二冻结微调与推理均沿用。仅在真正训练时注入;
        #   加载已存在 backbone 时保持 checkpoint 自洽 (权重与统计量配套)。
        _mean, _std = compute_x_l_stats([b['train_loader'] for b in bundles.values()], device)
        backbone.set_x_l_stats(_mean, _std)
        print(f"[标准化] X_l z-score 统计量已注入 (联合 {len(names)} 套训练集):\n"
              f"  mean={_mean.tolist()}\n  std ={_std.tolist()}")
        train_multitask_pretrain(
            backbone, heads, bundles, device,
            epochs=EPOCHS, lr=LEARNING_RATE,
            save_path=backbone_path if args.save else None,
        )

    # ---- 阶段二: 逐数据集冻结 backbone 微调分类头 + 推理 ----
    for name in names:
        b = bundles[name]
        print(f"\n[阶段二] 数据集 {name}: 冻结 backbone, 微调分类头")
        model = LineColorizerGNN(
            num_classes=b['num_classes'], num_triples=b['num_triples'],
            line_in_dim=LINE_FEAT_DIM, edge_attr_dim=EDGE_ATTR_DIM,
            hidden_dim=HIDDEN_DIM,
            line_enc_dim=LINE_ENC_DIM,
            edge_enc_dim=EDGE_ENC_DIM, num_gnn_layers=GNN_LAYERS,
            max_line_colors=MAX_LINE_COLORS, inv_triple_map=b['inv_triple_map']).to(device)
        model.backbone.load_state_dict(backbone.state_dict())

        head_path = os.path.join("Output", f"gnn_colorizer_v6_{name}{_ck_sfx}.pth")
        model = train_ll_gnn(
            model, b['train_loader'], b['val_loader'],
            b['val_lines'], b['val_colors'],
            b['line_palette'], b['color_palette'],
            epochs=EPOCHS, lr=LEARNING_RATE, device=device,
            save_path=head_path, patience=PATIENCE,
            save_weights=args.save,
            inv_triple_map=b['inv_triple_map'],
            line_color_to_fill=b['line_color_to_fill'],
            pair_transition=b['pair_transition'],
            freeze_backbone=True,
            allow_overwrite=bool(getattr(args, 'allow_overwrite_checkpoint', False)),
        )

        # ---- 推理: 对 predict 目录所有线稿上色 ----
        infer_files = sorted(glob.glob(os.path.join(b['root'], "predict/*")))
        if not infer_files:
            print(f"[{name}] 未在 {os.path.join(b['root'], 'predict')} 找到图像")
            continue
        fill_lineart = not args.no_fill_lineart
        for l_path in infer_files:
            base = os.path.splitext(os.path.basename(l_path))[0]
            out_png = os.path.join("output", f"{base}_colorized{('_' + _RUN_LABEL) if _RUN_LABEL else ''}.png")
            res_img, prob_a, prob_b, pred_a, pred_b, pred_line, top3_triples, top4_pairs, top4_side_a, top4_side_b = colorize_image_ll(
                model, l_path, b['line_palette'], b['color_palette'], device,
                inv_triple_map=b['inv_triple_map'],
                fill_lineart=fill_lineart,
                line_color_to_fill=b['line_color_to_fill'],
                out_dir="output", out_stem=base)
            cv2.imencode(".png", cv2.cvtColor(res_img, cv2.COLOR_RGBA2BGRA))[1].tofile(out_png)
            print(f"[{name}] 推理完成: {out_png}")


# ==================== 7. 主入口 ====================
def main():
    parser = argparse.ArgumentParser(description="GNN v6 Line-Centric L-L 动漫线稿上色")
    parser.add_argument("--vis-only", action="store_true",
                        help="仅打印图统计信息, 不训练")
    parser.add_argument("--save", action="store_true",
                        help="保存模型权重和图缓存 (默认不保存)")
    parser.add_argument("--no-fill-lineart", action="store_true",
                        help="推理后不进行色线填充 (色线本身填色 + 色线邻近后处理), 默认开启")
    parser.add_argument("--landscape", action="store_true",
                        help="训练/加载后输出 train+val 损失景观 3D 交互式 HTML (Debug/loss_landscape_3d.html)")
    parser.add_argument("--landscape-steps", type=int, default=LANDSCAPE_STEPS,
                        help="损失景观网格分辨率 (奇数, 默认 21)。41 更精细但约 3.8× 更慢; 15 更快仍可看形状。")
    parser.add_argument("--landscape-range", type=float, default=LANDSCAPE_RANGE,
                        help="损失景观扰动范围 ±range (默认 0.08)。")
    parser.add_argument("--landscape-skip-val", action="store_true",
                        help="损失景观只算 train 面 (不重复前向 val), 省约一半时间。")
    parser.add_argument("--landscape-loss", type=str, default="main",
                        choices=["main", "total"],
                        help="损失景观口径: 'main'(默认)=train 与 val 都算 loss_main, 两网格同量纲可比 "
                             "(Train/Val/Gap 可直接对比); 'total'=train 面算 loss_total(main+proto+region, "
                             "对齐优化器真正最小化目标) 但 val 面仍为 main → 量纲不同, train/val 不可比。")
    parser.add_argument("--landscape-dirs", type=int, default=LANDSCAPE_NUM_DIRECTIONS,
                        help="损失景观方向数 (多方向取均值降低随机切片方差, 默认 3)。"
                             "has_val 时实际只用 1 个梯度方向对(d_mean, d_conflict)。")
    parser.add_argument("--confusion", action="store_true",
                        help="评估后输出线级混淆矩阵 (a/b/c 三维拆分 + Top-K 错分对, Debug/confusion_matrix.json)")
    parser.add_argument("--ablation", type=str, default="",
                        help="贡献源消融: 字符组合 l/c 分别消 h_l(线几何流)/h_r(区域上下文流: 线色几何+位置+形状), "
                             "空=全开。例: '--ablation l' 只留区域流; '--ablation lc' 全消。")
    parser.add_argument("--input-roots", nargs='+', default=None,
                        help="多套数据集根目录 (空格分隔), 如: --input-roots input_A input_B input_C。"
                             "多套时共用判别器 backbone、分类头各自独立: 先联合训练 backbone, 再逐数据集微调分类头。"
                             "默认只用 INPUT_ROOT (单数据集, 走原有流程)。")
    parser.add_argument("--train-root", type=str, default=None,
                        help="训练集目录 (含 NotPainted/ 与 Painted/ 子目录)。默认 <input-root>/train")
    parser.add_argument("--val-root", type=str, default=None,
                        help="验证集目录 (含 NotPainted/ 与 Painted/ 子目录)。默认 <input-root>/val")
    parser.add_argument("--predict-files", nargs='+', default=None,
                        help="待上色线稿路径 (可多个)。默认 <input-root>/predict/* 下所有图")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出图像目录。默认 output")
    parser.add_argument("--seed", type=int, default=None,
                        help="固定随机种子 (复现/对比用)。默认 None = 每次随机种子。"
                             "2 图小样本下种子对结果有可见影响, 固定后可稳定复现。")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="运行设备 (cpu/cuda)。默认 auto: 有 CUDA 用 cuda，否则 cpu。")
    # ---- 消融开关：只覆盖显式给出的项；None = 原版训练/推理口径 ----
    parser.add_argument('--wt-triple', type=float, default=None, help='Triple CE 系数，默认 1')
    parser.add_argument('--wt-region', type=float, default=None, help='Region unary NLL 系数，默认 1')
    parser.add_argument('--wt-side', type=float, default=None, help='双侧 Side CE 系数，默认 1')
    parser.add_argument('--sideinferw', type=float, default=None,
                        help='Region unary 的 Side 融合 β∈[0,1]；训练和推理均生效')
    parser.add_argument('--mrf-lambda', type=float, default=None,
                        help='MRF pairwise 强度；只影响推理，默认 1，设 0 可做零重训消融')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='显式加载/保存 checkpoint 路径；用于同一权重零重训推理对照')
    parser.add_argument('--allow-overwrite-checkpoint', action='store_true',
                        help='允许覆盖已存在的 checkpoint (会先备份旧文件; 默认拒绝覆盖)')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='输出命名标签；未设置时消融/随机种子会自动生成隔离标签')
    parser.add_argument("--mc-samples", type=int, default=None,
                        help="MC Dropout 推理采样次数 (>1 开启: 推理期保持 dropout 开启, K 次前向平均 logits, "
                             "降低小样本方差)。默认 None = 用 MC_DROPOUT_SAMPLES(当前 0=关闭)。建议 10~20。")
    # 使用 parse_known_args 忽略 multiprocessing 子进程的内部参数
    # (PyInstaller 打包后子进程会传入 --multiprocessing-fork 等参数)
    args, _ = parser.parse_known_args()

    # --- 统一获取当前运行模块的全局命名空间 ---
    # 关键修复: 脚本直接运行时本文件以 __main__ 身份加载, 而 `import gnn_colorizer_v6_edge_ll`
    # 会二次加载成 *另一个* 模块对象 (两个独立全局变量字典)。之前用 `_M.XXX = ...` 改的是
    # 副本模块的全局, 而 compute_loss / extract_ll_graph / DropEdge 等函数读的是 __main__ 的
    # 旧全局, 导致所有运行期开关(proto/消融/dropedge/缓存版本)在日志"自检"里显示已改、
    # 实际却完全不生效。改用 globals() 直接操作当前模块命名空间, 无论以 __main__ 还是
    # 被 import 运行都指向真正生效的那一份。
    _G = globals()

    # --- 固定随机种子 (--seed): 在构建任何 dataset / 训练之前重新固定, 覆盖模块加载时的随机种子 ---
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[种子] 固定随机种子 = {args.seed}")
    else:
        print(f"[种子] 随机种子 = {_RANDOM_SEED}")

    # 仅显式设置的参数才覆盖基线；相同配置自动指向相同 checkpoint，
    # 推理参数 mrf-lambda 仅改变输出名，不意外触发重训。
    _loss_overrides = {
        'wt_triple': 'TRIPLE_LOSS_WEIGHT',
        'wt_region': 'REGION_LOSS_WEIGHT',
        'wt_side': 'SIDE_LOSS_WEIGHT',
        'sideinferw': 'SIDE_INFER_WEIGHT',
        'mrf_lambda': 'MRF_MISMATCH_WEIGHT',
    }
    for arg_name, glob_name in _loss_overrides.items():
        value = getattr(args, arg_name)
        if value is None:
            continue
        if value < 0 or (arg_name == 'sideinferw' and value > 1):
            parser.error(f'--{arg_name.replace("_", "-")} 数值范围非法: {value}')
        _G[glob_name] = float(value)

    import hashlib
    _train_identity = {
        'seed': args.seed,
        'loss_weights': {k: _G[v] for k, v in _loss_overrides.items()
                         if k != 'mrf_lambda'},
    }
    _default_identity = {
        'seed': None,
        'loss_weights': {
            'wt_triple': 1.0, 'wt_region': 1.0, 'wt_side': 1.0,
                'sideinferw': 0.0,
        },
    }
    _train_hash = '' if _train_identity == _default_identity else hashlib.sha256(
        json.dumps(_train_identity, sort_keys=True).encode()).hexdigest()[:10]
    _infer_identity = {'train_hash': _train_hash,
                       'mrf_lambda': _G['MRF_MISMATCH_WEIGHT']}
    _infer_hash = '' if (_train_hash == '' and args.mrf_lambda is None) else hashlib.sha256(
        json.dumps(_infer_identity, sort_keys=True).encode()).hexdigest()[:10]
    if args.run_tag is not None:
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', args.run_tag):
            parser.error('--run-tag 只能包含字母、数字、点、下划线或连字符（最多80字符）')
        _run_label = args.run_tag
    else:
        _run_label = ('abl_' + _infer_hash) if _infer_hash else ''
    _G['_RUN_LABEL'] = _run_label
    _G['_TRAIN_LABEL'] = ('abl_' + _train_hash) if _train_hash else ''
    print('[消融配置]', json.dumps({
        'loss_weights': _train_identity['loss_weights'],
        'mrf_lambda': _G['MRF_MISMATCH_WEIGHT'],
        'train_checkpoint_suffix': _G['_TRAIN_LABEL'],
        'output_suffix': _G['_RUN_LABEL'],
        'override_checkpoint': args.checkpoint,
    }, ensure_ascii=False, sort_keys=True))

    # --- MC Dropout 采样次数 (--mc-samples) ---
    if args.mc_samples is not None:
        _G['MC_DROPOUT_SAMPLES'] = args.mc_samples

    # --- 应用消融开关 (在构建任何 dataset 之前, 重新计算缓存版本) ---
    ab = args.ablation.lower()
    if ab:
        _G['ENABLE_LINE_STREAM'] = 'l' not in ab
        _G['ENABLE_RCOLOR_STREAM'] = 'c' not in ab
    # 无论是否消融, 缓存版本都按当前开关状态重算 (单一真相源 _ablation_cache_suffix),
    #   避免手动拼接与模块加载时拼接两处逻辑漂移导致缓存串数据。
    # 注意: base 版本号 (GRAPH_CACHE_BASE) 记录特征 schema 变更, 不能用硬编码旧版本覆盖。
    _G['GRAPH_CACHE_VERSION'] = GRAPH_CACHE_BASE + "_" + _ablation_cache_suffix()

    t_total_start = time.time()

    os.makedirs("Cache", exist_ok=True)
    os.makedirs("Debug", exist_ok=True)
    os.makedirs("Output", exist_ok=True)

    # --- 解析多套数据集根目录 (共享 backbone 的基本单元) ---
    input_roots = args.input_roots or [INPUT_ROOT]

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = (device.type == 'cuda')
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count())
    print(f"[性能] DataLoader workers: {num_workers}, pin_memory: {use_pin_memory}")

    print("构建数据集...")
    t_preprocess_start = time.time()
    bundles = {}
    for root in input_roots:
        b = build_dataset_bundle(root, num_workers, use_pin_memory, save_cache=args.save,
                                 train_root=args.train_root, val_root=args.val_root)
        if b is not None:
            bundles[b['name']] = b
    if not bundles:
        print("未检测到数据集。")
        return
    t_preprocess = time.time() - t_preprocess_start
    print(f"[计时] 预处理: {t_preprocess:.1f}s")

    # 单数据集流程: 解包第一个 bundle 到局部变量 (兼容原代码)
    first = list(bundles.values())[0]
    train_lines = first['train_lines']
    train_colors = first['train_colors']
    val_lines = first['val_lines']
    val_colors = first['val_colors']
    line_palette = first['line_palette']
    color_palette = first['color_palette']
    num_classes = first['num_classes']

    if args.vis_only:
        print("[vis-only] 构建图并生成可视化 HTML + filled 图像")
        for idx, (l_path, c_path) in enumerate(zip(train_lines[:3], train_colors[:3])):
            # HTML 图可视化
            out_html = f"Debug/graph_vis_v6_{idx}.html"
            visualize_ll_graph(l_path, c_path, line_palette, color_palette,
                               output_html=out_html)

            # Filled 图像: 用 target label 填色, 验证 region 分配
            l_bgr = cv2.imdecode(np.fromfile(l_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if l_bgr is None:
                continue
            l_rgb = process_alpha(l_bgr)
            c_bgr = cv2.imdecode(np.fromfile(c_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if c_bgr is None:
                continue
            c_rgb = process_alpha(c_bgr)
            graph = extract_ll_graph(l_rgb, c_rgb, line_palette, color_palette)
            if graph is None:
                continue
            markers = graph["markers"]
            skeleton_bin = graph["skeleton_bin"]
            num_regions = graph["num_regions"]
            h, w = l_rgb.shape[:2]
            filled = np.full((h, w, 3), 255, dtype=np.uint8)
            for r_id in range(num_regions):
                mask = (markers == r_id + 1)
                target_cid = int(graph["y"][r_id].item())
                if target_cid in color_palette:
                    filled[mask] = np.asarray(color_palette[target_cid], dtype=np.uint8)[:3]
            filled[skeleton_bin] = l_rgb[skeleton_bin][:, :3]
            cv2.imencode(".png", cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))[1].tofile(
                f"Debug/vis_filled_{idx}.png")

            # 输出每条线的 region 分配
            lr_a = graph["line_region_a"].numpy()
            lr_b = graph["line_region_b"].numpy()
            ll_a = graph["line_label_a"].numpy()
            ll_b = graph["line_label_b"].numpy()
            region_areas = {}
            for r_id in range(1, num_regions + 1):
                region_areas[r_id] = int(np.sum(markers == r_id))
            print(f"  [{idx}] {os.path.basename(l_path)}: "
                  f"Lines={graph['N_l']}, Regions={num_regions} → Debug/vis_filled_{idx}.png")
            for l_id in range(graph["N_l"]):
                ra, rb = int(lr_a[l_id]), int(lr_b[l_id])
                area_a = region_areas.get(ra + 1, 0)
                area_b = region_areas.get(rb + 1, 0)
                px = graph["line_info"][l_id]['pixel_count']
                mx, my = graph["line_info"][l_id]['mid']
                print(f"    L{l_id}: R{ra}(area={area_a}) R{rb}(area={area_b}) "
                      f"label_a={ll_a[l_id]} label_b={ll_b[l_id]} "
                      f"px={px} mid=({mx:.0f},{my:.0f})")
        return

    # --- 多数据集 → 两阶段: 先联合训练共享 backbone, 再逐数据集冻结 backbone 微调分类头 ---
    if len(bundles) > 1:
        _run_multitask_pipeline(bundles, device, args)
        return

    # --- 单数据集: bundle 已构建好 loader, 直接复用 ---
    train_loader = first['train_loader']
    val_loader = first['val_loader']
    triple_map = first['triple_map']
    num_triples = first['num_triples']
    inv_triple_map = first['inv_triple_map']
    line_color_to_fill = first['line_color_to_fill']

    model = LineColorizerGNN(
        num_classes=num_classes,# 目标调色板类别数（含透明背景类别 0）
        num_triples=num_triples,
        line_in_dim=LINE_FEAT_DIM,# Line 节点原始特征维度
        edge_attr_dim=EDGE_ATTR_DIM,# L-L 边属性维度
        hidden_dim=HIDDEN_DIM,# GNN 隐层维度
        line_enc_dim=LINE_ENC_DIM,# Line编码器中间维度
        edge_enc_dim=EDGE_ENC_DIM,# 边编码器中间维度
        num_gnn_layers=GNN_LAYERS,# GNN 层数
        max_line_colors=MAX_LINE_COLORS,# 固定常量 (跨数据集共享 backbone 所需的统一线色维度)
        inv_triple_map=inv_triple_map,
    )

    _ck_sfx = ('_' + _TRAIN_LABEL) if _TRAIN_LABEL else ''
    model_path = (args.checkpoint if args.checkpoint is not None else
                  os.path.join("Output", f"gnn_colorizer_v6_h{HIDDEN_DIM}{_ck_sfx}.pth"))
    # --- R2-A: 结构/实验身份 (写入 checkpoint 元数据; 只读校验, 不影响数值) ---
    _struct_meta = build_structure_meta(
        first.get('name', ''), args.seed, line_palette, color_palette, triple_map,
        hidden_dim=HIDDEN_DIM, gnn_layers=GNN_LAYERS,
        stroke_branch=STROKE_BRANCH, stroke_stat_dim=STROKE_STAT_DIM)
    print('[结构指纹] ' + json.dumps(
        {k: _struct_meta[k] for k in ('structure_id', 'dataset', 'seed', 'hidden_dim',
                                      'gnn_layers', 'stroke_branch', 'stroke_stat_dim',
                                      'num_classes', 'num_triples', 'code_fp')},
        ensure_ascii=False, sort_keys=True))
    _ckpt_loaded = False
    if args.checkpoint is not None and not os.path.isfile(model_path):
        raise FileNotFoundError(f'显式指定的 --checkpoint 不存在: {model_path}')
    if os.path.exists(model_path):
        # 强制校验并按精确路径加载; 任何不兼容都会抛错, 绝不回退到重新训练/删除。
        _meta, _missing, _unexpected = load_checkpoint_guarded(
            model_path, model, device, expect_meta=_struct_meta)
        _ckpt_loaded = True
        print(f"已加载 checkpoint: {model_path}")
        if isinstance(_meta, dict) and _meta.get('structure_id'):
            print(f"  [checkpoint] structure_id={_meta.get('structure_id')} "
                  f"dataset={_meta.get('dataset')} seed={_meta.get('seed')} "
                  f"stat_dim={_meta.get('stroke_stat_dim')}")

    if not _ckpt_loaded:
        t_train_start = time.time()
        model = train_ll_gnn(
            model, train_loader, val_loader,
            val_lines, val_colors,
            line_palette, color_palette,
            epochs=EPOCHS, lr=LEARNING_RATE, device=device,
            save_path=model_path, patience=PATIENCE,
            save_weights=args.save,
            inv_triple_map=inv_triple_map,
            line_color_to_fill=line_color_to_fill,
            pair_transition=first['pair_transition'],
            save_meta=_struct_meta,
            allow_overwrite=bool(args.allow_overwrite_checkpoint),
        )
        t_train = time.time() - t_train_start
        print(f"[计时] 训练: {t_train:.1f}s")
    else:
        t_train = 0.0

    model = model.to(device)
    # 复用已有 checkpoint 时也要输出评价指标；此前只有重新训练才进入成图评价，
    # 导致同一权重第一次运行与第二次运行的终端/log 口径不同。
    if t_train == 0.0 and val_loader is not None and inv_triple_map is not None:
        _val = evaluate_model(model, val_loader, device, inv_triple_map,
                              line_color_to_fill=line_color_to_fill)
        print(f"[已有权重验证] Graph RegionAcc(全部)={_val['val_region_acc']:.4f} "
              f"({_val['val_region_correct']}/{_val['val_region_total']}); "
              f"仅投票={_val['val_region_voted_acc']:.4f}; "
              f"覆盖率={_val['val_region_coverage']:.4f}; "
              f"PairAcc={_val['val_pair_acc']:.4f}")
        if val_lines and val_colors:
            evaluate_pixel_level(model, val_lines, val_colors, line_palette,
                                 color_palette, device, inv_triple_map,
                                 fill_lineart=not args.no_fill_lineart)

    # --- 混淆矩阵 (可选, 线级, a/b/c 三维拆分) ---
    if args.confusion and val_loader is not None and inv_triple_map is not None:
        print("\n[混淆矩阵] 生成线级混淆矩阵 (a_color / b_color / line_color)...")
        cm_result = compute_line_confusion_matrices(
            model, val_loader, device, inv_triple_map, line_palette, color_palette,
            line_color_to_fill=line_color_to_fill
        )
        out_cm = os.path.join("Debug", "confusion_matrix.json")
        with open(out_cm, "w", encoding="utf-8") as f:
            json.dump(cm_result, f, ensure_ascii=False, indent=2)
        print(f"[混淆矩阵] 已保存 → {out_cm}")
        out_cm_html = os.path.join("Debug", "confusion_matrix.html")
        with open(out_cm_html, "w", encoding="utf-8") as f:
            f.write(cm_result["html"])
        print(f"[混淆矩阵] 已保存热力图 → {out_cm_html}")
        # 打印 Top 错分对
        print("[混淆矩阵] Top 错分对 (真值→预测):")
        for item in cm_result["top_confusions"][:15]:
            print(f"  {item['dim']:12} {item['true']} → {item['pred']}  ({item['count']} 次)")

    # --- 推理: 对指定线稿 (或 predict 目录) 上色并输出到 output-dir ---
    t_infer = 0.0
    if args.predict_files is not None:
        infer_files = sorted(args.predict_files)
    else:
        infer_files = sorted(glob.glob(os.path.join(first['root'], "predict/*")))
    if infer_files:
        t_infer_start = time.time()
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[推理] 检测到 {len(infer_files)} 张 predict 图像, 输出到 {args.output_dir}")
        fill_lineart = not args.no_fill_lineart

        # val 真值按文件名建索引 (predict 与 val 一般同名), 供逐 region 错误图使用
        _gt_by_stem = {}
        for _lc in (val_colors or []):
            _gt_by_stem[os.path.splitext(os.path.basename(_lc))[0]] = _lc
        if _gt_by_stem:
            print(f"[错误图] 匹配到 {len(_gt_by_stem)} 张 val 真值, 将同时输出 <stem>_errors.png")

        # val 图 (带 GT 标签构建的那份 graph) 按文件名建索引: 只用于导出 GT 未见分组,
        #   内部会用 markers 指纹核对是否同一张图/同一套分区, 不匹配就不配对。
        _vgraph_by_stem = {}
        _vline_by_stem = {}
        try:
            _vds = val_loader.dataset if val_loader is not None else None
            if _vds is not None and getattr(_vds, 'graphs', None):
                for _vp, _vg in zip(val_lines or [], _vds.graphs):
                    _key = os.path.splitext(os.path.basename(_vp))[0]
                    _vgraph_by_stem[_key] = _vg
                    _vline_by_stem[_key] = _vp
                print(f"[GT 未见] 已索引 {len(_vgraph_by_stem)} 张 val 图的 GT 标签")
        except Exception as _e:
            print(f"[GT 未见] val 图索引失败: {_e}")

        # 学习用图: 取 1-shot 训练样本 (线稿 + 彩稿), 在 HTML 右下角浮窗展示
        _train_previews = []
        for _tl, _tc in list(zip(train_lines or [], train_colors or []))[:2]:
            _train_previews.append(("训练线稿", _tl))
            _train_previews.append(("训练彩稿", _tc))
        if _train_previews:
            print(f"[学习用图] 将在 HTML 浮窗展示 {len(_train_previews)} 张训练样本图")

        for l_path in infer_files:
            base = os.path.splitext(os.path.basename(l_path))[0]
            _artifact_stem = base + (('_' + _RUN_LABEL) if _RUN_LABEL else '')
            out_png = os.path.join(args.output_dir, f"{_artifact_stem}_colorized.png")
            out_html = os.path.join(args.output_dir, f"{_artifact_stem}_graph.html")
            # 按原图名寻找 GT，但输出文件随 run-tag 隔离，防止多个实验相互覆盖。
            _gt_path = _gt_by_stem.get(base) if _gt_by_stem else None
            # 清除本次将要重建的指标文件，避免评价失败时 HTML 误展示上次运行的旧图。
            if _gt_path is not None:
                for _suffix in ('_errors.png', '_error_hist.png', '_metrics.json'):
                    _stale = os.path.join(args.output_dir, _artifact_stem + _suffix)
                    if os.path.exists(_stale):
                        os.remove(_stale)

            # 1. 上色推理，得到彩色结果图 + 预测结果
            res_img, prob_a_lines, prob_b_lines, pred_a, pred_b, pred_line, top3_triples, top4_pairs, top4_side_a, top4_side_b = colorize_image_ll(
                model, l_path, line_palette, color_palette, device,
                inv_triple_map=inv_triple_map,
                fill_lineart=fill_lineart,
                line_color_to_fill=line_color_to_fill,
                out_dir=args.output_dir, out_stem=_artifact_stem,
                gt_color_path=_gt_path,
                gt_graph=_vgraph_by_stem.get(base),
                gt_line_path=_vline_by_stem.get(base),
            )
            cv2.imencode(".png", cv2.cvtColor(res_img, cv2.COLOR_RGBA2BGRA))[1].tofile(out_png)
            print(f"推理完成, 输出: {out_png}")

            # 2. 生成交互式图结构可视化 HTML（传入预测结果和边际概率）
            try:
                visualize_ll_graph(l_path, out_png, line_palette, color_palette,
                   output_html=out_html,
                   pred_a=pred_a, pred_b=pred_b,
                   logits_a=None, logits_b=None,
                   pred_line=pred_line,
                   prob_a_marginal=prob_a_lines,
                   prob_b_marginal=prob_b_lines,
                   top3_triples=top3_triples,
        top4_pairs=top4_pairs, top4_side_a=top4_side_a,
        top4_side_b=top4_side_b,
        confidence_path=os.path.join(args.output_dir, f"{_artifact_stem}_confidence.png"),
        disagreement_path=os.path.join(args.output_dir, f"{_artifact_stem}_disagreement.png"),
        confidence_diagnostics_path=os.path.join(args.output_dir, f"{_artifact_stem}_confidence_diagnostics.json"),
        pixel_error_path=(os.path.join(args.output_dir, f"{_artifact_stem}_pixel_errors.png")
                          if _gt_path else None),
        error_path=(os.path.join(args.output_dir, f"{_artifact_stem}_errors.png")
                    if _gt_path else None),
        metrics_path=(os.path.join(args.output_dir, f"{_artifact_stem}_metrics.json")
                      if _gt_path else None),
        train_previews=_train_previews)
            except Exception as e:
                print(f"  [警告] 生成图结构 HTML 失败: {e}")

        t_infer = time.time() - t_infer_start
        print(f"[推理] 已处理 {len(infer_files)} 张图，HTML 输出目录：{args.output_dir}（失败见上方警告）")
    else:
        print(f"[推理] 未在 {os.path.join(first['root'], 'predict')} 找到图像")

    # --- 支路融合标量终值 (改进清单第 2 项的判据: 支路是否被主路压回接近 0) ---
    try:
        if BRANCH_GATE_MODE == "node":
            _SHORT = ('coarse_amp', 'cont_coarse_amp', 'region_amp')
        else:
            # global 模式下 amp 不参与前向、无梯度, 恒为初值 -> 必须打印真正生效的全局标量
            _SHORT = ('coarse_scale', 'cont_coarse_scale', 'region_scale')
        _bs = {n.split('.')[-1]: float(p.detach().cpu())
               for n, p in model.named_parameters()
               if n.split('.')[-1] in _SHORT}
        print(f"[支路标量] init={BRANCH_SCALE_INIT} -> " + ", ".join(
            f"{k}={_bs[k]:.4f}" for k in sorted(_bs)), flush=True)
    except Exception as _e:
        print(f"[支路标量] 读取失败: {_e}", flush=True)

    # --- 时间汇总 ---
    t_total = time.time() - t_total_start
    print(f"\n{'='*50}")
    print(f"[计时汇总]")
    print(f"  预处理: {t_preprocess:.1f}s")
    if t_train > 0:
        print(f"  训  练: {t_train:.1f}s")
    if t_infer > 0:
        print(f"  推  理: {t_infer:.1f}s")
    print(f"  总  计: {t_total:.1f}s")
    print(f"{'='*50}")

    # --- 损失景观 3D (可选) ---
    if args.landscape and train_loader is not None:
        print("\n[损失景观] 生成 3D 损失景观 (train + val, 交互式 HTML)...")
        plot_loss_landscape_3d(
            model, train_loader, device, inv_triple_map,
            val_loader=None if args.landscape_skip_val else val_loader,
            steps=args.landscape_steps,
            range_alpha=args.landscape_range,
            range_beta=args.landscape_range,
            seed=LANDSCAPE_SEED,
            num_directions=args.landscape_dirs,
            epoch=EPOCHS,
            landscape_loss=args.landscape_loss,
            output_html="Debug/loss_landscape_3d.html",
        )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # PyInstaller 打包支持
    main()
