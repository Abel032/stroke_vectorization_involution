"""Build ONLY the optional Chinese report; never imported by GNN inference."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Preformatted

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
F='STSong-Light';INK=colors.HexColor('#13283a');BLUE=colors.HexColor('#17658d')
base = Path(__file__).resolve().parent
out = base/'B02_密集交点CPU优化说明.pdf'
S = {
 'title':ParagraphStyle('title',fontName=F,fontSize=20,leading=30,textColor=INK,spaceAfter=14),
 'subtitle':ParagraphStyle('subtitle',fontName=F,fontSize=10,leading=17,textColor=colors.HexColor('#586574'),spaceAfter=18),
 'head':ParagraphStyle('head',fontName=F,fontSize=13,leading=21,textColor=BLUE,spaceBefore=12,spaceAfter=8,keepWithNext=True),
 'body':ParagraphStyle('body',fontName=F,fontSize=9.4,leading=17,textColor=INK,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName=F,fontSize=8.4,leading=14,textColor=INK,spaceAfter=7,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName=F,fontSize=8.1,leading=13,wordWrap='CJK'),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=8.1,leading=12.5,leftIndent=6,rightIndent=4,backColor=colors.HexColor('#f1f4f7'),spaceBefore=5,spaceAfter=8),
}
p=lambda t,style='body':Paragraph(escape(str(t)),S[style])
def tbl(rows,widths):
 data=[[p(c,'cell') for c in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff6')),
   ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fa')]),
   ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#dde3e8')),
   ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
   ('LEFTPADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
 return t

def footer(canvas,doc):
 canvas.saveState();w,h=A4
 canvas.setStrokeColor(colors.HexColor('#d7e0e5'))
 canvas.line(19*mm,17*mm,w-19*mm,17*mm)
 canvas.setFont(F,8)
 canvas.setFillColor(colors.HexColor('#506270'))
 canvas.drawString(20*mm,12*mm,'B02 密集交点优化 · CPU 纯计算增量版 · 2026-11-16')
 canvas.drawRightString(w-19*mm,12*mm,str(doc.page))
 canvas.restoreState()

story=[
 p('B02 密集交点与 Stroke 碎裂优化','title'),
 p('CPU 增量实现与 GNN 接口说明｜基于用户上传代码及线稿｜2026-11-16','subtitle'),
 p('这是针对 Zhang 等人（Computer Graphics Forum, 2022）论文启发的 Python 研究实现的工程增量修订，不是论文作者的官方实现。核心变化围绕交点拓扑和 CPU 性能；Segment 宽度仍是每段一个平均值。'),
 p('一、输入与错误来源','head'),
 p('B02 为 2048×730 BGRA 透明线稿。旧版代码直接丢弃 Alpha、只按 BGR 转灰度，可能将透明黑底错误地解析为墨线。现统一按 Alpha 通道阈值 127 建掩码，并保留输入尺寸及坐标。密集区主要还有近邻错误聚类、节点内部小自环与剪枝后伪交点三个来源。'),
 p('二、同一二值输入下的对比','head'),
 tbl([['指标','上传原版','CPU 修订版'],
      ['初始 Segment','625','522'],['有效 Segment','560','414'],
      ['交汇点','221','162'],['2 度误标交点','34','0'],
      ['语义 Stroke','388','273'],['长度中位数','10.48 px','16.80 px'],
      ['桥接记录','455','363'],['一次完整推理','约 9.90 s','约 1.81 s']],[78*mm,43*mm,49*mm]),
 Spacer(1,5*mm),
 p('比较依据：旧版从本轮用户上传的 boundary_topology_python.zip 提取；两版本均读取同一份 alpha>127 的黑白二值图、保持 min_component=4。本表不直接与用户此前使用不同预处理得到的 596 段/336 Stroke 结果比较。速度为当前环境 CPU 单次测量，并非跨平台保证。','small'),
 p('三、优化原则','head'),
 p('降低 Stroke 数量不是终点。除了删除几何图中的重复微自环，只拼接方向可信的残余两臂；两臂同向或模糊时只取消错误的 junction 标签，不强制拼接，防止把相邻的不同画线强行合成一条 Stroke。')]
story += [PageBreak(),p('四、具体几何规则','head'),
 p('4.1 骨架图路径约束的交点聚类','body'),
 p('先用 KDTree 在欧氏半径 3.1px 内筛候选交点簇，再在原始像素骨架图验证连接：至多 3 条 8 邻接边，路径中间仅允许 2 度节点；合并后的空间跨度至多 5px。对于相距 3px 但不连通的两个 T 交点，旧算法会错合并，新规则不会。桥接用的内部 2 度像素并入交汇簇。'),
 p('4.2 Junction 内微小闭环抑制','body'),
 p('对于 start=end=同一个 junction、周长≤7px 且全部落在节点半径 3px 内的短自环，视为像素图冗余，不产生独立闭合 Stroke。不会删除普通独立闭环或大型附着闭环。该上限可显式设置为 0 以关闭。'),
 p('4.3 剪枝后 2 度节点恢复','body'),
 p('伪尖刺删除后，若某 junction 仅连接 2 个有效 Segment，比较两侧远离交点的方向：余弦≤0.35 时，将曲线样本直接沿公共节点拼接，使它重新成为一个 Segment；否则保留两个独立 Segment，并将该节点降级为非交汇端点，避免错误配对。'),
 p('4.4 相对弧长裁切保护','body'),
 p('单侧裁切上限改为 min(max_trim, 2×EDT+3, max(0.75,0.24×Segment 弧长))。防止短段两端被剪光；论文切线一致性判据仍用于寻找实际截断点。'),
 p('五、CPU 计时（B02 三次中位数）','head'),
 tbl([['调用输出模式','CPU 耗时','功能'],
      ['skeleton','1.59 s','中心线及基本元数据'],
      ['segments','1.63 s','另含 414 个 Segment 均宽标量'],
      ['full','1.81 s','另含 273 条 Semantic Stroke 拓扑']],[43*mm,30*mm,97*mm]),
 Spacer(1,5*mm),
 p('优化后仍有 222 / 363 = 61.2% 的桥接为 extrapolate。边界不连续时，几何外推仍是重要回退手段，未完全解决所有遮挡或交叉。均宽在本次 Alpha 掩码下中位 1.50px，不能与此前其他掩码条件下的 2.30px 混为同一测试结果。','small')]
story += [PageBreak(),p('六、GNN 纯计算调用','head'),
 p('被调接口不打开窗口、不绘制图像、不保存 PNG/SVG/JSON/PDF，不引入 Matplotlib。宽度估计不使用逐点 PCHIP，只返回每个关键点段的一个平均值。GNN 的 Line ID 按其自身图编号独立维护。'),
 Preformatted("""import cv2
from skeleton_api import extract_skeleton_mask, extract_skeleton_bundle
from vectorizer import Config

image = cv2.imread('examples/B02_transparent.png',
                   cv2.IMREAD_UNCHANGED)
mask = extract_skeleton_mask(image, Config(compute_widths=False))

bundle = extract_skeleton_bundle(image)
for s in bundle.segments:
    mean_width = s['mean_width_px']  # one scalar / segment
""",S['code']),
 Preformatted("""# GNN already has line_info / line_label_map:
from gnn_adapter import parameterize_gnn_line_graph
r = parameterize_gnn_line_graph(graph)
line_ids = r['line_ids']
mean_widths = r['line_mean_width_px']  # float32[N_line]
""",S['code']),
 p('原始 GNN 默认 LINE_SKELETON_BACKEND=legacy；显式启用 boundary 时，GNN Region 的 4 邻接背景分区若被破坏会报警并回退。对于原始 B02 的 alpha>0 GNN 墨线定义，分区保护实际会触发回退，不能静默强制采用新骨架。为避免 AA 扩宽，输出骨架使用 cv2.LINE_8。','small'),
 p('七、主要参数','head'),
 tbl([['参数','默认','用途'],
      ['junction_merge_mode','geodesic','geodesic 图距离；euclidean 旧版'],
      ['junction_merge_max_hops','3','两簇间允许的路径边数'],
      ['junction_blob_max_span','5.0','聚合后节点簇最大空间跨度'],
      ['junction_microcycle_max_length','7.0','短闭环最大周长，0=关闭'],
      ['degree2_outward_cos','0.35','可折叠两臂的向外切线门限'],
      ['trim_fraction_limit','0.24','每侧最大相对裁切比例'],
      ['alpha_threshold','127','透明线稿的 Alpha 阈值']],[69*mm,22*mm,79*mm]),
 Spacer(1,5*mm),
 p('八、验证与局限','head'),
 p('本次运行 36 项 pytest 回归通过，涵盖真实 B02、相邻非连通交点、闭环保护、两臂折叠、Alpha 与 GNN 接口。命令：python bench_b02_cpu.py --repeats 3。人脸密集区的真实 Stroke 归属仍可能不唯一；没有人工逐交点真值，不能把数量下降当作识别精度提升。详细分析见 docs/B02_密集交点优化_CPU.md。','small')]

doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,
       topMargin=20*mm,bottomMargin=23*mm,title='B02 密集交点 CPU 优化补充说明',author='Python research prototype')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
