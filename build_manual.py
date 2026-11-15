"""Build Chinese PDF manual with ReportLab (STSong-Light CJK CID font)."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
                                Spacer, Image, PageBreak, KeepTogether, Preformatted)

ROOT=Path(__file__).resolve().parent
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
CJK='STSong-Light'
BLACK=colors.HexColor('#172a3a')
BLUE=colors.HexColor('#17658d')
GRAY=colors.HexColor('#57616b')

sty={
 'title':ParagraphStyle('DocTitle',fontName=CJK,fontSize=21,leading=31,textColor=BLACK,spaceAfter=12),
 'sub':ParagraphStyle('DocSubtitle',fontName=CJK,fontSize=11,leading=18,textColor=GRAY,spaceAfter=18),
 'h1':ParagraphStyle('HeadOne',fontName=CJK,fontSize=14,leading=22,textColor=BLUE,spaceBefore=13,spaceAfter=8,keepWithNext=True),
 'h2':ParagraphStyle('HeadTwo',fontName=CJK,fontSize=11,leading=18,textColor=BLACK,spaceBefore=8,spaceAfter=5,keepWithNext=True),
 'body':ParagraphStyle('CJKBody',fontName=CJK,fontSize=9.4,leading=17,textColor=BLACK,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('CJKSmall',fontName=CJK,fontSize=8,leading=13,textColor=GRAY,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('CJKCell',fontName=CJK,fontSize=8,leading=12,textColor=BLACK,wordWrap='CJK'),
 'tableh':ParagraphStyle('CJKTableH',fontName=CJK,fontSize=8.4,leading=12,textColor=colors.white,wordWrap='CJK'),
 'code':ParagraphStyle('Code',fontName='Courier',fontSize=8.1,leading=12.7,textColor=BLACK,backColor=colors.HexColor('#f1f4f6'),leftIndent=8,rightIndent=7,spaceAfter=12),
}

def p(text,style='body'):
    return Paragraph(escape(text),sty[style])

def rich(text,style='body'):
    return Paragraph(text,sty[style])

def h(text):return p(text,'h1')

def h2(text):return p(text,'h2')

def code(text):return Preformatted(text,sty['code'])

def tbl(rows,widths=None):
    data=[]
    for i,row in enumerate(rows):
        data.append([p(v, 'tableh' if i==0 else 'cell') for v in row])
    table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLUE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eef4f7')]),
        ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d5e0e4')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    return table


def headerfooter(canv,doc):
    canv.saveState()
    w,hg=A4
    canv.setStrokeColor(colors.HexColor('#d7dfe3'))
    canv.line(18*mm,19*mm,w-18*mm,19*mm)
    canv.setFont(CJK,8)
    canv.setFillColor(GRAY)
    canv.drawString(19*mm,14*mm,'Boundary-based Topology Reconstruction - Python 研究复现原型')
    canv.drawRightString(w-19*mm,14*mm,f'{doc.page}')
    canv.restoreState()


def main():
    output=ROOT/'算法实现与使用手册.pdf'
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,
                          topMargin=19*mm,bottomMargin=25*mm,
                          title='线稿矢量化算法：Python 实现与使用手册',author='Research reproduction',
                          subject='Boundary-based topology reconstruction - Python implementation')
    story=[]
    story += [p('任意线宽线稿的边界拓扑重建矢量化','title'),
              p('Python 实现、算法解释、运行与评测手册｜基于 Zhang 等人，Computer Graphics Forum (2022)','sub'),
              p('这是一份可运行的研究复现原型，而不是论文作者的官方源码。论文中没有披露的实现细节使用了工程近似，并逐项说明。','body'),
              h('一、目的与总体流程'),
              p('输入黑色线条、白色背景的 PNG/JPG 灰度或彩色线稿。程序将宽线条变为几何中心线，利用边界方向和拓扑关系识别误差，并导出可编辑的 SVG 中心线。'),
              tbl([['阶段','采用的思路','产物'],
                   ['1. 初始提取','二值化 + Zhang–Suen 细化 + 有序轮廓','像素骨架、轮廓'],
                   ['2. 拓扑建模','按端点和交汇点追踪骨架段；边界按转角分段','线段图、切线'],
                   ['3. 删除伪结构','骨架法线与边界切线一致性、边界完整性、反向投射','删去伪尖刺/伪连接'],
                   ['4. 缺口重建','共享边界弧复制、共线直连、二次曲线外推','修复后中心线'],
                   ['5. 矢量导出','中心线简化及三次 Bézier SVG','SVG、PNG、JSON']], [29*mm,80*mm,61*mm]),
              Spacer(1,4*mm),
              p('论文第 3 节图 3 给出了主流程；第 4 节对应这四个核心阶段。本项目的技术细节以 docs/算法原理与复现说明.md 为准。','small'),
              h('二、安装与运行'),
              code('python -m pip install -r requirements.txt\npython demo.py\npython vectorizer.py examples/t_junction.png -o outputs/t_junction\npython -m pytest -q'),
              p('运行后可在 outputs/ 看到 .svg、_binary.png、_skeleton.png、_boundary.png、_refined.png 和 _metadata.json。无需 GPU，也无需额外训练文件。')]

    story += [PageBreak(),h('三、公式和判定标准'),
              h2('3.1 骨架出发的切线一致性（论文式 1–2）'),
              p('对中心线样本 c_i 计算切线 t(c_i)，沿法线两侧查找对应的边界点 bl(c_i)、br(c_i)。分别计算中心切线与左右边界切线的余弦一致性，取更大值 D_skeleton(c_i)。'),
              code('D_skeleton(ci) = max(cos(t(ci), t(bl)), cos(t(ci), t(br)))\nspike: for every ci in candidate, D_skeleton(ci) < 0.99'),
              p('尖刺候选段定义为一端在普通端点、一端在交汇点的骨架段。本项目采用无向切线（余弦绝对值），使边界逆向遍历不影响判定；这是与论文书写形式不同的实现选择。'),
              h2('3.2 边界引导的完整性（论文式 3）'),
              p('如果尖刺候选段左右边界都同时关联到相邻其他骨架段，那么它可能是细化过程中额外形成的错误分支。项目使用共享边界分组 ID 判断，并额外要求候选段较短且局部方向不够可靠，以减少误删真实的 T/Y 分支。'),
              h2('3.3 边界出发的一致性（论文式 4）'),
              code('D_boundary(ci) = max over bi whose normal casts to ci:\n                 cos(t(ci), t(bi))\nconnector: for every ci,\n   D_skeleton(ci) < 0.99 AND D_boundary(ci) < 0.95'),
              p('假连接候选段两端都位于交汇点。本项目通过边界 KDTree 和边界法线与指向骨架的近似垂直约束实现，属于离散投射近似。'),
              h2('3.4 边界引导和拓扑引导重建（论文 4.3）'),
              p('遇到共享边界的两个截断端点，从相同轮廓复制一段连续边界曲线，对其端点位移作插值，使新曲线连接两个骨架端点。对缺乏边界线索但方向共线的端点采用直接连接（余弦阈值 0.99）。其余断线使用二次多项式拟合并外推到交汇处。'),
              h('四、最重要的可配置参数'),
              tbl([['参数','默认值','用途'],
                   ['sample_step','0.5 px','稠密中心线样本和法线扫描步长'],
                   ['spike_cos','0.99','骨架出发的切线一致性阈值'],
                   ['connector_boundary_cos','0.95','边界出发的切线一致性阈值'],
                   ['collinear_cos','0.99','同一直线上的断线匹配'],
                   ['smooth','0.05','样条平滑参数的工程映射'],
                   ['svg_mode','centerline','可切换为 original 近似原始宽度']], [52*mm,31*mm,87*mm])]

    story += [PageBreak(),h('五、示例结果'),
              p('下图是程序随附的五个合成样例：输入由 OpenCV 绘制，输出是本项目算法生成的单像素中心线。依次为 T 形、X 形、Y 形交汇、闭合环和变线宽曲线。'),
              Image(str(ROOT/'outputs/comparison.png'),width=171*mm,height=77*mm),
              p('上排：输入。下排：中心线重建结果。请注意这些只是工程自测案例，不是论文作者的实验数据。','small'),
              h2('5.1 评价指标'),
              p('evaluate.py 提供对称 Chamfer（由预测到真值及真值到预测两项距离变换均值，再除以图像对角线）和 3×3 容差 Occupancy（预测像素落在膨胀真值中的百分比）。论文距离变换归一化细节和 Occupancy 式 6 的排版存在解释空间，因此此处数值不能直接与论文表 1 对比。'),
              tbl([['样例','对称 Chamfer','Occupancy'],
                   ['T 交汇','0.00050','100.0%'],
                   ['X 交汇','0.00052','100.0%'],
                   ['Y 交汇','0.00288','99.2%'],
                   ['闭合圆','0.00053','96.6%'],
                   ['变线宽','0.00055','97.4%']], [58*mm,56*mm,56*mm]),
              p('表中数字来源于随附版本在 512×512 合成例子上的自测（阈值和骨架实现不变时可复算），不代表自然图像的泛化效果。','small'),
              h2('5.2 直接使用矢量导出'),
              code('python vectorizer.py examples/varying_width.png \\\n  -o outputs/width_demo --svg-mode original\npython evaluate.py outputs/t_junction_refined.png \\\n  examples/t_junction_gt.png')]

    story += [PageBreak(),h('六、论文与代码的一一对应'),
              tbl([['论文位置','Python 实现'],
                   ['3：骨架/边界提取','preprocess / skeletonize / Boundaries'],
                   ['4.1.1：骨架拓扑','skeleton_graph / smooth_curve'],
                   ['4.1.2：边界分段','Boundaries.__init__ / groups'],
                   ['4.2.1：尖刺分类','Boundaries.initiated / classify_segments'],
                   ['4.2.2：假连接判定','Boundaries.boundary_initiated'],
                   ['4.3.1：边界补线','Boundaries.reference_patch / reconstruct'],
                   ['4.3.2：断线外推','trim_to_reliable / _quadratic_extension'],
                   ['输出 SVG','save_svg / _svg_path']], [58*mm,112*mm]),
              h('七、无法宣称完全一致的部分'),
              p('论文不是完整的工程规格，尤其没有给出交汇像素聚类、边界语义段对应、边界投射离散实现、全局拓扑优化、样条同义平滑参数、各种交错线的唯一正确推断细节。本项目采用可运行的局部规则，并且不会把这些规则称为作者源代码。'),
              p('原论文 4.1.2 节采用拟合误差超过 1px 的迭代边界分段；这里采用轮廓转角的直接分组。论文 4.3.2 节对剩余曲线可外推到边界并裁剪；这里优先参考原始交汇节点，因此遇到极密集多路交叉可能产生短余线。'),
              p('原作者在论文 5.4 节也指出：艺术化非光滑边界、很短且切线不一致的真实分支及草稿语义问题，仍可能影响结果。不要将本工具用于需要无人工审查的高可靠制造或测量场景。'),
              h('八、项目组织与接口'),
              code('from vectorizer import Config, vectorize, export_result\nimport cv2\nimage = cv2.imread("input.png")\nconfig = Config()\nresult = vectorize(image, config)\nexport_result(result, config, "outputs/my_result")'),
              p('result.strokes 是基本骨架/补线图元的 N×2 浮点坐标数组列表；result.semantic_strokes 是跨交汇点合并后的连续笔画。result.meta 含各类数量统计。完整开发和改进细节见 docs/算法原理与复现说明.md。'),
              h('参考文献'),
              p('Zibo Zhang, Xueting Liu, Chengze Li, Huisi Wu, Zhenkun Wen, “Vectorizing Line Drawings of Arbitrary Thickness via Boundary-based Topology Reconstruction,” Computer Graphics Forum 41(2), 2022, DOI: 10.1111/cgf.14485.','small')]

    # Supplemental page: explicit Stroke topology was missing in the initial version.
    example_pair=Table([[
        Image(str(ROOT/'outputs/x_junction_topology.png'),width=79*mm,height=79*mm),
        Image(str(ROOT/'outputs/t_junction_topology.png'),width=79*mm,height=79*mm)
    ]],colWidths=[85*mm,85*mm])
    example_pair.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                     ('LEFTPADDING',(0,0),(-1,-1),0),
                                     ('RIGHTPADDING',(0,0),(-1,-1),3)]))
    story += [PageBreak(),h('九、新增 Stroke 级拓扑标记（补充版）'),
              p('第一版只输出几何中心线分段和桥接曲线，不包含「哪些线段属于同一完整 Stroke」的显式关系。这一版增加独立的 stroke_topology.py，在交汇点处依据重建阶段已选定的边界匹配与切线配对，将原始骨架段归并为具有可追踪 ID 的完整笔画。'),
              tbl([['标记','含义','数据位置'],
                   ['S0、S1…','连续 Stroke；包含多个骨架段时会在重建配对处拼接','topology.strokes / semantic_strokes'],
                   ['J0、J1…','原始骨架交汇点；记录贯通配对和终止的 Stroke','topology.junctions'],
                   ['E0、E1…','无交汇的曲线端点；闭合环没有端点','topology.nodes'],
                   ['B0、B1…','重建补线；boundary/tangent/extrapolate','topology.bridges']],
                   [25*mm,92*mm,53*mm]),
              Spacer(1,4*mm),example_pair,
              p('左图 X 形：4 个基础骨架段归并为两条贯通 Stroke。右图 T 形：水平主干为一条贯通 Stroke，垂直支线在 J0 终止。彩色路径分别标为 S0 / S1，圆点标出 J 和 E。','small'),
              h2('9.1 新增的输出文件与调用'),
              code('python vectorizer.py examples/x_junction.png -o outputs/x_junction\n# outputs/x_junction_topology.png  (color view)\n# outputs/x_junction_topology.svg  (editable Stroke IDs)\n# outputs/x_junction_topology.json (topology graph)'),
              p('Python 访问：result.semantic_strokes 是连续 Stroke 的 N×2 像素坐标数组列表；result.topology["strokes"] 给出 Stroke ID、start_node、end_node、segment_ids、via_junctions；result.topology["junctions"][i]["continuations"] 记录交汇点处哪些段配对成同一 Stroke。原 result.strokes 仍保留基础几何分段，兼容旧版。'),
              p('这些 S/J/E 标记与文件格式是基于论文拓扑重建思路新增的工程输出，不是论文作者提供的标注规范。对 Y 形等存在多解的交汇，结果是局部几何推断；真实落笔顺序、遮挡层级、抬笔历史无法从单张图像可靠判定。','small'),
              p('更新后测试：15 项通过，覆盖 X/T/Y 标记、闭环和新 SVG/JSON/PNG 导出。完整说明见 README.md 与 docs/算法原理与复现说明.md。','small')]

    doc.build(story,onFirstPage=headerfooter,onLaterPages=headerfooter)
    print(output)

if __name__=='__main__':main()
