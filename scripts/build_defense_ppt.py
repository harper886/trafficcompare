from pathlib import Path

import win32com.client


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PPTX_PATH = OUT / "traffic_collision_defense.pptx"


def rgb(r, g, b):
    return r + g * 256 + b * 65536


BG = rgb(5, 11, 22)
PANEL = rgb(12, 24, 44)
PANEL_2 = rgb(18, 35, 62)
CYAN = rgb(34, 211, 238)
BLUE = rgb(96, 165, 250)
AMBER = rgb(245, 158, 11)
ROSE = rgb(244, 63, 94)
GREEN = rgb(52, 211, 153)
WHITE = rgb(241, 245, 249)
MUTED = rgb(148, 163, 184)


def set_text(shape, text, size=20, color=WHITE, bold=False, font="Microsoft YaHei"):
    tr = shape.TextFrame.TextRange
    tr.Text = text
    tr.Font.Name = font
    tr.Font.Size = size
    tr.Font.Color.RGB = color
    tr.Font.Bold = -1 if bold else 0
    shape.TextFrame.MarginLeft = 8
    shape.TextFrame.MarginRight = 8
    shape.TextFrame.MarginTop = 4
    shape.TextFrame.MarginBottom = 4
    return shape


def add_text(slide, text, left, top, width, height, size=20, color=WHITE, bold=False, align=1):
    box = slide.Shapes.AddTextbox(1, left, top, width, height)
    set_text(box, text, size=size, color=color, bold=bold)
    box.TextFrame.TextRange.ParagraphFormat.Alignment = align
    return box


def add_rect(slide, left, top, width, height, fill=PANEL, line=rgb(30, 64, 100), radius=False):
    shape_type = 5 if radius else 1
    s = slide.Shapes.AddShape(shape_type, left, top, width, height)
    s.Fill.ForeColor.RGB = fill
    s.Line.ForeColor.RGB = line
    s.Line.Weight = 1
    return s


def add_image(slide, name, left, top, width, height):
    path = ASSETS / name
    if not path.exists():
        return None
    return slide.Shapes.AddPicture(str(path), 0, -1, left, top, width, height)


def add_title(slide, title, subtitle=None):
    add_text(slide, title, 42, 28, 760, 42, size=26, color=WHITE, bold=True)
    if subtitle:
        add_text(slide, subtitle, 44, 68, 820, 28, size=12, color=MUTED)
    line = slide.Shapes.AddShape(1, 44, 104, 210, 3)
    line.Fill.ForeColor.RGB = CYAN
    line.Line.Visible = 0


def add_footer(slide, idx):
    add_text(slide, f"{idx:02d}", 900, 508, 40, 20, size=10, color=MUTED, align=2)


def add_bullets(slide, items, left, top, width, height, size=18, color=WHITE):
    text = "\n".join([f"• {item}" for item in items])
    box = add_text(slide, text, left, top, width, height, size=size, color=color)
    box.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 4
    return box


def add_metric_card(slide, label, value, left, top, width=150, height=72, color=CYAN):
    add_rect(slide, left, top, width, height, fill=PANEL_2, line=color, radius=True)
    add_text(slide, label, left + 10, top + 10, width - 20, 18, size=10, color=MUTED)
    add_text(slide, value, left + 10, top + 30, width - 20, 28, size=20, color=color, bold=True)


def add_table(slide, headers, rows, left, top, width, height, font_size=12):
    table_shape = slide.Shapes.AddTable(len(rows) + 1, len(headers), left, top, width, height)
    table = table_shape.Table
    for c, h in enumerate(headers, 1):
        cell = table.Cell(1, c)
        cell.Shape.Fill.ForeColor.RGB = rgb(15, 46, 74)
        set_text(cell.Shape, h, size=font_size, color=WHITE, bold=True)
    for r, row in enumerate(rows, 2):
        for c, val in enumerate(row, 1):
            cell = table.Cell(r, c)
            cell.Shape.Fill.ForeColor.RGB = PANEL if r % 2 == 0 else rgb(8, 18, 32)
            set_text(cell.Shape, str(val), size=font_size, color=WHITE)
    return table_shape


def add_notes(slide, notes):
    try:
        slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = notes
    except Exception:
        pass


def add_slide(prs, idx, title, subtitle=None):
    slide = prs.Slides.Add(idx, 12)
    slide.Background.Fill.ForeColor.RGB = BG
    if title:
        add_title(slide, title, subtitle)
    add_footer(slide, idx)
    return slide


def build():
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    prs = app.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    # 1 cover
    s = add_slide(prs, 1, None)
    add_image(s, "promo-defense-cover.png", 0, 0, 960, 540)
    overlay = s.Shapes.AddShape(1, 0, 0, 960, 540)
    overlay.Fill.ForeColor.RGB = rgb(0, 0, 0)
    overlay.Fill.Transparency = 0.35
    overlay.Line.Visible = 0
    add_text(s, "基于 Attention-LSTM 的\n交通碰撞风险预测与可解释分析系统", 70, 128, 720, 120, size=32, color=WHITE, bold=True)
    add_text(s, "毕业设计答辩  |  NYC & Chicago  |  Risk Forecasting Dashboard", 74, 270, 760, 28, size=15, color=CYAN)
    add_text(s, "学生：________    指导老师：________    日期：________", 74, 420, 760, 28, size=14, color=WHITE)
    add_notes(s, "各位老师好，我的毕业设计题目是基于 Attention-LSTM 的交通碰撞风险预测与可解释分析系统。")

    # 2 background
    s = add_slide(prs, 2, "研究背景与问题", "交通安全预警需要同时关注时序变化、空间关联和漏报风险")
    add_image(s, "project-intro.png", 520, 122, 380, 260)
    add_bullets(s, [
        "城市交通碰撞具有突发性和时空相关性",
        "传统统计方法难以刻画复杂动态变化",
        "交通安全场景中漏报代价高于误报",
        "研究目标：提前识别高风险区域和时间片",
    ], 70, 145, 400, 240, size=19)
    add_metric_card(s, "Task", "Risk Warning", 70, 390, 180, 72, CYAN)
    add_metric_card(s, "Focus", "High Recall", 270, 390, 180, 72, ROSE)
    add_notes(s, "本页说明研究背景：交通碰撞风险受时间、空间和区域特征共同影响，预警系统更需要减少漏报。")

    # 3 dataset
    s = add_slide(prs, 3, "数据集构建", "双城交通时序数据、事故标签和空间邻接关系")
    add_image(s, "paper-dataset-construction.png", 500, 120, 390, 250)
    headers = ["Dataset", "Time", "Regions", "Features", "Samples", "Positive"]
    rows = [
        ["NYC", "13,128", "64", "116", "840,192", "22.53%"],
        ["Chicago", "8,784", "27", "116", "237,168", "13.38%"],
    ]
    add_table(s, headers, rows, 60, 145, 410, 105, font_size=10)
    add_bullets(s, [
        "输入结构：time × region × feature",
        "标签表示区域时间片是否存在碰撞或异常",
        "邻接矩阵描述道路、POI 和历史记录关系",
    ], 70, 285, 390, 130, size=17)
    add_notes(s, "本项目使用 NYC 和 Chicago 两个城市数据集，包括交通时序、标签、坐标映射和空间邻接关系。")

    # 4 route
    s = add_slide(prs, 4, "技术路线", "从数据构建到模型训练、实验评估和前端展示")
    steps = ["Data", "Model", "Evaluation", "Export", "Dashboard"]
    x = 80
    for i, step in enumerate(steps):
        add_rect(s, x + i * 165, 205, 120, 80, fill=PANEL_2, line=CYAN, radius=True)
        add_text(s, step, x + i * 165, 226, 120, 28, size=18, color=WHITE, bold=True, align=2)
        if i < len(steps) - 1:
            add_text(s, "→", x + i * 165 + 125, 226, 38, 30, size=28, color=CYAN, bold=True, align=2)
    add_bullets(s, [
        "后端完成完整训练和实验评估",
        "导出前端可读 JSON 保证浏览器交互性能",
        "可视化系统展示风险、标签校验和空间解释",
    ], 120, 340, 720, 100, size=17)
    add_notes(s, "本页说明整体技术路线：数据、模型、评估、导出和前端展示。")

    # 5 model
    s = add_slide(prs, 5, "模型结构", "Attention-LSTM 用于建模交通风险的时序变化")
    add_image(s, "paper-model-structure.png", 65, 122, 830, 300)
    add_bullets(s, [
        "Attention 模块突出关键时间片和区域特征",
        "LSTM 编码器捕捉历史交通状态变化",
        "输出风险概率并形成预警结果",
    ], 80, 438, 760, 58, size=15)
    add_notes(s, "模型采用 Attention-LSTM。Attention 用于权重分配，LSTM 用于建模时间依赖。")

    # 6 optimization
    s = add_slide(prs, 6, "模型优化与后处理", "平滑门和流式后处理提升输出稳定性")
    add_image(s, "paper-ablation-study.png", 510, 118, 380, 255)
    add_bullets(s, [
        "平滑门减少相邻时间片预测抖动",
        "流式后处理增强连续输出稳定性",
        "预警业务中更关注高风险样本召回",
        "目标是在召回能力和稳定性之间取得平衡",
    ], 70, 145, 390, 210, size=18)
    add_metric_card(s, "NYC Recall", "0.8265", 70, 390, 170, 70, GREEN)
    add_metric_card(s, "Chicago Recall", "0.7055", 265, 390, 170, 70, GREEN)
    add_notes(s, "本页说明后处理模块。交通预警不希望概率频繁跳变，因此引入平滑和流式处理。")

    # 7 experiment design
    s = add_slide(prs, 7, "实验设计", "多模型对比和多指标评估")
    add_image(s, "paper-experiment-comparison.png", 500, 120, 390, 270)
    add_bullets(s, [
        "对比模型：GSNet、STG2Seq、ConvLSTM、LSTM",
        "传统模型：LightGBM、XGBoost、LR、ARIMA、HA",
        "评价指标：AUC-PR、AUC-ROC、F1、Accuracy、Recall",
        "类别不平衡场景下重点关注 AUC-PR 和 Recall",
    ], 70, 145, 390, 230, size=17)
    add_notes(s, "本页说明实验设置，强调交通碰撞数据不平衡，不能只看 Accuracy。")

    # 8 results
    s = add_slide(prs, 8, "实验结果", "Myplan 在双城数据集上取得较好的综合表现")
    headers = ["Dataset", "AUC-PR", "AUC-ROC", "F1", "Accuracy", "Recall"]
    rows = [
        ["NYC", "0.6955", "0.8786", "0.6443", "0.7865", "0.8265"],
        ["Chicago", "0.5617", "0.8290", "0.4730", "0.7733", "0.7055"],
    ]
    add_table(s, headers, rows, 85, 145, 790, 118, font_size=13)
    add_bullets(s, [
        "NYC 上 AUC-PR、AUC-ROC、F1、Recall 均表现较优",
        "Chicago 上模型在高风险样本召回方面具有优势",
        "预警场景中 Recall 提升意味着更少漏报",
    ], 110, 315, 720, 105, size=18)
    add_notes(s, "本页讲核心实验结果。强调召回率和 AUC-PR 对交通安全预警更重要。")

    # 9 ablation
    s = add_slide(prs, 9, "消融实验", "验证平滑门和流式后处理的有效性")
    headers = ["Variant", "NYC Recall", "Chicago Recall"]
    rows = [
        ["No Gate + No Stream", "0.7108", "0.6318"],
        ["No Gate", "0.8046", "0.6693"],
        ["No Stream", "0.7407", "0.6401"],
        ["Full Model", "0.8265", "0.7055"],
    ]
    add_table(s, headers, rows, 70, 145, 430, 180, font_size=12)
    add_image(s, "paper-ablation-study.png", 540, 132, 340, 215)
    add_bullets(s, [
        "完整模型在两个城市上 Recall 均最高",
        "模块移除后综合性能下降",
        "说明性能提升来自多模块协同作用",
    ], 90, 375, 760, 80, size=17)
    add_notes(s, "本页说明消融实验，完整模型比去除模块后的版本更稳定。")

    # 10 frontend
    s = add_slide(prs, 10, "前端可视化系统", "风险热力图、标签校验和空间解释联动展示")
    add_image(s, "frontend-real-screenshot.png", 55, 115, 850, 318)
    add_bullets(s, [
        "城市切换、时间轴播放、地图交互",
        "展示风险概率、流量偏差、TP / FP / FN 校验",
        "右侧面板提供局部特征和空间邻接解释",
    ], 80, 448, 780, 55, size=14)
    add_notes(s, "本页展示真实前端截图，说明系统能联动展示地图、指标和解释面板。")

    # 11 frontend data and boundary
    s = add_slide(prs, 11, "前端数据说明与系统边界", "代表性 JSON 样本用于保证浏览器交互性能")
    headers = ["City", "Frames", "Regions", "Records", "With Labels"]
    rows = [
        ["NYC", "13", "64", "832", "296"],
        ["Chicago", "13", "27", "351", "83"],
    ]
    add_table(s, headers, rows, 90, 145, 760, 105, font_size=13)
    add_bullets(s, [
        "前端加载导出的 JSON，而不是直接加载全量 .npy",
        "部分区域描述和展示文案属于可视化语义映射",
        "当前系统是离线预测与可视化原型",
        "不是实时生产级交通平台",
    ], 115, 305, 700, 130, size=18)
    add_notes(s, "本页必须讲清楚边界：不是实时系统，前端展示的是后端导出的代表性样本。")

    # 12 deploy
    s = add_slide(prs, 12, "项目部署与运行", "Git LFS 管理大文件，本地 Web 服务运行前端")
    add_image(s, "paper-deployment-workflow.png", 500, 120, 385, 260)
    add_bullets(s, [
        "GitHub 仓库保存代码、文档、图片和 LFS 指针",
        ".npy / .h5 大文件需要 git lfs pull",
        "前端通过 python -m http.server 8000 启动",
        "不要直接使用 Download ZIP 获取数据集",
    ], 70, 145, 390, 215, size=17)
    add_notes(s, "本页说明项目如何下载和运行，强调 Git LFS。")

    # 13 summary
    s = add_slide(prs, 13, "总结", "完成从数据、模型、实验到可视化的完整闭环")
    add_bullets(s, [
        "构建 NYC 和 Chicago 双城交通风险预测数据流程",
        "设计并训练 Attention-LSTM 风险预测模型",
        "完成多模型对比实验和消融实验",
        "实现前端风险热力图、标签校验和空间解释",
        "形成论文图表、表格、答辩问答和演示材料",
    ], 100, 150, 760, 220, size=21)
    add_metric_card(s, "Pipeline", "Complete", 120, 405, 180, 70, CYAN)
    add_metric_card(s, "Datasets", "2 Cities", 390, 405, 180, 70, AMBER)
    add_metric_card(s, "Dashboard", "Ready", 660, 405, 180, 70, GREEN)
    add_notes(s, "总结项目完成内容，强调完整闭环。")

    # 14 future
    s = add_slide(prs, 14, "不足与展望", "从离线原型扩展到实时交通预警系统")
    add_bullets(s, [
        "当前系统为离线预测与可视化原型",
        "前端展示采用代表性时间帧抽样",
        "尚未接入实时交通流和事故接口",
        "后续可融合真实 GIS 边界、天气、施工和大型活动数据",
        "进一步扩展在线预测和实时预警服务",
    ], 100, 150, 740, 220, size=21)
    add_text(s, "Thank You", 350, 420, 260, 45, size=30, color=CYAN, bold=True, align=2)
    add_notes(s, "最后说明不足与展望，主动强调系统边界。")

    prs.SaveAs(str(PPTX_PATH), 24)
    prs.Close()
    app.Quit()
    print(PPTX_PATH)


if __name__ == "__main__":
    build()
