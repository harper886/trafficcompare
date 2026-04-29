from pathlib import Path

import win32com.client


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PPTX_PATH = OUT / "traffic_collision_defense_clean.pptx"


def rgb(r, g, b):
    return r + g * 256 + b * 65536


WHITE = rgb(255, 255, 255)
BG = rgb(248, 250, 252)
NAVY = rgb(15, 23, 42)
TEXT = rgb(30, 41, 59)
MUTED = rgb(100, 116, 139)
BLUE = rgb(37, 99, 235)
CYAN = rgb(8, 145, 178)
LIGHT_BLUE = rgb(219, 234, 254)
LIGHT_CYAN = rgb(207, 250, 254)
AMBER = rgb(217, 119, 6)
LIGHT_AMBER = rgb(254, 243, 199)
GREEN = rgb(22, 163, 74)
LIGHT_GREEN = rgb(220, 252, 231)
ROSE = rgb(225, 29, 72)
LIGHT_ROSE = rgb(255, 228, 230)
BORDER = rgb(226, 232, 240)


def set_text(shape, text, size=18, color=TEXT, bold=False, font="Microsoft YaHei"):
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


def add_text(slide, text, left, top, width, height, size=18, color=TEXT, bold=False, align=1):
    box = slide.Shapes.AddTextbox(1, left, top, width, height)
    set_text(box, text, size=size, color=color, bold=bold)
    box.TextFrame.TextRange.ParagraphFormat.Alignment = align
    return box


def add_shape(slide, left, top, width, height, fill=WHITE, line=BORDER, radius=True):
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


def add_framed_image(slide, name, left, top, width, height):
    add_shape(slide, left - 5, top - 5, width + 10, height + 10, fill=WHITE, line=BORDER, radius=True)
    return add_image(slide, name, left, top, width, height)


def add_title(slide, title, subtitle=None, idx=None):
    add_text(slide, title, 54, 30, 760, 36, size=25, color=NAVY, bold=True)
    if subtitle:
        add_text(slide, subtitle, 56, 68, 740, 24, size=11, color=MUTED)
    bar = slide.Shapes.AddShape(1, 56, 99, 86, 4)
    bar.Fill.ForeColor.RGB = BLUE
    bar.Line.Visible = 0
    if idx is not None:
        add_text(slide, f"{idx:02d}", 874, 34, 40, 22, size=11, color=MUTED, align=2)


def add_bullets(slide, items, left, top, width, height, size=17):
    text = "\n".join([f"• {item}" for item in items])
    box = add_text(slide, text, left, top, width, height, size=size, color=TEXT)
    box.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 6
    return box


def add_tag(slide, label, left, top, width, color=BLUE, fill=LIGHT_BLUE):
    add_shape(slide, left, top, width, 30, fill=fill, line=fill, radius=True)
    add_text(slide, label, left + 6, top + 6, width - 12, 16, size=10, color=color, bold=True, align=2)


def add_metric(slide, label, value, left, top, width=150, color=BLUE, fill=LIGHT_BLUE):
    add_shape(slide, left, top, width, 78, fill=fill, line=fill, radius=True)
    add_text(slide, label, left + 12, top + 12, width - 24, 18, size=10, color=MUTED)
    add_text(slide, value, left + 12, top + 34, width - 24, 28, size=22, color=color, bold=True)


def add_table(slide, headers, rows, left, top, width, height, font_size=11):
    table_shape = slide.Shapes.AddTable(len(rows) + 1, len(headers), left, top, width, height)
    table = table_shape.Table
    for c, h in enumerate(headers, 1):
        cell = table.Cell(1, c)
        cell.Shape.Fill.ForeColor.RGB = LIGHT_BLUE
        set_text(cell.Shape, h, size=font_size, color=NAVY, bold=True)
    for r, row in enumerate(rows, 2):
        for c, val in enumerate(row, 1):
            cell = table.Cell(r, c)
            cell.Shape.Fill.ForeColor.RGB = WHITE if r % 2 == 0 else BG
            set_text(cell.Shape, str(val), size=font_size, color=TEXT)
    return table_shape


def add_notes(slide, notes):
    try:
        slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = notes
    except Exception:
        pass


def add_slide(prs, idx, title, subtitle=None):
    slide = prs.Slides.Add(idx, 12)
    slide.Background.Fill.ForeColor.RGB = BG
    add_title(slide, title, subtitle, idx)
    return slide


def build():
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    prs = app.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    # 1 cover
    s = prs.Slides.Add(1, 12)
    s.Background.Fill.ForeColor.RGB = BG
    add_shape(s, 42, 42, 876, 456, fill=WHITE, line=BORDER, radius=True)
    add_framed_image(s, "promo-defense-cover.png", 520, 86, 330, 250)
    add_tag(s, "Graduation Defense", 78, 86, 150, BLUE, LIGHT_BLUE)
    add_text(s, "基于 Attention-LSTM 的\n交通碰撞风险预测与可解释分析系统", 76, 138, 390, 104, size=29, color=NAVY, bold=True)
    add_text(s, "NYC & Chicago · Risk Forecasting · Explainable Dashboard", 80, 270, 400, 28, size=13, color=CYAN)
    add_text(s, "学生：________    指导老师：________\n专业：________    日期：________", 80, 388, 430, 56, size=13, color=MUTED)
    add_notes(s, "各位老师好，我的毕业设计题目是基于 Attention-LSTM 的交通碰撞风险预测与可解释分析系统。")

    # 2 background
    s = add_slide(prs, 2, "研究背景与问题", "交通安全预警需要提前识别高风险区域，尤其要减少漏报。")
    add_shape(s, 58, 128, 395, 285, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "城市交通碰撞具有突发性和时空相关性",
        "传统统计方法难以刻画复杂动态变化",
        "碰撞样本存在类别不平衡问题",
        "预警场景中漏报代价高于误报",
    ], 78, 155, 350, 190, size=19)
    add_metric(s, "Research Task", "Risk Warning", 78, 345, 160, BLUE, LIGHT_BLUE)
    add_metric(s, "Key Goal", "High Recall", 260, 345, 160, ROSE, LIGHT_ROSE)
    add_framed_image(s, "project-intro.png", 508, 126, 360, 245)
    add_notes(s, "本页讲研究背景，重点说明交通安全场景中漏报代价高，因此更关注风险识别能力。")

    # 3 dataset
    s = add_slide(prs, 3, "数据集构建", "使用双城交通时序数据、事故标签、区域映射和空间邻接关系。")
    add_framed_image(s, "paper-dataset-construction.png", 552, 130, 315, 210)
    headers = ["Dataset", "Time", "Regions", "Features", "Samples", "Positive"]
    rows = [
        ["NYC", "13,128", "64", "116", "840,192", "22.53%"],
        ["Chicago", "8,784", "27", "116", "237,168", "13.38%"],
    ]
    add_table(s, headers, rows, 62, 138, 445, 104, font_size=10)
    add_shape(s, 62, 280, 445, 118, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "输入结构：time × region × feature",
        "标签表示对应区域和时间片是否存在风险",
        "road / poi / record 描述空间邻接关系",
    ], 80, 300, 410, 80, size=16)
    add_notes(s, "本页介绍数据集构建，说明输入数据和标签结构。")

    # 4 route
    s = add_slide(prs, 4, "技术路线", "数据构建、模型训练、实验评估、结果导出和前端展示形成闭环。")
    steps = [("Data", LIGHT_BLUE, BLUE), ("Model", LIGHT_CYAN, CYAN), ("Evaluate", LIGHT_AMBER, AMBER), ("Export", LIGHT_GREEN, GREEN), ("Dashboard", LIGHT_ROSE, ROSE)]
    x = 76
    for i, (label, fill, color) in enumerate(steps):
        add_shape(s, x + i * 166, 200, 124, 78, fill=fill, line=fill, radius=True)
        add_text(s, label, x + i * 166, 225, 124, 24, size=18, color=color, bold=True, align=2)
        if i < len(steps) - 1:
            add_text(s, "→", x + i * 166 + 126, 222, 34, 28, size=24, color=MUTED, bold=True, align=2)
    add_shape(s, 118, 330, 724, 82, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "后端完成完整训练和实验评估，前端加载导出的 JSON 样本",
        "可视化系统展示风险概率、标签校验和空间解释",
    ], 145, 352, 680, 48, size=16)
    add_notes(s, "本页说明技术路线，从数据到模型再到前端展示。")

    # 5 model
    s = add_slide(prs, 5, "模型结构", "Attention-LSTM 用于建模交通风险的时间依赖和关键特征。")
    add_framed_image(s, "paper-model-structure.png", 74, 125, 812, 275)
    add_bullets(s, [
        "Attention 模块突出关键时间片和区域特征",
        "LSTM 编码历史交通状态变化，输出风险概率",
    ], 110, 424, 720, 52, size=16)
    add_notes(s, "本页讲模型结构，LSTM 负责时序建模，Attention 负责关键特征权重分配。")

    # 6 postprocess
    s = add_slide(prs, 6, "模型优化与后处理", "平滑门和流式后处理用于提升预测结果稳定性。")
    add_shape(s, 62, 128, 400, 246, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "平滑门减少相邻时间片预测抖动",
        "流式后处理增强连续输出稳定性",
        "预警系统更关注高风险样本召回",
        "目标是在召回能力和稳定性之间取得平衡",
    ], 85, 158, 350, 170, size=18)
    add_metric(s, "NYC Recall", "0.8265", 88, 390, 155, GREEN, LIGHT_GREEN)
    add_metric(s, "Chicago Recall", "0.7055", 270, 390, 170, GREEN, LIGHT_GREEN)
    add_framed_image(s, "paper-ablation-study.png", 520, 135, 330, 222)
    add_notes(s, "本页解释平滑门和流式后处理，强调输出稳定和召回率。")

    # 7 experiment design
    s = add_slide(prs, 7, "实验设计", "多模型对比和多指标评估验证模型有效性。")
    add_framed_image(s, "paper-experiment-comparison.png", 540, 130, 320, 235)
    add_shape(s, 62, 128, 420, 260, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "深度模型：GSNet、STG2Seq、ConvLSTM、LSTM",
        "传统模型：LightGBM、XGBoost、LR、ARIMA、HA",
        "评价指标：AUC-PR、AUC-ROC、F1、Accuracy、Recall",
        "类别不平衡下重点关注 AUC-PR 和 Recall",
    ], 84, 158, 380, 176, size=17)
    add_notes(s, "本页讲实验设计，说明对比模型和评价指标。")

    # 8 results
    s = add_slide(prs, 8, "实验结果", "本文模型在双城数据集上取得较好的综合表现。")
    headers = ["Dataset", "AUC-PR", "AUC-ROC", "F1", "Accuracy", "Recall"]
    rows = [
        ["NYC", "0.6955", "0.8786", "0.6443", "0.7865", "0.8265"],
        ["Chicago", "0.5617", "0.8290", "0.4730", "0.7733", "0.7055"],
    ]
    add_table(s, headers, rows, 85, 140, 790, 116, font_size=13)
    add_metric(s, "NYC Best Recall", "0.8265", 120, 310, 180, GREEN, LIGHT_GREEN)
    add_metric(s, "Chicago Recall", "0.7055", 390, 310, 180, GREEN, LIGHT_GREEN)
    add_metric(s, "Main Metric", "AUC-PR", 660, 310, 180, BLUE, LIGHT_BLUE)
    add_bullets(s, [
        "交通碰撞数据不平衡，因此不能只看 Accuracy",
        "较高 Recall 表示模型能发现更多真实风险区域",
    ], 120, 420, 720, 50, size=16)
    add_notes(s, "本页讲核心实验结果，强调 Recall 和 AUC-PR。")

    # 9 ablation
    s = add_slide(prs, 9, "消融实验", "验证平滑门和流式后处理对模型性能的贡献。")
    headers = ["Variant", "NYC Recall", "Chicago Recall"]
    rows = [
        ["No Gate + No Stream", "0.7108", "0.6318"],
        ["No Gate", "0.8046", "0.6693"],
        ["No Stream", "0.7407", "0.6401"],
        ["Full Model", "0.8265", "0.7055"],
    ]
    add_table(s, headers, rows, 65, 135, 420, 180, font_size=12)
    add_framed_image(s, "paper-ablation-study.png", 535, 130, 330, 210)
    add_bullets(s, [
        "完整模型在两个城市上 Recall 均最高",
        "模块移除后性能下降，说明改进有效",
    ], 105, 380, 720, 54, size=17)
    add_notes(s, "本页讲消融实验，完整模型优于去掉模块后的版本。")

    # 10 dashboard
    s = add_slide(prs, 10, "前端可视化系统", "风险热力图、标签校验和空间解释联动展示。")
    add_framed_image(s, "frontend-real-screenshot.png", 60, 120, 840, 315)
    add_bullets(s, [
        "城市切换、时间轴播放、地图交互",
        "展示风险概率、流量偏差、TP / FP / FN 校验和邻接解释",
    ], 90, 455, 760, 42, size=14)
    add_notes(s, "本页展示真实前端截图，说明系统展示能力。")

    # 11 boundary
    s = add_slide(prs, 11, "前端数据说明与系统边界", "代表性 JSON 样本用于保证浏览器交互性能。")
    headers = ["City", "Frames", "Regions", "Records", "With Labels"]
    rows = [
        ["NYC", "13", "64", "832", "296"],
        ["Chicago", "13", "27", "351", "83"],
    ]
    add_table(s, headers, rows, 95, 140, 770, 105, font_size=13)
    add_shape(s, 95, 290, 770, 130, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "前端加载导出的 JSON，而不是直接加载全量 .npy 文件",
        "当前系统定位为离线预测与可视化原型",
        "尚未接入实时交通接口，不是生产级实时平台",
    ], 122, 318, 720, 75, size=18)
    add_notes(s, "本页必须讲清系统边界，避免被认为夸大。")

    # 12 deployment
    s = add_slide(prs, 12, "项目部署与运行", "Git LFS 管理大文件，本地 Web 服务运行前端。")
    add_framed_image(s, "paper-deployment-workflow.png", 522, 130, 335, 225)
    add_shape(s, 62, 128, 410, 248, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "GitHub 保存代码、文档、图片和 LFS 指针",
        ".npy / .h5 大文件需要 git lfs pull",
        "前端通过 python -m http.server 8000 启动",
        "不要直接使用 Download ZIP 获取数据集",
    ], 84, 158, 370, 176, size=17)
    add_notes(s, "本页讲部署运行和 Git LFS 注意事项。")

    # 13 summary
    s = add_slide(prs, 13, "总结", "完成从数据、模型、实验到可视化的完整闭环。")
    add_bullets(s, [
        "构建 NYC 和 Chicago 双城交通风险预测数据流程",
        "设计并训练 Attention-LSTM 风险预测模型",
        "完成多模型对比实验和消融实验",
        "实现前端风险热力图、标签校验和空间解释",
    ], 120, 140, 720, 185, size=21)
    add_metric(s, "Pipeline", "Complete", 130, 380, 180, BLUE, LIGHT_BLUE)
    add_metric(s, "Datasets", "2 Cities", 390, 380, 180, AMBER, LIGHT_AMBER)
    add_metric(s, "Dashboard", "Ready", 650, 380, 180, GREEN, LIGHT_GREEN)
    add_notes(s, "总结项目成果，强调闭环完整。")

    # 14 future
    s = add_slide(prs, 14, "不足与展望", "从离线原型扩展到实时交通预警系统。")
    add_shape(s, 90, 132, 780, 250, fill=WHITE, line=BORDER, radius=True)
    add_bullets(s, [
        "当前系统为离线预测与可视化原型",
        "前端展示采用代表性时间帧抽样",
        "尚未接入实时交通流和事故接口",
        "后续可融合真实 GIS 边界、天气、施工和大型活动数据",
        "进一步扩展在线预测和实时预警服务",
    ], 118, 160, 730, 180, size=20)
    add_text(s, "Thank You", 346, 425, 270, 42, size=30, color=BLUE, bold=True, align=2)
    add_notes(s, "最后说明不足与展望。")

    prs.SaveAs(str(PPTX_PATH), 24)
    prs.Close()
    app.Quit()
    print(PPTX_PATH)


if __name__ == "__main__":
    build()
