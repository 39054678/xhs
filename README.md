# REDBOOK 数据分析项目

本项目用于小红书数据分析、分类建模及研究报告撰写。

## 目录

- `data/raw/`：原始数据压缩包；分析时保持原文件不变。
- `notebooks/`：按论文章节归类的 Jupyter Notebook。
- `outputs/`：Notebook 生成的图表、CSV 和 JSON 结果。
- `scripts/`：用于生成 Notebook 等辅助脚本。
- `report/`：LaTeX 报告源码、插图和最终 PDF。
- `archive/`：旧版本、备份和可再生成文件，确认无用前不删除。
- `tmp/`：PDF 页面截图、文本提取等临时材料。

## 使用约定

请从项目根目录启动 Jupyter，以便 Notebook 中的相对路径正确解析。第 10 章和热度回归 Notebook 从 `data/raw/` 读取 ZIP 数据，并分别将结果写入 `outputs/chapter10/` 和 `outputs/chapter11/`。第 8 章 Notebook 当前仍从其配置的外部 CSV 目录读取数据，结果写入 `outputs/chapter08/`；在完成原始 ZIP 读取适配前，该章节尚不能仅依靠仓库内容从零复现。

论文入口为 `report/main.tex`，应在 `report/` 目录中编译。最终 PDF 位于 `report/final/`。

`archive/` 和 `tmp/` 中的内容均未纳入主要工作流，可在项目复现验证完成后另行清理。

## 当前完成状态

- 第 8 章相关性分析已有完整执行 Notebook 和图表/CSV 输出。
- 第 10 章完成五个分类基准模型，主要比较 PR-AUC、ROC-AUC、F1、精确率和召回率。
- 热度回归实验完成线性模型、随机森林、Extra Trees 和梯度提升树比较，主要使用 RMSLE、Spearman 和 Top 20% 召回率选型。
- 报告中的多模态、时序和图网络模型属于后续研究方案；当前没有将它们写成已经完成的数值实验。

## 已知复现缺口

- 第 3、6、8 章的早期 Notebook 仍包含外部数据目录或未完整执行单元，需要迁移到 `data/raw/` 工作流。
- 当前模型使用分层随机划分；正式部署或对外报告前应补充时间外验证和按博主分组验证。
- `report/main.tex` 是论文入口；最终交付前应重复编译直到交叉引用稳定，并将最新 PDF 同步到 `report/final/`。
