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

请从项目根目录启动 Jupyter，以便 Notebook 中的相对路径正确解析。第 10 章 Notebook 从 `data/raw/` 读取 ZIP 数据，并将结果写入 `outputs/chapter10/`。第 8 章 Notebook 当前仍从其配置的外部 CSV 目录读取数据，结果写入 `outputs/chapter08/`。

论文入口为 `report/main.tex`，应在 `report/` 目录中编译。最终 PDF 位于 `report/final/`。

`archive/` 和 `tmp/` 中的内容均未纳入主要工作流，可在项目复现验证完成后另行清理。
