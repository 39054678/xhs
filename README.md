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

请从项目根目录启动 Jupyter，以便 Notebook 中的相对路径正确解析。第 3、6、8、10、11 章的分析均从 `data/raw/` 读取 ZIP 数据，生成结果按章节写入 `outputs/`。

建议使用 Python 3.12，并在项目根目录安装已验证依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

可按以下顺序重新生成并执行主要 Notebook：

```powershell
python scripts/build_chapter3_notebooks.py
jupyter nbconvert --to notebook --execute notebooks/chapter03/code3.1.6.ipynb --output code3.1.6.ipynb --output-dir notebooks/chapter03 --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/chapter03/3.2.1.ipynb --output 3.2.1.ipynb --output-dir notebooks/chapter03 --ExecutePreprocessor.timeout=-1

python scripts/build_chapter6_notebook.py
jupyter nbconvert --to notebook --execute notebooks/chapter06/6.1_2.ipynb --output 6.1_2.ipynb --output-dir notebooks/chapter06 --ExecutePreprocessor.timeout=-1

python scripts/build_chapter8_notebook.py
jupyter nbconvert --to notebook --execute notebooks/chapter08/chapter8_correlation_analysis.ipynb --output chapter8_correlation_analysis_executed.ipynb --output-dir notebooks/chapter08 --ExecutePreprocessor.timeout=-1

jupyter nbconvert --to notebook --execute notebooks/chapter10/chapter10_classification.ipynb --output chapter10_classification.ipynb --output-dir notebooks/chapter10 --ExecutePreprocessor.timeout=-1

python scripts/build_chapter11_notebook.py
jupyter nbconvert --to notebook --execute notebooks/chapter11/chapter11_heat_regression.ipynb --output chapter11_heat_regression_executed.ipynb --output-dir notebooks/chapter11 --ExecutePreprocessor.timeout=-1

python scripts/run_robustness_validation.py
```

论文入口为 `report/main.tex`，应在 `report/` 目录中运行 `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`。最终 PDF 位于 `report/final/`。

`archive/` 和 `tmp/` 中的内容均未纳入主要工作流，可在项目复现验证完成后另行清理。

## 当前完成状态

- 第 3 章完成文本长度、情感、可读性、信息密度和标题匹配度分析；两本 Notebook 均已从仓库 ZIP 重跑。
- 第 6 章完成星期、日期类型和发布地的探索性分析，Notebook 已从仓库 ZIP 重跑。
- 第 8 章相关性分析已有完整执行 Notebook 和图表/CSV 输出。
- 第 10 章完成五个分类基准模型，主要比较 PR-AUC、ROC-AUC、F1、精确率和召回率。
- 热度回归实验完成线性模型、随机森林、Extra Trees 和梯度提升树比较，主要使用 RMSLE、Spearman 和 Top 20% 召回率选型。
- `scripts/run_robustness_validation.py` 已完成时间外切分和按博主分组验证，结果写入 `outputs/chapter12/`。
- 报告中的多模态、时序和图网络模型属于后续研究方案；当前没有将它们写成已经完成的数值实验。

## 当前边界

- 当前仅完成一次时间外边界和一次博主分组切分；正式部署前仍需使用多个滚动窗口复核，并完成阈值校准。
- 当前归档没有与每条笔记稳定配对的原始图片、视频、曝光量和传播网络，因而多模态、时序和图模型仍属于后续方案。
