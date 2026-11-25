# 含储能增强频率响应的系统规划

本项目研究单母线频率响应、储能增强频率响应（EFR）以及机组和储能联合容量规划。通过动态仿真生成频率最低点样本，训练 ReLU 回归网络，并用大 M 线性化将其嵌入 Gurobi 混合整数规划，再用动态模型复核。

## 笔记本

- `cw2_Q3加入储能和规划.ipynb`：主模型、样本生成、网络比较、规划求解和动态复核。
- `CW2_Q2(1).ipynb`：EFR 模型及成本推导。
- `cw2.ipynb`：无储能参考模型。完整网格约 9621.8 万条，生成和训练耗时较长且需要数 GB 磁盘空间。
- `CW2_Q2_original.ipynb`：Q2 原始笔记。

## 安装与运行

使用 Python 3.10 或 3.11 创建虚拟环境，在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cw2-frequency --display-name 'Python (CW2 frequency)'
jupyter lab
```

Windows 使用 `.venv\Scripts\activate` 激活环境。打开所需笔记本，选择 `Python (CW2 frequency)` 内核，再选择“全部运行”。每个笔记本应从头顺序执行，运行目录为项目根目录。主模型的交互图保存在笔记本输出中，可通过运行单元格重建。

Gurobi 优化需要有效的 Gurobi 许可证。受限许可证可能无法求解本项目的混合整数模型。`slycot` 安装可能需要 Fortran 编译器或 conda 环境。训练与多情景优化可能运行数分钟或更久。

## 数据和可复现性

大型 Parquet 样本、生成的权重和模型审计表不提交。主笔记本的采样、训练和求解单元会重新生成 `cw2_joint_samples.parquet` 和 `cw2_joint_model_audit.csv`。参考笔记本依次执行后会生成 `cw2_samples/`、`cw2_nadir_relu.npz` 和交互图。随机种子在笔记本中指定。

仓库保留绘制图 11、12 及其对比图所用的小型汇总 CSV。要重新计算，请先运行主笔记本的采样单元，再从项目根目录执行：

```bash
python compare_analytic_milp.py
python compare_analytic_strategies.py
python figure6_animation.py
```

PDF、Word 资料不包含在仓库中。笔记本中对本地论文或题目文档的链接需要用户自行提供原文件。

仓库中的笔记本不含运行输出，运行后可重新生成。项目维护者如需提交前自动去除大型输出，在仓库根目录配置：

```bash
git config filter.notebook-clean.clean 'python3 tools/strip_notebook.py'
git config filter.notebook-clean.smudge cat
```
