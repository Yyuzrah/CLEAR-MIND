# 第一版复现仓库整理报告

## 1. 整理结论

本次整理没有把原项目整体发布，也没有把旧实验目录换一个位置继续保存。
最终仓库只实现一条第一版主线：

```text
SWC/metadata
→ 固定数据 manifest
→ Arakelov–Green 谱
→ 三个输入 adapter
→ GNN-alpha / TreeLSTM-arch3 / MLP
→ 训练、评估与本地结果归档
```

代码已经迁移为可通过 `pip install` 安装的 Python package，正式实验由 YAML
配置和统一命令行入口驱动。旧 notebook 保留原输出、原执行状态和原文件哈希，
用于证明代码与结果来源；它们不再承担正式运行职责。

## 2. 第一版范围

纳入内容：

- 上游方法：Arakelov–Green/effective-resistance 谱；
- 数据集：ACT-4、JML-4、BIL-6；
- 下游：`gnn_alpha`、`treelstm_arch3`、`mlp`；
- folds 0–7 训练、folds 8–9 测试；
- 五个正式随机种子：42、1453、666、114514、1919810；
- 数据审计、谱缓存、训练、评估、预测、日志、checkpoint 和环境记录。

明确排除：

- Tropical Jacobian；
- CVP/Babai；
- Graph Transformer；
- TreeMoCo、MorphVAE；
- 调参网格、消融脚本和无关 notebook；
- 旧缓存、权重、日志和生成结果；
- 原始 SWC 数据。

这些排除项既不在 package 中，也不会被 Git 跟踪。完整迁移工作存档位于公开仓库
之外，不属于 GitHub 发布内容。

## 3. 源代码来源

| 第一版组件 | 权威来源 | 整理结果 |
|---|---|---|
| 数据 exact loader、谱与 MLP | `V.alpha_MLP_clean_champion_reproduction.ipynb` | 拆分为数据层、谱模块、MLP adapter、模型和 trainer |
| TreeLSTM-arch3 | `V.alpha_TreeLSTM_arch3.ipynb` | 拆分为树 adapter、batch collate、模型和 trainer |
| GNN-alpha 架构 | `V.alpha_GNN.ipynb` | 独立为 `models/gnn_alpha.py` |
| GNN 确定性点云工程实现 | `EM/vbeta_gnn.py` | 只迁移 baseline 点云构建，不迁移 V.beta 的其他方法 |
| 严格 BIL GNN 协议 | `EM/run_valpha_bil_reg_5seeds.py` | 固定 `_reg.swc` cohort 和五种子训练设置 |

三份 notebook 的公开副本均为字节级原样复制，SHA-256 记录在
[`SOURCE_MAPPING.md`](SOURCE_MAPPING.md) 和
`notebooks/reference/SHA256SUMS` 中。

## 4. 数据统一情况

三个模型不再各自读取 CSV、扫描目录或模糊匹配文件，而是共同消费
`manifests/<dataset>.csv`。每行固定：

- `sample_id`；
- 精确相对路径与文件名；
- SWC SHA-256；
- label 名称与整数 ID；
- fold 和 train/test phase；
- BIL 是否为 `_reg.swc`。

锁定规模为：

| 数据集 | Train | Test | Total |
|---|---:|---:|---:|
| ACT-4 | 400 | 95 | 495 |
| JML-4 | 332 | 76 | 408 |
| BIL-6 | 958 | 242 | 1,200 |

BIL 三个下游严格使用同一批 1,200 个 `_reg.swc`。JML 原项目存在
331/77 和 332/76 两套不兼容 fold；第一版选择 champion MLP exact loader
实际使用的 332/76 版本。详细证据见
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)。

## 5. 模块化设计

```text
data/
  registry → manifest → SWC parser → NeuronRecord
                                      │
spectral/
  ArakelovGreenSpectrum ──────────────┘
                 │
                 ├── MLPAdapter      → MLP
                 ├── GNNAdapter      → GNNAlpha
                 └── TreeLSTMAdapter → TreeLSTMArch3
                                            │
                                  shared trainer/artifact writer
```

- `data/` 是唯一数据入口；
- `spectral/` 只负责谱计算与缓存键；
- `adapters/` 把同一个 `NeuronRecord` 转成不同下游输入；
- `models/` 中每个文件只定义一个下游模型，不读取数据、不计算谱；
- `training/` 统一 seed、DataLoader、优化器、scheduler、early stopping、
  评估和结果落盘；
- `configs/` 分离 dataset、spectral、model 和九个 experiment 组合。

未来增加新谱方法时注册新的 extractor；增加新下游时添加独立 adapter、model
和 config，不需要修改既有 manifest 或其他下游模型。

## 6. 路径、缓存与运行环境

所有旧硬编码绝对路径已从正式管线移除。原始数据位置通过
`--data-root` 或 `EM_CONNECTOME_DATA_ROOT` 提供。生成文件只写入：

- `cache/`：本地谱和数据索引缓存；
- `runs/`：每次训练的独立结果目录。

两者均被 Git 忽略。谱缓存键同时包含 SWC 哈希、manifest 哈希、谱方法、
全部谱参数和实现版本，因而不会误用来自其他数据或其他谱方法的旧缓存。

仓库同时提供：

- `pyproject.toml`：package 和依赖声明；
- `requirements.txt`：训练/运行依赖的便捷安装入口；
- `requirements-dev.txt`：开发/验证依赖的便捷安装入口；
- `environment.yml`：conda 安装入口；
- `requirements/reference-environment.txt`：本次验证机器的精确版本记录；
- 每个 run 的 `environment.json`：实际 Python、PyTorch、CUDA、cuDNN 和设备。

## 7. 复现验证

第一版建立了五层证据：

1. 数据：样本数量、ID、fold、label、路径和 SWC 哈希；
2. 谱：直接执行未修改 MLP notebook 谱代码得到的真实样本 golden vectors；
3. adapter：GNN 点云和 TreeLSTM 树结构/父子方向/root/谱标签对齐；
4. 模型：从 notebook 动态加载原类，比较参数键、shape、固定 batch logits 和 loss；
5. 实验：九个配置各运行五个正式种子，并按已发布容差检查均值。

<!-- FORMAL_RESULTS_START -->
在 Python 3.11.9、PyTorch 2.11.0+cu128、CUDA 12.8、cuDNN 9.19、
RTX 5080 Laptop GPU 上完成了全部 `3 × 3 × 5 = 45` 次正式运行。表中为
五种子 test accuracy 的均值 ± 总体标准差，Macro-F1 为五种子均值：

| Dataset | Model | Accuracy (%) | Mean Macro-F1 (%) | 状态 |
|---|---|---:|---:|---|
| ACT-4 | GNN-alpha | 57.26 ± 1.95 | 55.92 | 新 v1 baseline；与原工程 baseline 57.05% 相差 +0.21 pp |
| ACT-4 | TreeLSTM-arch3 | 71.37 ± 1.55 | 70.70 | 历史 72.42%，相差 −1.05 pp，位于 ±2 pp 容差内 |
| ACT-4 | MLP | 63.79 ± 1.07 | 62.67 | 五个种子与 champion notebook 逐个完全一致 |
| JML-4 | GNN-alpha | 70.79 ± 2.55 | 74.71 | 新 canonical baseline |
| JML-4 | TreeLSTM-arch3 | 76.05 ± 2.81 | 80.19 | 新 canonical baseline |
| JML-4 | MLP | 60.26 ± 1.53 | 60.00 | 五个种子与 champion notebook 逐个完全一致 |
| BIL-6 | GNN-alpha | 91.82 ± 0.48 | 84.42 | 严格历史 baseline 92.73%，相差 −0.91 pp，位于 ±3 pp 容差内 |
| BIL-6 | TreeLSTM-arch3 | 93.80 ± 0.26 | 87.83 | 历史均值 93.80%，均值完全一致 |
| BIL-6 | MLP | 73.31 ± 0.77 | 63.13 | 五个种子与 champion notebook 逐个完全一致 |

遵循原实验，folds 8–9 同时用于 early stopping/checkpoint 选择和结果报告；
因此这些数值是 test-selected 复现结果，不应解释为独立盲测估计。

MLP 的允许均值误差为 ±0.5 个百分点，TreeLSTM 为 ±2 个百分点，GNN 为
±3 个百分点。容差及对应 config、manifest、feature-set 哈希固定在
`reports/v1_regression_targets.yaml`，45 行去本机路径后的正式结果固定在
`reports/v1_formal_results.csv`。

结果 gate 已验证：

```powershell
em-connectome verify results
```

它要求九个实验各有五个唯一正式种子，并同时检查：

- train/test 数量；
- resolved config SHA-256；
- manifest SHA-256；
- Arakelov–Green feature-set hash；
- accuracy 有限且在 `[0, 1]`；
- 五种子均值位于对应容差内。
<!-- FORMAL_RESULTS_END -->

## 8. 与历史结果的解释边界

- MLP notebook 是完整的 champion reproduction 来源，可进行严格逐种子对照；
- TreeLSTM ACT/BIL 使用相同 cohort，可比较历史五种子统计量，但 CUDA/PyTorch
  的 scatter/reduction 算子允许出现少量样本级差异；
- TreeLSTM JML 的旧 77-test 结果不能用来验证统一后的 76-test manifest，
  因此第一版结果是新的 canonical baseline；
- GNN notebook 的保存输出只到 ACT epoch 1，没有完整五种子结果。GNN 的架构和
  adapter 可做代码等价验证，ACT/JML 的正式统计量作为第一版新 baseline；
- 严格 BIL GNN 可与原 `_reg.swc` 五种子脚本的统计结果比较。

本报告不会把不兼容数据划分、未完成 notebook 或不同方法的结果描述成严格复现。

## 9. 最终公开仓库边界

GitHub 只需要以下内容：

```text
configs/
docs/
manifests/
notebooks/reference/
reports/
requirements/
requirements.txt
requirements-dev.txt
scripts/
src/em_connectome/
tests/
.gitignore
environment.yml
pyproject.toml
README.md
```

`runs/`、`cache/`、Python/tool cache、旧项目副本和完整迁移存档均不进入提交。

## 10. 最终审计

在本报告定稿时完成了以下检查：

| 检查 | 结果 |
|---|---|
| `pip install -e ".[train,test]"` | 成功安装 `em-connectome==0.1.0` |
| wheel 构建 | 成功；wheel 仅含 23 个 package 源文件和 4 个 metadata 文件 |
| `pytest -q` | 24 passed |
| `ruff check .` | passed |
| `ruff format --check .` | 41 files already formatted |
| `em-connectome verify all` | data、features、adapters、results 全部通过 |
| 最终 smoke matrix | 9/9 运行成功 |
| 正式 regression matrix | 45/45 运行成功 |
| 正式 run artifact | 45 个唯一目录、360 个必需文件、6,195 行预测，零缺失 |
| notebook | 3/3 nbformat 有效，3/3 SHA-256 与原文件一致 |
| Git 发布候选 | 77 个文件、约 974 KiB，最大文件约 288 KiB |
| 大文件与本地产物 | 无大于 1 MB 的发布候选；`cache/`、`runs/` 均被忽略 |

因此，第一版已经形成一个最小但完整的可安装、可命令行运行、可验证且可扩展
复现仓库。未替课题组作出的发布层决策只有开源许可证和最终作者信息；当前
`pyproject.toml` 明确保留了发布前选择许可证的提示。
