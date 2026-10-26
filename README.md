# BiMSeg 模块消融：plant-seg-ablation

基于 `xuanpluspro/plant-seg-lab` 的独立消融仓库。固定起点为提交 `049500a93d81fbf6021614a1c9d31264e210c917`，详见 `UPSTREAM.json`。原仓库不受本项目影响；原始说明保存在 `docs/UPSTREAM_README.md`，源模块归属及许可证继续保留。

## 1. 三个开关与八种配置

| 配置名 | MSCA | DRCF（三层） | DCAM（三层） |
|---|:---:|:---:|:---:|
| baseline | 关 | 关 | 关 |
| msca | 开 | 关 | 关 |
| drcf | 关 | 开 | 关 |
| dcam | 关 | 关 | 开 |
| msca_drcf | 开 | 开 | 关 |
| msca_dcam | 开 | 关 | 开 |
| drcf_dcam | 关 | 开 | 开 |
| full | 开 | 开 | 开 |

默认三个模块全部开启。每个开关控制该类模块的所有使用位置。关闭时使用固定替代结构，输入/输出形状和双向传递路径保持一致：

- **MSCA-off**：`Identity(F4)`，无新增参数。
- **DRCF-off**：深层双线性上采样（与原 DRCF 相同的 align_corners=True）+ 浅层特征相加 + 1×1 Conv/BN/ReLU，输出为浅层尺寸。
- **DCAM-off**：浅层 2×2 平均池化 + 深层特征相加 + 1×1 Conv/GN/ReLU，输出为深层尺寸。

基础替代结构见 `model/basic_fusion.py`。baseline 仍保留 ResNet50、横向通道映射、双向路径、前景辅助监督、RGB 细节分支和分割头，**不是另一种 ResNet50-U-Net**。损失函数、预训练来源、数据增强、通道数、训练预算等在各配置中固定。

完整模块相对基础替代的比较会同时改变结构和参数量；它不证明“同参数量下”的优势。各组都会统计实际参数量与速度。DRM/CBM、差异调制、Top-k 等内部机制的消融不属于本次八组实验。

## 2. AutoDL 下载与依赖

```bash
cd /root/autodl-tmp
git clone https://github.com/xuanpluspro/plant-seg-ablation.git
cd plant-seg-ablation
pip install -r requirements.txt
```

建议使用原 plant-seg-lab 的 Python/PyTorch 环境。数据和预训练文件继续使用原来的绝对路径，不必复制。后续更新：

```bash
cd /root/autodl-tmp/plant-seg-ablation
git pull --ff-only origin main
```

## 3. GWFSS 四分类：一条命令运行八组

以下沿用已有目录和 ResNet50 骨干权重，batch=32，epochs=50，seed=11。`--num-classes` 在本项目中表示**不含背景的类别数**，因此四分类填写 3。

```bash
python batch_ablation.py \
  --data-path /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --mask-encoding index \
  --num-classes 3 \
  --backbone-weights /root/autodl-tmp/plant-seg-lab/pretrained/unet_resnet_voc.pth \
  --no-freeze-backbone-bn \
  --batch-size 32 \
  --epochs 50 \
  --workers 4 \
  --device cuda \
  --amp \
  --seeds 11 \
  --output-root run/ablation/gwfss_bs32_ep50
```

默认顺序是 baseline → msca → drcf → dcam → msca_drcf → msca_dcam → drcf_dcam → full。每组启动独立进程，上一组退出后再开始下一组。某组失败会记录状态并继续后续组，最终返回非零退出码。推荐在 tmux 中运行。

想先验证运行链路，可在新目录中只运行两组一轮：

```bash
python batch_ablation.py \
  --data-path /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --mask-encoding index --num-classes 3 \
  --backbone-weights /root/autodl-tmp/plant-seg-lab/pretrained/unet_resnet_voc.pth \
  --epochs 1 --batch-size 2 --workers 4 --device cuda --amp \
  --variants baseline full \
  --output-root run/ablation/gwfss_smoke
```

这只是链路检查，不作为精度结论。正式实验使用前面的 50 轮命令和独立输出目录。

其他数据集只需替换数据路径、编码和前景类别数：

| 数据 | mask-encoding | num-classes |
|---|---|---:|
| 二分类，背景0/前景255 | binary255 | 1 |
| 二分类，背景0/前景1 | index | 1 |
| GWFSS，类别ID 0..3 | index | 3 |
| RiceSEG，类别ID 0..5 | index | 5 |

例如 RiceSEG 数据路径为 `/root/autodl-tmp/nclassbenchmark/data/RiceSegnclass`。请为不同数据集使用不同 output-root。输入图像和标签均要求 512×512；支持 `ImageSets/Segmentation/*.txt` 和 `ImageSets/*.txt`，以及清单 ID 中的内部空格。

## 4. 单组手动运行

```bash
python train_medical.py \
  --data-path /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --mask-encoding index --num-classes 3 \
  --backbone-weights /root/autodl-tmp/plant-seg-lab/pretrained/unet_resnet_voc.pth \
  --no-msca --drcf --dcam \
  --epochs 50 --batch-size 32 --workers 4 --device cuda --amp \
  --output run/train/gwfss_without_msca
```

这对应 `drcf_dcam`。不写任何模块参数则为 full。不要使用旧的 `--no-dynamic-fusion` 代表 DCAM 消融：旧参数对应另一种 FEM 结构，不是本项目定义的基础替代；新训练中会明确拒绝该用法。

Python 构建示例：

```python
from model import OSSSeg
model = OSSSeg(num_classes=4, width=256,
               use_msca=False, use_drcf=True, use_dcam=True)
```

Python 的 `num_classes` 是包含背景的总类别数，与训练 CLI 的前景数量不同。模块只在构建时切换，训练中不随机开关。

## 5. 断点继续

重复原八组命令，仅额外添加 `--resume`：

```text
原来的 python batch_ablation.py 命令 ... --resume
```

必须保持相同 output-root、variants、seeds、数据划分和训练配置，包括总 epochs。已经成功完成且存在 completed.json 的组会跳过；中断组存在 last.pth 时从其优化器、调度器、AMP 与随机状态恢复。不会重新初始化为预训练权重。失败前没有 last.pth 的非空输出目录需先检查/移走，或改用新 output-root，避免覆盖半成品。

修改 batch size、epochs、模型代码或增强策略后应使用新 output-root；不能把改变条件后的训练混入同一消融。

## 6. 结果放在哪里

上面的正式 GWFSS 命令输出到：

```text
/root/autodl-tmp/plant-seg-ablation/run/ablation/gwfss_bs32_ep50/
```

根目录：

| 文件 | 内容 |
|---|---|
| `ablation_results.csv` | 所有配置总体指标，含 msca/drcf/dcam、seed、split、mIoU、Dice、参数量、FPS |
| `test_class_iou.csv` | 测试集论文宽表，每配置/种子一行，每类 IoU 分列 |
| `val_class_iou.csv` | 验证集相同形式的宽表 |
| `all_class_metrics.csv` | 全部配置的逐类指标长表 |
| `sweep_config.json` | 固定训练条件、模型代码哈希、增强策略和划分 ID |
| `sweep_status.json` | 每组成功/失败状态 |
| `ablation_manifest.json` | 汇总来源和单位 |
| `logs/` | 每组终端训练日志 |

各配置目录，例如 `baseline_seed11/`、`full_seed11/`，继续保存：best.pth、last.pth、config.json、splits.json、history.csv/json、metrics.json、results.csv、class_metrics.csv、confusion_matrix.csv，以及训练曲线和可选 benchmark.json。

所有配置按**验证集 mIoU（包含背景）**选最佳权重。最终加载 best.pth 评价登记的 val/test。区域指标使用整个数据集混淆矩阵；标准 Dice 仅平均前景；多分类边界逐图逐前景类别计算。分数保存为 0–1，距离为像素。GFLOPs 统计不完整时为空，不冒充完整计算量。

配置和权重都记录模块开关，预测/评估时按 checkpoint 自动构建对应结构，不用重新手填开关。

## 7. 重新汇总、重新推理

已有 metrics.json 时，重新生成总表无需 GPU：

```bash
python summarize_ablation.py --runs-root run/ablation/gwfss_bs32_ep50
```

需要重新计算指标或测速时，直接评估保存好的模型：

```bash
python evaluate_ablation.py \
  --runs-root run/ablation/gwfss_bs32_ep50 \
  --data-path /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --device cuda --batch-size 8 --workers 4 \
  --output-root run/evaluation/gwfss_ablation_v1
```

这不会训练或覆盖原权重；默认测量参数/FLOPs/FPS，可用 `--no-benchmark` 跳过测速。保留训练时的清单；对应源脚本会核对 checkpoint 中保存的 train/val/test ID。单模型也可继续使用原有 `summarize.py`、`val.py`、`predict.py`、`benchmark.py`。

## 8. 多种子与完整模型兼容

正式报告若使用多次重复，可将训练命令的 `--seeds 11` 改为 `--seeds 11 12 13`，每个配置分别训练三次。总表保留每个种子，不挑最好种子；均值和标准差在汇报阶段计算。

full 配置保留源模型的参数键和计算顺序。源版本 plant-seg-lab v4 的完整模型权重可以严格加载并评估；关闭模块的模型则必须用自己的 checkpoint。既有完整模型结果只有在数据划分、预训练、增强、轮数和选模方式一致时才可作为 full 组，否则请和其余配置一起重新训练。使用 ResNet50 预训练初始化各组编码器，不把完整模型的已训练权重直接用于不同配置的训练消融。

## 9. 验证

```bash
python -m unittest discover -s tests -p 'test_ablation_model.py' -v
python -m unittest discover -s tests -p 'test_ablation_workflow.py' -v
python -m unittest discover -s tests -v
```

模型测试使用真实 ResNet50：八种配置 × 二/四/六分类的 CPU 前向和反向、开关对应的模块类型、各配置 checkpoint 重载，以及 full 与固定源版本的参数键/初始化权重/输出逐元素比较。另有批量运行、续跑、失败记录和逐类汇总测试。源模型参考只用于测试，位于 `tests/upstream_ossseg_reference.py`，不参与生产模型构建。

CPU 合成检查验证工程与数值链路，不证明真实数据精度、CUDA/AMP 完整训练稳定性或 batch=32 的显存容量。原始模块文献与许可证见 model 的 SOURCE 文档、LICENSE.MSCN 和 model/LICENSE.MODA。
