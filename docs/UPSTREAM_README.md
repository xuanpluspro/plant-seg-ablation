# Plant Seg Lab：RGB 植物语义分割

当前固定使用 **MPM-5531 + 稀疏注意力** 替换 CSSP，并使用跨尺度动态调制替换三个 CAFR。支持二分类和多分类。该组合的真实精度需要实验验证，不预先宣称提升。

## 模型结构

RGB → ResNet-50四级特征 → 通道统一 → 最深层MPM-5531＋稀疏注意力＋残差 → DRM/CBM自顶向下融合 → 辅助前景引导 → 三个跨尺度动态调制 → 多尺度解码器＋RGB细节分支 → K类logits。

- `model/mpm.py`：通道四分，依次经过5×5、5×5、3×3、1×1普通卷积及BN/ReLU；两个5×5分支共享卷积和BN。四路结果与原输入仅拼接一次到2C，再由CBR融合回C得到Fm。Q由Fm经1×1投影生成；K/V由CAM（坐标注意力）输出经逐位置通道LayerNorm和1×1投影生成，分别保留约1/2、1/6位置的双Top-k注意力聚合后，经输出投影加回Fm。默认8头，α/β从0.5开始学习且无非负或和为1的约束。CBR保留MPCNet的可学习7×7深度卷积激活；CAM采用h-swish。不增加第二组多尺度池化。
- `model/dynamic_fusion.py`：浅层降采样与深层投影，共同生成仿射调制参数；细化后加深层残差。三个位置参数独立。
- `model/ssaf.py`：SACF中的SFA替换为DRM（深层），SDE替换为CBM（浅层）；保留原来的双线性上采样、双Sigmoid空间融合权重和Conv/BN/ReLU输出投影。
- `model/fem.py`：保留原CAFR/FEM，供`--no-dynamic-fusion`对照使用。
- `model/losses.py`：K类交叉熵＋前景Dice＋0.2×辅助前景并集BCE。

512×512输入时，四层统一后的特征为[B,256,128,128]、[B,256,64,64]、[B,256,32,32]、[B,256,16,16]，MPM位于最后一层。动态调制依次更新1/8、1/16、1/32特征。默认宽度256，`--width`可调整。特征交互路径使用FP32，骨干和分割头支持AMP。

当前不提供旧CSSP实现选择、旧模型加载适配或卷积核切换。当前稀疏注意力模型采用v4检查点格式；旧完整模型请使用Git历史版本评估。新实验从ResNet骨干预训练权重开始，不覆盖旧结果目录。数据集、权重文件不会随更新删除。

## AutoDL 安装与更新

使用相互匹配的PyTorch和torchvision（PyTorch≥2.4），然后：

```bash
cd /root/autodl-tmp
git clone https://github.com/xuanpluspro/plant-seg-lab.git
cd plant-seg-lab
pip install -r requirements.txt
```

已有仓库只需在没有训练运行时执行：

```bash
cd /root/autodl-tmp/plant-seg-lab
git pull --ff-only origin main
pip install -r requirements.txt
```

## 数据目录与标签

把不同数据集放到`data/`下，不提交Git。`--data-path`指向含JPEGImages、SegmentationClass、ImageSets的VOC根目录。清单支持`ImageSets/Segmentation/*.txt`或`ImageSets/*.txt`两种布局：train.txt、val.txt和可选test.txt，每行一个无扩展名ID，内部空格和括号原样保留（例如`ARVALIS_ (1)`）。划分必须互不重叠，训练前准备好测试清单，不自动重新划分。

图像和掩码都必须512×512；训练采用同步翻转、直角旋转和温和缩放/裁切/填充，以及仅作用于RGB的轻度亮度/对比度扰动。验证和测试不缩放、不随机增强。RGB按ImageNet mean/std归一化。

| 掩码像素 | 参数 |
|---|---|
| 0背景、255前景 | `--mask-encoding binary255 --num-classes 1` |
| 0背景、1前景 | `--mask-encoding index --num-classes 1` |
| 0、1、2、3、4 | `--mask-encoding index --num-classes 4` |

`--num-classes`不包含背景。默认auto接受二分类0/1或0/255；多分类保留连续类别ID。P模式PNG保留原索引，彩色RGB标签需先转索引。不支持忽略标签；binary255中的255是前景。

```bash
python check_data.py --data-path data/VOC201201 --mask-encoding index --num-classes 1
```

## 预训练与训练

默认使用`pretrained/resnet50.pth`，缺失时报错，不回退随机初始化。可用`python download_pretrained.py`下载torchvision官方ResNet-50 ImageNet-1K V2权重，或手动放入兼容的RGB骨干state_dict。只初始化骨干，分割模块需要训练。

默认batch32、30轮，AdamW，学习率1e-4，余弦调度，AMP开启。默认解冻骨干BN：训练时使用当前batch统计并更新运行均值/方差，验证与推理仍使用eval模式的运行统计。骨干卷积与BN仿射参数均训练；MPM内部BN照常训练。`--freeze-backbone-bn`可手动固定骨干BN统计（不冻结卷积或BN仿射参数）；`--no-freeze-backbone-bn`显式解冻，与新默认相同。

此默认值修改不改变模型结构或v4权重格式；加载权重仍使用检查点记录的设置。旧运行如果保存的是`freeze_backbone_bn=true`，断点恢复须显式加`--freeze-backbone-bn`，否则配置校验会拒绝恢复。不要为了切换BN模式覆盖旧实验，改用新结果目录。

```bash
python train_medical.py \
  --data-path data/VOC201201 \
  --mask-encoding index \
  --num-classes 1 \
  --backbone-weights pretrained/resnet50.pth \
  --batch-size 32 --epochs 30 --workers 4 --device cuda --amp \
  --output run/train/vegann01_mpm5531_sparse_drmcbm_dynamic_bs32_ep30
```

显存不足降低batch；输出目录须为空。`--from-scratch`显式随机初始化，短流程测试建议epochs2、batch1。动态调制默认开启，`--no-dynamic-fusion`恢复原FEM用于对照；MPM保持固定。

当前版本断点恢复使用`--resume 运行目录/last.pth`、相同`--output`和全部原训练参数（包括总epochs），不需再次提供骨干权重。恢复优化器、调度器、AMP和随机状态。

## 自动结果与评估口径

训练/验证均有tqdm进度条，终端显示train/val loss、mIoU、Dice和耗时。所有二分类和多分类任务统一按验证集mIoU（包含背景）保存best.pth；测试集不参与选模。结束后自动加载best.pth评估val与登记的test，保存：

| 文件 | 内容 |
|---|---|
| best.pth、last.pth | 最佳权重、完整训练恢复状态 |
| config.json、splits.json | 参数、数据配置与划分 |
| history.json、history.csv | 逐轮指标、学习率、耗时 |
| metrics.json、results.csv | 最佳轮次及验证/测试汇总 |
| class_metrics.csv | 每类IoU、Accuracy、Dice、Precision、Recall和真实像素数 |
| benchmark.json | 参数、计算量完整性、FPS和延迟 |
| loss_curve.png、metrics_curve.png | 训练曲线 |

指标已统一为 bimseg-segmentation-v2，详见文末“统一指标与结果格式”。前景宏平均与前景并集、空类别和边界处理均有明确区分。


没有test.txt时只输出验证结果，不冒充测试结果。`--no-benchmark`只关闭测速，仍导出精度；测速失败也保留已保存精度。

## 验证、汇总和预测

```bash
python val.py --data-path data/VOC201201 --weights run/train/你的实验/best.pth --split test
python summarize.py --data-path data/VOC201201 --weights run/train/你的实验/best.pth --device cuda --batch-size 32
python predict.py --data-path data/VOC201201/JPEGImages --weights run/train/你的实验/best.pth --output run/predict/你的实验
```

验证/汇总检查划分与训练一致。预测二分类默认输出0/255 PNG与H×W概率，可用`--mask-output-encoding index`导出0/1；多分类PNG为0..K-1，概率K×H×W，并输出彩色叠加图。输出目录须为空，不含大图滑窗。

## 参数量与测速

完整参数量以百万为单位。fvcore乘加计1次运算；存在unsupported_ops时gflops留空，counted_gflops仅为已统计部分，不能视为完整GFLOPs。

FPS固定batch1、512×512、PyTorch eager FP32，关闭autocast与TF32，默认预热50次、测量200次。CUDA Event计时只包含模型前向，不含读图、传输、预处理、后处理，不是TensorRT或端到端速度。CPU不生成GPU FPS。元数据记录设备和软件版本。

```bash
python benchmark.py --weights run/train/你的实验/best.pth --device cuda
```

## 测试与来源

```bash
python -m unittest discover -s tests -v
```

CPU合成数据测试覆盖512前向/反向、检查点恢复、动态融合开关、二分类/五分类训练、结果与预测导出；不能代替真实植物精度和CUDA验证。默认结构不保证比原方法更好。

MODA模块来源、分割改造与许可证见model/README.md及LICENSE.MODA；MPCNet来源与改写见model/MPM_SOURCE.md。当前5531是改写，不能称原版MPM严格复现。旧图和旧实验代码可在Git历史中查找。


## SACF：DRM替换SFA，CBM替换SDE

三个自顶向下位置均改为：深层[B,C,h,w]→DRM→上采样到[2h,2w]；浅层[B,C,2h,2w]→CBM；再使用原有自适应融合，输出[B,C,2h,2w]。三个位置不共享参数。三个CAFR动态调制、损失、标签编码和指标口径保持原设置；最深层MPM现已加入稀疏注意力。

DRM用两路3×3普通卷积（dilation=1/2），直接从输入经GAP生成两个图像级Softmax权重，逐分支加权后Concat并用1×1融合，输出x+gamma*fused；gamma初始化0。CBM采用DWConv→PWConv→GELU局部分支，加上GAP→Conv→Sigmoid生成的通道权重对原输入的重加权，经过输出投影再加输入。没有导入完整HFCB的BN、两路平均或额外外层gamma；来源与改写见model/SACF_SOURCE.md。

检查点格式更新为v4；旧完整模型（包括v3）不能resume到新结构，请从相同ResNet骨干预训练开始新实验。旧权重和结果文件不会被删除，旧代码保留在Git历史。当前已加入Q/KV投影、双Top-k注意力和融合特征残差。


## MPM 稀疏注意力的实现约定

`Fm = CBR(Concat(conv5(x1), conv5(x2), conv3(x3), conv1(x4), x))`。
`Q = ConvQ(Fm)`；`K,V = Split(ConvKV(LN(CAM(Fm))))`；
`A = α·Softmax(Top-half(QKᵀ/√d))·V + β·Softmax(Top-sixth(QKᵀ/√d))·V`；
`output = Fm + ConvOut(A)`。多头拆分和恢复空间形状省略于上述表达式，d为每头通道数。

- 默认C=256、8头、每头32通道；512输入对应16×16特征，单头注意力为256×256。1024输入对应32×32特征，单头注意力为1024×1024；计算/显存随位置数平方增长。
- LayerNorm仅在CAM后的KV分支，按每个空间位置的C通道归一化；Q不额外归一化。不用BN、GroupNorm代替此LN。卷积投影无额外激活；没有新增Dropout或残差缩放系数。
- Top-k以`max(1, floor(N/divisor))`保留位置，divisor固定2/6；未选位置在Softmax前设为负无穷。支持非正方形和1×1特征；保持原始H/W，不从sqrt(N)推断。
- 相关性、Softmax和注意力聚合用FP32。模块在OSSSeg中仍处于原有FP32交互路径；没有引入einops或timm依赖。
- 掩码与矩阵乘法仍为稠密实现，不把Top-k宣称为稀疏算子加速。完整GFLOPs仍按现有工具支持情况报告。
- v4不能加载旧完整模型继续训练。`pretrained/resnet50.pth`仍可用；请新建上述带`sparse`的实验目录。

来源与差异见[model/MPM_SOURCE.md](model/MPM_SOURCE.md)。这是MPCNet与MSCN思路的组合改造，不是原论文严格复现，真实精度与CUDA性能尚需实际训练测量。


## 使用完整 U-Net 中的 ResNet-50 预训练权重

将`unet_resnet_voc.pth`上传至`pretrained/`，训练时指定：

```bash
python train_medical.py \
  --data-path data/VOCdevkit-3.3 \
  --mask-encoding binary255 --num-classes 1 \
  --backbone-weights pretrained/unet_resnet_voc.pth \
  --batch-size 32 --epochs 50 --workers 4 --device cuda --amp \
  --output run/train/vocdevkit33_unetvoc_backbone_bntrain_bs32_ep50
```

加载器接受标准torchvision ResNet-50 state_dict、`resnet.*`或`backbone.*`命名空间（支持`module.`前缀），以及`state_dict`、`model_state_dict`、`model`、`net`字典包装。完整U-Net只提取骨干，解码器不加载；ImageNet的fc分类层也不加载。全部骨干参数和BN缓冲区在加载前严格检查名称与尺寸，不兼容则报错，不使用strict=False静默漏载。终端会显示识别的命名空间、加载张量数和跳过条目数。

这不改变当前模型架构、损失或v4检查点格式，也不把旧U-Net解码器迁移过来。默认仍是`pretrained/resnet50.pth`，只有显式指定时才切换。默认骨干BN解冻；比较预训练权重时保持BN和其他训练设置一致，结果使用新目录。实际用户权重需上传后由严格检查确认兼容；文件名不能证明其训练来源。


## 上传预训练权重后，先检查再训练

```bash
python check_pretrained.py \
  --weights pretrained/unet_resnet_voc.pth \
  --device cuda
```

默认按当前完整模型（width256、512×512、batch1、骨干BN解冻）检查：严格提取全部骨干权重、检测NaN/Inf及非法BN方差、随机输入前向/损失/反向梯度、eval前向。CUDA默认启用AMP，CPU使用FP32。成功末尾打印`CHECK PASSED`并返回退出码0；失败打印`CHECK FAILED`并返回非零，指出文件缺失、名称/尺寸不兼容或数值异常。

希望按训练batch32检查时加`--batch-size 32`，请在其他训练停止后运行。支持`--num-classes`（不含背景）、`--width`、`--size`、`--freeze-backbone-bn`、`--no-dynamic-fusion`、`--no-amp`，应与目标训练设置保持一致。检查不会更新模型优化器参数或写入权重，不使用真实数据；内存中的BN统计会在合成训练前向中更新，原文件不变。检查通过只证明本次权重加载和合成前后向可运行，不保证标签正确、真实精度或整个训练过程不发生显存不足。


## 两种数据划分清单布局

无需复制或移动清单，所有训练、检查、验证和最终测试导出入口统一支持：

| 布局 | train / val / 可选 test |
|---|---|
| 标准VOC | `ImageSets/Segmentation/train.txt`、`val.txt`、`test.txt` |
| 平铺 | `ImageSets/train.txt`、`val.txt`、`test.txt` |

按每个split自动查找；两处都有同名文件时，ID列表与顺序必须一致，否则明确报错，避免误用旧划分。之前复制出的相同清单仍兼容；若之后修改划分，应同步两份或仅保留一份。独立划分、重复ID与数据泄漏校验保持不变；平铺test.txt也会自动登记并参与最终测试。检查点记录的仍是ID列表，不绑定清单目录。

外部数据集用绝对路径即可，例如：

```bash
python check_data.py \
  --data-path /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --mask-encoding index --num-classes 3
```

这里总共4类（背景0、前景1/2/3），所以`--num-classes 3`。数据目录及掩码不做任何自动修改。


文件名可包含内部空格和括号：清单一整行`ARVALIS_ (1)`对应原图`JPEGImages/ARVALIS_ (1).png`（也支持其他已列出的图片扩展名）与掩码`SegmentationClass/ARVALIS_ (1).png`。不按空格拆字段，不要删除文件名中的空格；只忽略行首尾空白和空行。清单仍不写路径或扩展名，禁止路径分隔符、NUL和重复ID，跨划分泄漏检查保持不变。图片、掩码及原始清单均不会自动重命名。



## 统一按验证集 mIoU 选模

新实验的 best.pth 一律根据验证集 Mean IoU 最高值保存，前景 IoU 仍正常计算和导出。config.json 与检查点记录 selection_metric="Mean IoU"，最佳分数记录为 best_score。相同分数保留更早轮次。

旧检查点评估仍按其原始选模规则标注，不会改写成按 mIoU 选模。旧二分类运行按前景 IoU 选择过最佳权重，不能直接 resume 到新规则：请使用相同骨干预训练权重和新输出目录重新训练。旧多分类运行本就使用 Mean IoU，可继续恢复。旧结果文件不会自动重算或覆盖。


## 温和数据增强

训练保留水平/垂直翻转（各0.5概率）和0/90/180/270度等概率旋转。以0.5概率进行0.8–1.2倍等比例缩放，再随机裁切或随机位置填充回512×512；图像用双线性插值，标签用最近邻插值，新增边缘图像填充RGB(124,116,104)、标签填充背景0。以0.5概率对图像依次施加亮度和对比度扰动，系数各自独立采样自0.9–1.1；不改变标签。不添加色调/饱和度扰动或宽高比拉伸。验证/测试保持固定输入。

具体策略记录在config.json和新检查点的augmentation_config中。改变策略后请使用新输出目录从骨干预训练开始训练；旧运行不能直接resume混入新增强，旧权重仍可评估/推理。更新不重写历史指标、不改数据集或模型参数量。对比模型需要使用相同增强协议后再报告统一条件下的结果。



## 统一指标与结果格式（bimseg-segmentation-v2）

三个仓库使用内容完全一致的 `segmentation_metrics.py`。区域指标先累计整个评估集的混淆矩阵，行=真实类别、列=预测类别；不对逐图 IoU/Dice 求平均。

| 标准字段 | 计算口径 |
|---|---|
| `miou` / `macc` | 全部类别（包含背景）的 IoU / Recall 宏平均；所有任务按验证集 mIoU 选择最佳权重 |
| `foreground_miou` | 仅前景各类别 IoU 的宏平均；二分类即前景 IoU |
| `union_foreground_iou` | 合并所有非背景类别后的 IoU，不能反映前景内部错分 |
| `dice` / `precision` / `recall` | 全数据集统计，前景类别宏平均 |
| `dice_all` | 包含背景的 Dice 宏平均，作为补充，不能与 `dice` 混用 |
| `pixacc` / `fwiou` | 总体像素准确率 / 按真实像素频率加权的 IoU |
| `specificity` | 前景并集的特异度，即背景 Recall |
| `boundary_f1` / `hd95` / `assd` | 逐图逐前景类别计算，先在图内对有效前景类别平均，再对有效图片平均；另存每类结果 |

区域指标分母为零时记录 JSON `null` / CSV 空单元格，宏平均排除该项。边界类别在预测和真实标签中都为空时不参与平均；仅一方为空时 BF1=0、HD95/ASSD=图像对角线。整幅图没有任何有效前景类别对时不参与边界均值；整个数据集都没有时为 null。Boundary F1 默认容差 2 像素，HD95/ASSD 单位为评估尺寸下的像素；其他分数为 0–1。忽略像素不参与统计；将其屏蔽可能在忽略区边缘产生轮廓，比较时必须使用相同有效像素和分辨率。

每次最终评估统一写出：
- `metrics.json`：完整结果、每类指标、混淆矩阵和协议版本。
- `results.csv`：固定字段，每行对应 val 或 test。区域 `precision` 与测速 `precision_mode` 分开，避免覆盖。
- `class_metrics.csv`：每个数据划分、每个类别一行，含 IoU、Dice、Precision、Recall、像素数及边界结果；背景边界列留空。
- `confusion_matrix.csv`：长表 `split,true_class,predicted_class,pixels`。
- 开启复杂度测试时另有 `benchmark.json`：完整网络参数量、GFLOPs、FP32 单图前向延迟/FPS等。

批量仓库的实验根目录 `results.csv` 也使用相同列，每个模型/种子/划分一行，以 `name`、`seed`、`split` 区分，不再使用 `test_` / `val_` 前缀的宽表。模型子目录同样保存上述四个文件。缺失或失败数据留空，失败行记录非零 `status`。

旧 JSON 中的兼容字段暂时保留，标准 CSV 只使用上表字段。`plant-seg-lab` 的 `Foreground IoU` 现在对应 `foreground_miou`，原合并值改称 `Union Foreground IoU`；`nclassbenchmark` 的旧 `mdice` 仍是含背景均值，论文推荐取标准 `dice`（等于 `mdice_fg`）。各仓库的损失函数、模型结构、数据增强和预训练设置不因这次指标统一而改变。

旧权重可以重新评价，无需重新训练；旧 CSV 不会自动变为新口径。使用对应原始训练参数在两个 benchmark 仓库追加 `--eval-only`，或在 plant-seg-lab 使用 `summarize.py` / `evaluate_transfer.py` 重新生成结果。不要混合旧边界指标和 v2 指标。统计代码的验证命令：`python -m unittest discover -s tests -p 'test_metric_*.py' -v`（仅需 NumPy/SciPy，无需 GPU）。
