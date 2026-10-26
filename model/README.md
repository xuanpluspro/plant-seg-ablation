# 模型来源与当前结构

## MODA来源

上游：https://github.com/shuaihao-han/MODA
论文：Han等，MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images，AAAI 2026。

fem.py保留上游特征模块，ssaf.py保留原融合单元但将SFA/SDE改为DRM/CBM，LICENSE.MODA为上游Apache-2.0许可证；没有复制数据或权重。RGB版本中的spectral等名称指学习到的通道特征，不代表RGB具有额外光谱波段。

当前RGB ResNet-50提取四层特征，投影至统一通道后，最深层固定使用MPM-5531与稀疏注意力；DRM/CBM融合模块自顶向下融合，真实像素掩膜监督辅助前景引导F*(1+sigmoid(aux))，三个跨尺度动态调制模块自底向上更新，再经多尺度解码器与RGB细节分支输出K类预测。动态融合可关闭，恢复fem.py模块；没有检测头或NMS。

主损失为交叉熵和前景类别Dice，辅助BCE权重0.2，多分类辅助目标为所有前景并集。这不是原检测模型的官方分割复现。

## MPM-5531

mpm.py基于MPCNet官方模块改写，固定四分支5/5/3/1普通卷积。两个5×5分支共享卷积与BN；分支与原输入拼接，经过CBR及CAM。CBR的自定义7×7深度卷积激活保持不变。来源与授权说明见[MPM_SOURCE.md](MPM_SOURCE.md)。融合特征Fm经独立1×1投影生成Q，CAM输出先做逐位置通道LayerNorm，再经1×1投影生成K/V。双Top-k分支保留约1/2与1/6位置，输出经1×1投影加回Fm。8个头，不增加多尺度池化；检查点格式v4。

## 跨尺度动态调制

dynamic_fusion.py受Dino U-Net的FAPM仿射调制启发，采用相邻两层输入；调制条件包含两层投影和归一化差异。细化保留归一化、激活、可分离卷积和SE，捷径来自深层投影。不是原版FAPM的严格复现，新颖性与真实收益需另行验证。使用实验需引用对应来源。

## SACF分支替换

ssaf.py用DRM处理深层、CBM处理浅层；保留尺寸对齐和原双权重融合，删除旧SFA/SDE及未使用的距离/条带卷积变体。来源和差异见[SACF_SOURCE.md](SACF_SOURCE.md)。
