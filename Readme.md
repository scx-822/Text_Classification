# Text Classification

本项目用于中文文本分类实验，当前主方案为 **RoBERTa + Supervised Contrastive Learning + CLS pooling**。

## 项目结构

```text
Dataset/    训练集和测试集
configs/    实验配置文件
Docs/       实验方案、调参和结果说明
models/     本地预训练模型配置、词表和 tokenizer 文件
src/        训练、预测、调参和模型代码
outputs/    本地训练输出，不上传 GitHub
```

## 数据格式

训练集和测试集默认路径：

```text
Dataset/roberta_train.xlsx
Dataset/roberta_test.xlsx
```

数据表至少需要包含：

```text
text      文本内容
label     类别名称
label_id  类别编号
```

如果存在 `leakage_group_id`，训练脚本会优先使用分组感知的训练/验证集划分，减少相似样本泄露。

## 主实验配置

当前主配置文件：

```text
configs/roberta_supcon_cls_joint_tuned.yaml
```

核心设置：

```text
model_name: models/hfl_chinese_roberta_wwm_ext
pooling: cls
max_length: 192
batch_size: 16
learning_rate: 3.0e-05
epochs: 8
temperature: 0.07
lambda_contrast: 0.1
validation_split: 0.2
```

总损失为：

```text
L_total = L_cls + lambda_contrast * L_supcon
```

其中 `L_cls` 是交叉熵分类损失，`L_supcon` 是监督对比学习损失。

## 模型权重说明

GitHub 不允许上传超过 100MB 的单个文件，因此以下大文件没有上传：

```text
models/**/pytorch_model.bin
outputs/
*.pt
*.pth
*.ckpt
*.safetensors
```

运行训练前，需要把本地预训练模型权重放到对应目录，例如：

```text
models/hfl_chinese_roberta_wwm_ext/pytorch_model.bin
```

训练产生的 checkpoint 会保存在 `outputs/` 下，但该目录只保留在本地，不进入 GitHub。

## 训练

```powershell
python src\train.py --config configs\roberta_supcon_cls_joint_tuned.yaml
```

训练完成后，最佳模型默认保存到：

```text
outputs/roberta_supcon_cls_joint_tuned/checkpoints/best_model.pt
```

## 预测测试集

```powershell
python src\predict.py `
  --checkpoint outputs\roberta_supcon_cls_joint_tuned\checkpoints\best_model.pt `
  --data Dataset\roberta_test.xlsx `
  --output outputs\roberta_supcon_cls_joint_tuned\predictions\manual_predictions.xlsx
```

## 验证集联合调参

联合调参脚本：

```powershell
python src\tune_valid.py --config configs\roberta_supcon_cls_joint_tuned.yaml
```

本项目曾联合调整：

```text
learning_rate
lambda_contrast
temperature
dropout
gradient_accumulation_steps
max_length
```

调参结果和说明见 `Docs/`。
