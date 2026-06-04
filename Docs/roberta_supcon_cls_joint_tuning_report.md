# RoBERTa + SupCon + CLS 联合调参实验汇报

## 1. 实验目的

本实验面向三分类文本分类任务，类别包括 `安全`、`质量` 和 `其他`。在前期 `RoBERTa + SupCon + CLS` 实验基础上，本次重点验证：将 `learning_rate`、`lambda_contrast`、`temperature`、`dropout`、`gradient_accumulation_steps` 和 `max_length` 同时纳入搜索空间后，是否能够进一步提升模型效果。

本次调参遵守干净实验原则：调参阶段只使用训练集，并从训练集中划分验证集；测试集只在最终确定超参数后进行一次评估，不参与选参。

## 2. 数据划分

数据来自 `Dataset/roberta_train.xlsx` 和 `Dataset/roberta_test.xlsx`。

| 数据部分 | 样本数 | 安全 | 质量 | 其他 |
| --- | ---: | ---: | ---: | ---: |
| 训练源数据 | 270 | 90 | 90 | 90 |
| 实际训练集 | 216 | 72 | 72 | 72 |
| 验证集 | 54 | 18 | 18 | 18 |
| 测试集 | 140 | 23 | 53 | 64 |

划分方式为 `stratified_group_kfold`。其中 `leakage_group_id` 在训练集和验证集之间没有重叠，用于降低相似样本同时出现在训练和验证中的风险。

文本长度统计如下：

| 数据部分 | min | p50 | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练源数据 | 8 | 38.0 | 82.0 | 110.85 | 272 |
| 测试集 | 11 | 38.0 | 97.4 | 121.05 | 150 |

## 3. 模型方法

本实验使用 `RoBERTa + SupCon + CLS` 结构：

```text
文本输入
  |
Tokenizer
  |
Chinese RoBERTa Encoder
  |
CLS 向量表示 h
  |----------------------------|
  |                            |
分类头                         Projection Head
  |                            |
分类 logits                    对比学习向量 z
  |                            |
分类损失 L_cls                 对比损失 L_contrast
  |----------------------------|
                |
          总损失 L_total
```

其中：

- 基座模型：`models/hfl_chinese_roberta_wwm_ext`
- 原始模型来源：`hfl/chinese-roberta-wwm-ext`
- 句向量表示：取 RoBERTa 的 `CLS` 向量
- 分类损失：`CrossEntropyLoss`
- 对比损失：`Supervised Contrastive Loss`
- 对比学习向量维度：`128`

总损失函数为：

```text
L_total = L_cls + lambda_contrast * L_contrast
```

`lambda_contrast` 控制对比损失在总损失中的权重；`temperature` 控制对比学习中相似度分布的尖锐程度。

## 4. 联合调参设计

前期实验曾分别调过普通超参数和 `max_length`。本次将多个关键参数放到同一个搜索空间中联合搜索，避免只看单个参数时忽略参数之间的相互影响。

调参脚本：

```powershell
python src\tune_valid.py --config configs\roberta_supcon_cls.yaml --output-dir outputs\roberta_supcon_cls_joint_tuning --search-mode joint_compact
```

搜索空间如下：

| 参数 | 候选值 | 作用 |
| --- | --- | --- |
| `learning_rate` | `2e-5`, `3e-5` | 控制模型参数更新步长 |
| `lambda_contrast` | `0.1`, `0.2` | 控制对比损失权重 |
| `temperature` | `0.07`, `0.1` | 控制 SupCon 相似度分布 |
| `dropout` | `0.1`, `0.2` | 控制正则化强度 |
| `gradient_accumulation_steps` | `1`, `2` | 控制等效 batch 更新频率 |
| `max_length` | `192`, `256` | 控制文本最大截断长度 |

本次为紧凑网格搜索，共 `2 * 2 * 2 * 2 * 2 * 2 = 64` 组组合。主验证指标为 `accuracy`，同时记录 `Macro-F1`。

## 5. 最优超参数

验证集最优 trial 为 `trial_033`，对应参数如下：

| 参数 | 最优取值 |
| --- | ---: |
| `learning_rate` | `3e-5` |
| `lambda_contrast` | `0.1` |
| `temperature` | `0.07` |
| `dropout` | `0.1` |
| `gradient_accumulation_steps` | `1` |
| `max_length` | `192` |

最终训练配置如下：

| 配置项 | 取值 |
| --- | --- |
| 配置文件 | `configs/roberta_supcon_cls_joint_tuned.yaml` |
| 输出目录 | `outputs/roberta_supcon_cls_joint_tuned` |
| `pooling` | `cls` |
| `batch_size` | `16` |
| `eval_batch_size` | `32` |
| `epochs` | `8` |
| `warmup_ratio` | `0.1` |
| `weight_decay` | `0.01` |
| `max_grad_norm` | `1.0` |
| `early_stopping_patience` | `2` |
| `seed` | `42` |

## 6. 验证集结果

最终模型在第 `4` 轮达到验证集最优：

| Epoch | Train Loss | Valid Loss | Valid Accuracy | Valid Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.3385 | 1.1432 | 72.22% | 71.96% |
| 2 | 0.8242 | 0.9502 | 79.63% | 79.39% |
| 3 | 0.4807 | 0.8865 | 77.78% | 77.07% |
| 4 | 0.2812 | 0.8410 | 87.04% | 86.97% |
| 5 | 0.2114 | 1.0417 | 85.19% | 85.10% |
| 6 | 0.1624 | 1.3062 | 77.78% | 75.77% |

验证集最优结果：

| 指标 | 结果 |
| --- | ---: |
| Best Epoch | 4 |
| Valid Accuracy | 87.04% |
| Valid Macro-F1 | 86.97% |

## 7. 测试集结果

最终确定超参数后，使用以下命令进行训练和测试：

```powershell
python src\train.py --config configs\roberta_supcon_cls_joint_tuned.yaml
```

测试集整体指标：

| 指标 | 结果 |
| --- | ---: |
| Test Accuracy | 82.86% |
| Test Macro-F1 | 81.66% |
| Test Weighted-F1 | 82.99% |
| Test Loss | 0.8065 |
| Test Classification Loss | 0.4568 |
| Test Contrastive Loss | 3.4972 |

测试集每类指标：

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 安全 | 66.67% | 86.96% | 75.47% | 23 |
| 质量 | 87.04% | 88.68% | 87.85% | 53 |
| 其他 | 87.50% | 76.56% | 81.67% | 64 |

混淆矩阵如下，行表示真实类别，列表示预测类别：

|  | 预测安全 | 预测质量 | 预测其他 |
| --- | ---: | ---: | ---: |
| 真实安全 | 20 | 1 | 2 |
| 真实质量 | 1 | 47 | 5 |
| 真实其他 | 9 | 6 | 49 |

## 8. 与前期实验对比

| 实验 | Test Accuracy | Test Macro-F1 | 输出目录 |
| --- | ---: | ---: | --- |
| RoBERTa + SupCon + CLS 原始参数 | 80.00% | 78.82% | `outputs/roberta_supcon_cls` |
| RoBERTa + SupCon + CLS，验证集调参 | 80.71% | 79.34% | `outputs/roberta_supcon_cls_tuned` |
| RoBERTa + SupCon + CLS，验证集调参 + max_length=192 | 82.14% | 80.72% | `outputs/roberta_supcon_cls_max_length_tuned` |
| RoBERTa + SupCon + CLS，联合调参 | 82.86% | 81.66% | `outputs/roberta_supcon_cls_joint_tuned` |

联合调参相比原始参数，测试准确率提升 `2.86` 个百分点；相比单独调整 `max_length=192`，测试准确率提升 `0.72` 个百分点。

## 9. 结论

本次联合调参证明多个超参数可以同时搜索，且确实比单独调 `max_length` 略有提升。当前最优干净结果为：

```text
RoBERTa + SupCon + CLS，联合调参
Test Accuracy = 82.86%
Test Macro-F1 = 81.66%
```

从结果看，`max_length=192`、`learning_rate=3e-5`、`temperature=0.07` 对当前数据更合适；`lambda_contrast=0.1` 比更大的对比损失权重更稳。当前模型仍未达到 92%，主要瓶颈可能不再只是超参数，而是训练集和测试集之间部分样本的标注口径差异，尤其是 `安全` 与 `其他` 的边界。

## 10. 输出文件

- 调参记录：`outputs/roberta_supcon_cls_joint_tuning/logs/tuning_summary.json`
- 最终配置：`configs/roberta_supcon_cls_joint_tuned.yaml`
- 最优模型：`outputs/roberta_supcon_cls_joint_tuned/checkpoints/best_model.pt`
- 训练历史：`outputs/roberta_supcon_cls_joint_tuned/logs/training_history.json`
- 测试指标：`outputs/roberta_supcon_cls_joint_tuned/logs/test_metrics.json`
- 测试预测：`outputs/roberta_supcon_cls_joint_tuned/predictions/roberta_supcon_test_predictions.csv`
- 测试预测 Excel：`outputs/roberta_supcon_cls_joint_tuned/predictions/roberta_supcon_test_predictions.xlsx`
