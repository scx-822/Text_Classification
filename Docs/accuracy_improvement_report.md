# 准确率提升诊断报告

## 当前结论

当前保留的实验集中在 `RoBERTa + SupCon + CLS` 方案及其验证集超参数搜索。测试集标签同步后，最高干净测试准确率目前为 83.57%。当前最优干净方案是同时联合调整 `learning_rate`、`lambda_contrast`、`temperature`、`dropout`、`gradient_accumulation_steps` 和 `max_length` 后得到的 RoBERTa 方案。

## 已完成实验

| 实验                                                | 是否干净 | 测试准确率 | Macro-F1 | 输出目录                                        |
| --------------------------------------------------- | -------- | ---------: | -------: | ----------------------------------------------- |
| RoBERTa + SupCon + CLS                              | 是       |     80.00% |   78.82% | `outputs/roberta_supcon_cls`                  |
| RoBERTa + SupCon + CLS，验证集调参                  | 是       |     80.71% |   79.34% | `outputs/roberta_supcon_cls_tuned`            |
| RoBERTa + SupCon + CLS，验证集调参 + max_length=192 | 是       |     82.14% |   80.72% | `outputs/roberta_supcon_cls_max_length_tuned` |
| RoBERTa + SupCon + CLS，联合调参                    | 是       |     83.57% |   82.66% | `outputs/roberta_supcon_cls_joint_tuned`      |

说明：当前表格只保留仍在项目中的实验结果。其中 `RoBERTa + SupCon + CLS，联合调参` 已在测试集标签同步后重新训练和评估；前三个历史实验结果暂未针对同步后的测试集逐一重跑。

## 测试集标签同步记录

本次测试集更新后，发现 `Dataset/roberta_test.xlsx` 中存在 1 条 `label` 与 `label_id` 不一致的记录：

| source_excel_row | 文本 | 更新前 label | 更新前 label_id | 同步后 label_id |
| ---: | --- | --- | ---: | ---: |
| 22 | `部分灭火器过期、失效。` | `安全` | 2 | 0 |

训练集中的标签映射为：

| label_id | label |
| ---: | --- |
| 0 | 安全 |
| 1 | 质量 |
| 2 | 其他 |

评估脚本实际使用 `label_id` 计算指标。因此，如果只修改 `label` 文本而不同步 `label_id`，测试指标不会按新标签生效。已将原测试集备份为 `Dataset/roberta_test.before_label_id_sync.xlsx`，并把当前测试集同步为：

| 类别 | label_id | 测试集数量 |
| --- | ---: | ---: |
| 安全 | 0 | 24 |
| 质量 | 1 | 53 |
| 其他 | 2 | 63 |

## RoBERTa 超参数搜索

本次只对 `RoBERTa + SupCon + CLS` 做验证集搜索，搜索阶段只读取训练集，并从训练集中切分验证集，没有读取测试集。

搜索空间：

| 参数                            | 搜索值                                   |
| ------------------------------- | ---------------------------------------- |
| `learning_rate`               | `1e-5`, `1.5e-5`, `2e-5`, `3e-5` |
| `lambda_contrast`             | `0.05`, `0.1`, `0.2`               |
| `temperature`                 | `0.07`, `0.1`                        |
| `dropout`                     | `0.1`                                  |
| `gradient_accumulation_steps` | `2`                                    |

验证集最优组合为：

```yaml
learning_rate: 3.0e-5
lambda_contrast: 0.1
temperature: 0.07
dropout: 0.1
gradient_accumulation_steps: 2
```

该组合验证集准确率为 81.48%，高于原始 RoBERTa 配置的 79.63%；正式测试准确率为 80.71%，高于原始 RoBERTa 配置的 80.00%。

随后固定上述最优组合，只对 `max_length` 做单变量搜索。搜索阶段仍然只读取训练集，并从训练集中切分验证集，没有读取测试集。

搜索空间：

| 参数           | 搜索值                                     |
| -------------- | ------------------------------------------ |
| `max_length` | `96`, `128`, `192`, `256`, `320` |

验证集结果：

| max_length | Best Epoch | Valid Accuracy | Valid Macro-F1 |
| ---------: | ---------: | -------------: | -------------: |
|         96 |          7 |         83.33% |         83.17% |
|        128 |          5 |         83.33% |         83.30% |
|        192 |          8 |         87.04% |         86.85% |
|        256 |          5 |         81.48% |         81.51% |
|        320 |          5 |         83.33% |         83.38% |

验证集最优 `max_length` 为 `192`。使用该配置进行最终训练和测试后，正式测试准确率为 82.14%，Macro-F1 为 80.72%。

在此基础上，又做了一次紧凑联合搜索，同时调整 `learning_rate`、`lambda_contrast`、`temperature`、`dropout`、`gradient_accumulation_steps` 和 `max_length`。搜索阶段仍然只读取训练集，并从训练集中切分验证集，没有读取测试集。

联合搜索空间：

| 参数                            | 搜索值             |
| ------------------------------- | ------------------ |
| `learning_rate`               | `2e-5`, `3e-5` |
| `lambda_contrast`             | `0.1`, `0.2`   |
| `temperature`                 | `0.07`, `0.1`  |
| `dropout`                     | `0.1`, `0.2`   |
| `gradient_accumulation_steps` | `1`, `2`       |
| `max_length`                  | `192`, `256`   |

本次联合搜索共 64 组组合，验证集最优组合为：

```yaml
learning_rate: 3.0e-5
lambda_contrast: 0.1
temperature: 0.07
dropout: 0.1
gradient_accumulation_steps: 1
max_length: 192
```

该组合验证集准确率为 87.04%，验证集 Macro-F1 为 86.97%。测试集标签同步后，使用该配置重新训练和测试，正式测试准确率为 83.57%，Macro-F1 为 82.66%，Weighted-F1 为 83.65%，是当前保留实验中的最高干净结果。

同步后测试集混淆矩阵如下，行是真实类别，列是预测类别：

|  | 预测安全 | 预测质量 | 预测其他 |
| --- | ---: | ---: | ---: |
| 真实安全 | 21 | 1 | 2 |
| 真实质量 | 1 | 47 | 5 |
| 真实其他 | 8 | 6 | 49 |

## 为什么 92% 上不去

主要问题不是类别不平衡。当前训练集是 90/90/90，三类完全平衡。核心问题是训练集和测试集的标注口径存在冲突，尤其是“其他”和“安全/质量”的边界。

典型证据：

| 关键词/模式 | 训练集标签分布        | 测试集冲突样本                                                     |
| ----------- | --------------------- | ------------------------------------------------------------------ |
| `配电箱`  | 6/6 为 `安全`       | `工地试验室，混凝土室内配电箱、开关箱破损未修复。` 标为 `其他` |
| `警示`    | 4/4 为 `安全`       | `施工场地未设警示围护。` 标为 `其他`                           |
| `标识`    | 训练集大多为 `其他` | `洞内围岩监控量测测点，未设标识牌。` 标为 `安全`               |

其中 `部分灭火器过期、失效。` 已从 `其他` 同步修正为 `安全`，因此不再作为标注冲突证据。其余边界样本仍会导致模型出现“按训练集学是对的，按测试集被判错”的情况。只靠换模型或调损失，很难解决这种标注口径冲突。

## 已新增文件

- `configs/roberta_supcon_cls_tuned.yaml`：RoBERTa + SupCon + CLS 验证集调参后的配置。
- `configs/roberta_supcon_cls_max_length_tuned.yaml`：RoBERTa + SupCon + CLS 继续调整 `max_length` 后的配置。
- `configs/roberta_supcon_cls_joint_tuned.yaml`：RoBERTa + SupCon + CLS 联合调参后的配置。
- `src/tune_valid.py`：只使用训练集/验证集的超参数搜索脚本，支持默认搜索、`max_length_only` 搜索和 `joint_compact` 联合搜索。

## 后续要达到 92% 需要的条件

1. 明确三类标注规则，尤其是 `其他` 与 `安全/质量` 的边界。
2. 对当前测试错例中类似模式补充训练样本，例如“配电箱/警示围护但标为其他、标识类样本边界不一致”的样本。
3. 如果业务上确实存在固定口径，应把口径写成规则后处理，并在验证集上验证，而不是直接按测试错例硬编码。
4. 如果要继续引入 `location` 规则，应该先调整验证集构造，让 low/medium/high 地点等级在验证集中都有足够样本，再决定是否保留地点分支。
