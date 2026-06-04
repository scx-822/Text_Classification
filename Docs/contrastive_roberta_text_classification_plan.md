# 基于中文 RoBERTa 与监督式对比学习的文本分类方案

## 1. 任务设定

本任务是中文文本分类任务，数据文件为：

- `Dataset/roberta_train.xlsx`
- `Dataset/roberta_test.xlsx`

当前读取到的核心字段如下：

```text
text
label
label_id
orig_id
check_unit
location
responsible_unit
source_excel_row
leakage_group_id
```

建模使用字段：

- 输入文本：`text`
- 分类标签：`label_id`
- 标签名称：`label`
- 分组字段：`leakage_group_id`

训练集共有 270 条样本，测试集共有 140 条样本。类别为：

| 类别 | label_id |
| ---- | -------- |
| 安全 | 0        |
| 质量 | 1        |
| 其他 | 2        |

训练集类别分布：

| 类别 | label_id | 样本数 |   占比 |
| ---- | -------- | -----: | -----: |
| 安全 | 0        |     90 | 33.33% |
| 质量 | 1        |     90 | 33.33% |
| 其他 | 2        |     90 | 33.33% |

测试集类别分布：

| 类别 | label_id | 样本数 |   占比 |
| ---- | -------- | -----: | -----: |
| 安全 | 0        |     23 | 16.43% |
| 质量 | 1        |     53 | 37.86% |
| 其他 | 2        |     64 | 45.71% |

文本长度统计：

| 数据集 | 最短 | P50 | P90 | P95 | 最长 |
| ------ | ---: | --: | --: | --: | ---: |
| 训练集 |    8 |  38 |  82 | 111 |  272 |
| 测试集 |   11 |  38 |  97 | 121 |  150 |

`leakage_group_id` 统计：

- 训练集：117 个 group，组大小从 1 到 12 不等；
- 测试集：36 个 group，组大小从 1 到 16 不等。

当前训练集已经是三类均衡数据。本版本继续不考虑类别不平衡处理，分类损失直接使用普通交叉熵。

## 2. 总体方法

采用：

```text
中文预训练模型 + 分类头 + Projection Head + 监督式对比学习
```

模型同时学习两个目标：

1. 分类目标：预测文本属于 `安全`、`质量`、`其他` 哪一类。
2. 对比学习目标：让同类文本的向量更接近，不同类文本的向量更远。

最终损失为：

$$
L_{\text{total}}
=
L_{\text{cls}}
+
\lambda_{\text{contrast}} L_{\text{contrast}}
$$

其中：

- $L_{\text{cls}}$ 是普通交叉熵分类损失；
- $L_{\text{contrast}}$ 是监督式对比学习损失；
- $\lambda_{\text{contrast}}$ 是对比学习损失权重。

## 3. 基座模型

第一版推荐使用：

```text
hfl/chinese-roberta-wwm-ext
```

可选增强模型：

```text
hfl/chinese-macbert-base
hfl/chinese-roberta-wwm-ext-large
hfl/chinese-macbert-large
```

推荐顺序：

1. 先使用 `hfl/chinese-roberta-wwm-ext` 跑通完整流程。
2. 如果验证集指标不理想，再替换为 `hfl/chinese-macbert-base`。
3. 如果显存充足，再尝试 large 版本。

## 4. 数据划分

测试集不参与训练和调参。验证集从训练集中划分。

推荐：

```text
validation_split = 0.2
random_seed = 42
```

按当前 270 条训练数据划分，20% 验证集约为 54 条。理想情况下验证集每类约 18 条：

```text
安全：约 18 条
质量：约 18 条
其他：约 18 条
```

由于数据中存在 `leakage_group_id`，建议优先使用 group-aware split，避免同一个 group 中高度相似的样本同时出现在训练集和验证集中。

推荐优先级：

1. 优先使用 `leakage_group_id` 做 group-aware split。
2. 如果 group-aware split 导致验证集类别缺失，则使用普通 stratified split。
3. 输出训练集和验证集的类别分布、group 重叠检查结果。

说明：这里使用 stratified split 只是为了保证验证集中每个类别都出现，不属于类别不平衡处理。

## 5. 模型结构

整体结构如下：

```text
文本输入
  |
Tokenizer
  |
Chinese RoBERTa Encoder
  |
句向量表示 h
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

句向量表示使用 `[CLS]` hidden state：

```text
h = last_hidden_state[:, 0, :]
```

分类头：

```text
Dropout
Linear(hidden_size, num_labels)
```

如果使用 RoBERTa base：

```text
hidden_size = 768
num_labels = 3
```

因此分类头输出：

```text
logits = [安全分数, 质量分数, 其他分数]
```

Projection Head：

```text
Linear(hidden_size, hidden_size)
GELU
Linear(hidden_size, contrastive_dim)
L2 Normalize
```

推荐：

```text
contrastive_dim = 128
```

## 6. Supervised Contrastive Learning

监督式对比学习使用标签构造正负样本关系。

对于一个 batch 中的样本 `i`：

- 正样本：batch 内与 `i` 标签相同的其他样本；
- 负样本：batch 内与 `i` 标签不同的样本。

例如：

```text
样本1：安全
样本2：安全
样本3：质量
样本4：其他
```

对样本1来说：

```text
正样本：样本2
负样本：样本3、样本4
```

SupCon 损失为：

$$
L_{\text{contrast}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
-
\frac{1}{|P(i)|}
\sum_{j \in P(i)}
\log
\frac{
\exp(\operatorname{sim}(z_i, z_j) / \tau)
}{
\sum_{k=1}^{N}
\exp(\operatorname{sim}(z_i, z_k) / \tau)
}
\right]
$$

其中：

- $N$ 是 batch size；
- $z_i$ 是第 `i` 个样本经过 Projection Head 后的向量；
- $P(i)$ 是样本 `i` 的正样本集合；
- $\operatorname{sim}$ 使用 cosine similarity；
- $\tau$ 是温度系数。

推荐：

```text
temperature = 0.1
```

如果某个样本在当前 batch 内没有同类正样本，则该样本不参与 SupCon anchor 计算。为了让对比学习有效，batch size 不宜太小。

## 7. 损失函数

分类损失使用普通交叉熵：

$$
L_{\text{cls}}
=
\operatorname{CrossEntropyLoss}(\text{logits}, \text{label\_id})
$$

总损失：

$$
L_{\text{total}}
=
L_{\text{cls}}
+
\lambda_{\text{contrast}} L_{\text{contrast}}
$$

推荐初始值：

```text
lambda_contrast = 0.1
```

可搜索：

```text
lambda_contrast in [0.05, 0.1, 0.2, 0.5]
```

如果加入对比学习后验证集分类指标下降，优先降低 `lambda_contrast`。如果几乎没有变化，可以尝试增大 `lambda_contrast` 或增大 batch size。

## 8. Batch 构造

虽然本版本不考虑类别不平衡，但 SupCon 仍然需要 batch 内存在同类样本，否则正样本不足。

推荐：

```text
batch_size = 16 或 24
```

对于当前 3 分类任务，可以使用普通 shuffle。训练时建议统计每个 batch 中有效 SupCon anchor 数量，如果有效 anchor 太少，再考虑简单的 batch 约束：

```text
每个 batch 尽量包含每个类别至少 2 条样本
```

这不是为了处理类别不平衡，而是为了让监督式对比学习有足够正样本。

## 9. 训练流程

1. 读取 `roberta_train.xlsx` 和 `roberta_test.xlsx`。
2. 使用 `text` 作为输入，`label_id` 作为标签。
3. 从训练集中划分训练/验证集。
4. 加载 tokenizer 和中文 RoBERTa。
5. 前向传播得到句向量 `h`。
6. `h` 输入分类头，得到分类 logits。
7. `h` 输入 Projection Head，得到对比学习向量 `z`。
8. 使用 logits 和 `label_id` 计算普通交叉熵。
9. 使用 batch 内标签关系计算 SupCon Loss。
10. 合并损失，反向传播更新参数。
11. 在验证集上评估并保存最优 checkpoint。
12. 使用最优 checkpoint 对测试集预测或评估。

## 10. 参数更新范围

总损失会更新三部分参数：

```text
RoBERTa Encoder
分类头
Projection Head
```

更新路径：

```text
L_cls -> 分类头 -> RoBERTa Encoder
L_contrast -> Projection Head -> RoBERTa Encoder
```

因此：

- RoBERTa Encoder 同时被分类损失和对比损失更新；
- 分类头只被分类损失更新；
- Projection Head 只被对比损失更新。

Tokenizer、`lambda_contrast`、`temperature`、Dropout、GELU、L2 Normalize 不作为训练参数更新。

## 11. 推荐初始配置

```yaml
model_name: hfl/chinese-roberta-wwm-ext
max_length: 256
pooling: cls
batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 2.0e-5
epochs: 8
warmup_ratio: 0.1
weight_decay: 0.01
dropout: 0.1
classification_loss: cross_entropy
contrastive_loss: supervised_contrastive
contrastive_dim: 128
temperature: 0.1
lambda_contrast: 0.1
validation_split: 0.2
split_strategy: group_aware_split_if_feasible
main_metric: accuracy
aux_metrics:
  - macro_f1
  - per_class_f1
early_stopping_patience: 2
seed: 42
```

文本长度统计显示训练集 P95 约为 111，最长为 272。因此：

- `max_length=128` 速度更快，但可能截断少量长文本；
- `max_length=256` 更稳，推荐第一版使用；
- `max_length=384` 可作为对照实验。

## 12. 评估指标

本版本不围绕类别不平衡设计，因此主指标可以使用：

```text
accuracy
```

同时建议保留辅助指标：

- macro-F1；
- 每类 precision；
- 每类 recall；
- 每类 F1；
- confusion matrix。

保留这些辅助指标不是为了处理类别不平衡，而是为了更清楚地观察模型在哪个类别上容易出错。

## 13. 消融实验

建议至少做三组实验：

| 实验          | 方法                            | 目的                   |
| ------------- | ------------------------------- | ---------------------- |
| Baseline      | RoBERTa + CrossEntropy          | 基础分类性能           |
| SupCon        | RoBERTa + CrossEntropy + SupCon | 验证对比学习收益       |
| Lambda Search | 不同 `lambda_contrast`        | 找到合适的对比损失权重 |

如果时间允许，再做 backbone 对比：

| 实验    | Backbone                        |
| ------- | ------------------------------- |
| RoBERTa | `hfl/chinese-roberta-wwm-ext` |
| MacBERT | `hfl/chinese-macbert-base`    |

## 14. 第一版结论

第一版建议采用：

```text
hfl/chinese-roberta-wwm-ext
+ CrossEntropyLoss
+ Supervised Contrastive Loss
+ group-aware validation split if feasible
```

核心公式：

$$
L_{\text{total}}
=
L_{\text{CE}}
+
\lambda_{\text{contrast}} L_{\text{SupCon}}
$$

推荐初始参数：

```text
max_length = 256
batch_size = 16
gradient_accumulation_steps = 2
temperature = 0.1
lambda_contrast = 0.1
validation_split = 0.2
```

该版本专注于验证“监督式对比学习是否能提升中文 RoBERTa 文本分类效果”，不引入任何类别不平衡处理策略。
