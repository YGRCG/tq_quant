# FU 每日主力合约样本

这个目录构建一个简单的“每日主力样本”版本：

```text
读取原始 data/1min_FU/FU*.csv
每个合约独立计算增强特征和未来 5 分钟标签
每个交易日选日内 filter_volume 总和最大的合约
只保留当天这个主力合约的样本
用这个样本训练 long/short 概率模型
```

它不是传统换月复权连续合约，不拼接价格；每条样本仍然来自原始合约自己的特征和标签。这样可以避免换月价差污染，同时减少每分钟频繁切换合约。

## 步骤 1：构建每日主力增强特征样本

默认范围是 `FU2401` 到 `FU2609`：

```powershell
python ml\fu_main_minute\01_build_main_minute_dataset.py
```

输出：

```text
ml/fu_main_minute/output/fu_main_minute_dataset.csv
ml/fu_main_minute/output/fu_main_minute_dataset.metadata.json
```

默认选择规则：

```text
同一个 trade_date 内，选日内 filter_volume 总和最大的合约
当天只保留这个合约的样本
```

常用参数：

```powershell
python ml\fu_main_minute\01_build_main_minute_dataset.py --start-contract FU2401 --end-contract FU2609
python ml\fu_main_minute\01_build_main_minute_dataset.py --include-contracts FU2509,FU2601,FU2605,FU2609
python ml\fu_main_minute\01_build_main_minute_dataset.py --min-volume 10 --min-volume-ma20 50
python ml\fu_main_minute\01_build_main_minute_dataset.py --profit-ticks 10 --loss-ticks 5 --horizon-minutes 5
```

如果要复现旧的“每分钟最大成交量合约”版本：

```powershell
python ml\fu_main_minute\01_build_main_minute_dataset.py --main-mode minute
```

如果传入已有多合约数据集，脚本只做主力样本筛选，不重新计算特征：

```powershell
python ml\fu_main_minute\01_build_main_minute_dataset.py --input ml\fu\output\fu_multi_dataset_2401_2609.csv
```

## 步骤 2：训练模型

```powershell
python ml\fu_main_minute\02_train_model.py
```

输出：

```text
ml/fu_main_minute/output/models/fu_main_minute_long_model.joblib
ml/fu_main_minute/output/models/fu_main_minute_short_model.joblib
ml/fu_main_minute/output/reports/fu_main_minute_metrics.json
ml/fu_main_minute/output/reports/fu_main_minute_test_predictions.csv
```

训练使用时间顺序切分：

```text
前 70% 时间：训练
中间 15% 时间：验证
最后 15% 时间：测试
```
