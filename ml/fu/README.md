# FU2609 5 分钟障碍概率模型

这个目录用于训练概率模型，回答下面这个问题：

> 在下一根 1 分钟 K 线开盘价入场后，未来 5 个日历分钟内，
> 先触及顺向 10 跳，还是先触及反向 5 跳？对应概率分别是多少？

流程会训练两个独立的多分类模型：

- `long_label`：做多入场后的结果，取值为 `win`、`loss`、`none`
- `short_label`：做空入场后的结果，取值为 `win`、`loss`、`none`

默认假设：

- 输入数据：`data/1min_FU/FU2609.csv`
- 最小跳动：`1.0`
- 止盈障碍：`10` 跳
- 止损障碍：`5` 跳
- 观察窗口：下一根 K 线开盘入场后的 `5` 个日历分钟
- 如果同一根 1 分钟 K 线内同时触及止盈和止损，默认标记为 `loss`

## 步骤 1：检查数据

```powershell
python ml\fu\01_inspect_data.py
```

该脚本会输出行数、日期范围、缺失值、价格/成交量统计，以及时间间隔分布。

## 步骤 2：构建带标签的数据集

```powershell
python ml\fu\02_build_dataset.py
```

输出文件：

- `ml/fu/output/fu2609_dataset.csv`
- `ml/fu/output/fu2609_dataset.metadata.json`

常用参数：

```powershell
python ml\fu\02_build_dataset.py --tick-size 1 --profit-ticks 10 --loss-ticks 5 --horizon-minutes 5
python ml\fu\02_build_dataset.py --same-bar-policy win
python ml\fu\02_build_dataset.py --drop-any-ambiguous
```

## 步骤 3：训练模型

```powershell
python ml\fu\03_train_model.py
```

输出文件：

- `ml/fu/output/models/fu2609_long_model.joblib`
- `ml/fu/output/models/fu2609_short_model.joblib`
- `ml/fu/output/reports/fu2609_metrics.json`
- `ml/fu/output/reports/fu2609_test_predictions.csv`
- `ml/fu/output/reports/fu2609_long_win_calibration.csv`
- `ml/fu/output/reports/fu2609_short_win_calibration.csv`

训练脚本使用按时间顺序切分的数据集，并在训练集、验证集、测试集边界附近留出少量隔离样本，降低未来标签泄漏风险。
如果本地安装了 LightGBM，会优先使用 LightGBM；否则回退到 sklearn 的 RandomForest。

## 概率转交易期望

测试报告会分别计算多头和空头的期望 ticks：

```text
EV ticks = P(win) * 10 - P(loss) * 5
```

第一版中，`none` 结果的期望 ticks 按 0 处理。报告里还会输出不同期望阈值下的真实平均 ticks，
方便比较模型预测期望和样本外实际表现。

## 特征说明

数据集构建脚本会生成基础 K 线 rolling 特征，以及增强特征：

- 日内/交易段高低点、开盘价位置
- 过去 3/5/15 根 K 线聚合形态
- 过去 20/60 根 K 线高低点突破/假突破
- ATR、波动、成交量、成交额 rolling 分位数
- 价量持仓组合与交互项

修改特征代码后，已有 CSV 不会自动更新，需要重新运行 `02_build_dataset.py` 或 `04_build_multi_contract_dataset.py`。

## 多合约拼接流程

单合约 FU2609 的前期样本很不活跃，容易造成训练集和测试集分布不一致。多合约流程采用下面的方式：

```text
每个合约单独读取
每个合约单独计算 rolling 特征
每个合约单独生成未来 5 分钟三重障碍标签
过滤太短、太不活跃的样本
最后把所有合约样本 concat 到同一个训练表
```

这样不会让 `FU2605` 的最后一分钟和 `FU2606` 的第一分钟连在一起计算特征或标签。

## 步骤 4：构建多合约数据集

默认会扫描 `data/1min_FU/FU*.csv`，跳过少于 5000 行的合约，并过滤当前成交量小于 5、过去 20 分钟平均成交量小于 20 的样本：

```powershell
python ml\fu\04_build_multi_contract_dataset.py
```

输出文件：

- `ml/fu/output/fu_multi_dataset.csv`
- `ml/fu/output/fu_multi_dataset.metadata.json`

如果只想先跑近期几个合约：

```powershell
python ml\fu\04_build_multi_contract_dataset.py --start-contract FU2605 --end-contract FU2609 --output ml\fu\output\fu_multi_dataset_smoke.csv
```

常用参数：

```powershell
python ml\fu\04_build_multi_contract_dataset.py --contract-glob FU26*.csv
python ml\fu\04_build_multi_contract_dataset.py --start-contract FU2401 --end-contract FU2609
python ml\fu\04_build_multi_contract_dataset.py --include-contracts FU2509,FU2601,FU2605,FU2609
python ml\fu\04_build_multi_contract_dataset.py --min-volume 10 --min-volume-ma20 50
python ml\fu\04_build_multi_contract_dataset.py --drop-any-ambiguous
```

## 步骤 5：训练多合约模型

默认使用时间顺序切分，并且默认不使用 `class_weight=balanced`，避免把 `win/loss` 概率过度抬高：

```powershell
python ml\fu\05_train_multi_contract_model.py
```

如果训练的是上面的近期合约烟测数据：

```powershell
python ml\fu\05_train_multi_contract_model.py --dataset ml\fu\output\fu_multi_dataset_smoke.csv --model-dir ml\fu\output\models_multi_smoke --report-dir ml\fu\output\reports_multi_smoke
```

输出文件：

- `ml/fu/output/models_multi/fu_multi_long_model.joblib`
- `ml/fu/output/models_multi/fu_multi_short_model.joblib`
- `ml/fu/output/reports_multi/fu_multi_metrics.json`
- `ml/fu/output/reports_multi/fu_multi_test_predictions.csv`
- `ml/fu/output/reports_multi/fu_multi_long_win_calibration.csv`
- `ml/fu/output/reports_multi/fu_multi_short_win_calibration.csv`
- `ml/fu/output/reports_multi/fu_multi_long_win_calibration_calibrated.csv`
- `ml/fu/output/reports_multi/fu_multi_short_win_calibration_calibrated.csv`
- `ml/fu/output/reports_multi/fu_multi_raw_ev_breakdown.csv`
- `ml/fu/output/reports_multi/fu_multi_calibrated_ev_breakdown.csv`
- `ml/fu/output/reports_multi/fu_multi_ev_breakdown.csv`
- `ml/fu/output/reports_multi/fu_multi_long_feature_importance.csv`
- `ml/fu/output/reports_multi/fu_multi_short_feature_importance.csv`

`fu_multi_ev_breakdown.csv` 会按 `all`、`month`、`contract`、`side`、`month_side`、`contract_side` 输出不同 EV 阈值下的真实表现。`ev_source=raw` 表示原始模型概率，`ev_source=calibrated` 表示使用验证集做 isotonic 概率校准后的结果。

也可以用指定合约做样本外测试，例如用 `FU2609` 做测试集：

```powershell
python ml\fu\05_train_multi_contract_model.py --split-mode contract_holdout --valid-contracts FU2608 --test-contracts FU2609
```

`contract_holdout` 只是在合约维度留出测试合约，训练集里仍可能包含相同日历时间的其他合约样本；如果你要模拟“未来时间”的样本外效果，优先使用默认的 `time_ratio` 切分。

完整多合约数据集可能比较大。建议先用 `--start-contract`、`--end-contract` 或 `--include-contracts` 跑小范围，确认标签分布和模型效果后再扩大范围。
