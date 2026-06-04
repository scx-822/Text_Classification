训练命令：python src\train.py --config configs\roberta_supcon_no_imbalance.yaml

单独用已训练好的最优模型预测测试集：python src\predict.py --checkpoint outputs\roberta_supcon_no_imbalance\checkpoints\best_model.pt --data Dataset\roberta_test.xlsx --output outputs\roberta_supcon_no_imbalance\predictions\manual_predictions.xlsx
