import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 保持学术图表字体

# ==========================================
# 1. 路径配置 (请替换为你的实际路径)
# ==========================================
MODEL_PATH = r"I:\pinghemiyou\cailiao\train\MCSC\qt\extract\train_noclear\best_yield_model.pkl"
NEW_DATA_PATH = r"I:\pinghemiyou\cailiao\train\MCSC\qt\extract\train_clear\Cleaned_Traits_Report_No_Outliers.csv"
OUTPUT_CSV_PATH = r"I:\pinghemiyou\cailiao\train\MCSC\qt\extract\train_noclear\inference_clear\predicted_yield_result.csv"
OUTPUT_PLOT_PATH = r"I:\pinghemiyou\cailiao\train\MCSC\qt\extract\train_noclear\inference_clear\predicted_yield_scatter.png"

# 如果你的新数据中包含真实的产量标签，请填写列名。
# 如果是未知标签的新数据进行盲测，请保持不变，代码会自动跳过指标计算。
TRUE_LABEL_COL = 'yield'


def main():
    # ==========================================
    # 2. 加载模型与提取特征
    # ==========================================
    print("正在加载训练好的模型权重...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件: {MODEL_PATH}")
        return

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, "steps") and hasattr(model.steps[-1][1], "feature_names_in_"):
        expected_features = list(model.steps[-1][1].feature_names_in_)
    else:
        raise ValueError("无法提取特征名称，请确保模型支持 feature_names_in_。")

    # ==========================================
    # 3. 加载与预处理推理数据
    # ==========================================
    print(f"\n正在加载待预测数据: {NEW_DATA_PATH}")
    df_new = pd.read_csv(NEW_DATA_PATH, encoding='gbk')

    missing_features = [f for f in expected_features if f not in df_new.columns]
    if missing_features:
        raise ValueError(f"数据维度不匹配！缺失以下特征: \n{missing_features}")

    X_new = df_new[expected_features]
    if X_new.isnull().values.any():
        print("检测到缺失值，正在使用均值填充...")
        X_new = X_new.fillna(X_new.mean())

    # ==========================================
    # 4. 执行预测
    # ==========================================
    print("\n开始进行产量预测...")
    predictions = model.predict(X_new)
    df_new['predicted_yield'] = predictions

    # ==========================================
    # 5. 指标计算与可视化 (当存在真实标签时)
    # ==========================================
    if TRUE_LABEL_COL in df_new.columns and not df_new[TRUE_LABEL_COL].isnull().all():
        print("\n--- 检测到真实产量数据，开始计算评估指标 ---")
        y_true = df_new[TRUE_LABEL_COL]
        y_pred = df_new['predicted_yield']

        # 计算指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"推理集 R²   : {r2:.4f}")
        print(f"推理集 RMSE : {rmse:.4f}")
        print(f"推理集 MAE  : {mae:.4f}")

        # 绘制散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.7, color='dodgerblue', edgecolors='k', s=60, label='Predictions')

        # 绘制 y=x 完美预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')

        plt.title('Actual vs Predicted Yield (Inference Set)', fontsize=16)
        plt.xlabel('Actual Yield (kg/tree)', fontsize=20)
        plt.ylabel('Predicted Yield (kg/tree)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # 在图表上添加指标文本框
        textstr = f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20,
                 verticalalignment='top', bbox=props)

        plt.legend(loc='lower right', fontsize=18)
        plt.tight_layout()

        # 确保保存目录存在
        import os
        os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
        print(f">> 推理集预测散点图已保存为: {OUTPUT_PLOT_PATH}")
    else:
        print(f"\n--- 未在数据中检测到真实标签列 '{TRUE_LABEL_COL}'，跳过指标计算和可视化 ---")

    # ==========================================
    # 6. 保存预测结果
    # ==========================================
    cols = df_new.columns.tolist()
    if 'predicted_yield' in cols:
        cols.remove('predicted_yield')
        cols.insert(1, 'predicted_yield')  # 挪到前面方便查看
    df_new = df_new[cols]

    df_new.to_csv(OUTPUT_CSV_PATH, index=False, encoding='gbk')
    print(f"\n>> 结果已保存至: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()