import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # 新增：用于统计学异常诊断
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import joblib

# 忽略所有的警告
warnings.filterwarnings('ignore')
# 单独强制忽略由于未收敛带来的警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 保持学术图表字体

# ==========================================
# 1. 加载与清理数据
# ==========================================
print("正在加载数据...")
df = pd.read_csv(r"I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\Comprehensive_Traits_Report.csv", encoding='gbk')
df = df.dropna(subset=['yield'])

# 动态剔除不需要的列
cols_to_drop = ['tree_id', 'yield']
if '果实数（个）' in df.columns: cols_to_drop.append('果实数（个）')
if '单果重（kg）' in df.columns: cols_to_drop.append('单果重（kg）')

X_raw = df.drop(columns=cols_to_drop)
y_raw = df['yield']

# 使用均值填充
X_raw = X_raw.fillna(X_raw.mean())

# ==========================================
# 1.5 异常值清洗
# ==========================================
print("\n--- 0. 数据清洗：基于 OLS 学生化残差的强影响点剔除 ---")

# 为特征矩阵添加常数列，以拟合标准 OLS 模型
X_ols = sm.add_constant(X_raw)

# 拟合普通最小二乘法 (OLS) 线性回归模型
ols_model = sm.OLS(y_raw, X_ols).fit()

# 获取模型的统计影响指标
influence = ols_model.get_influence()

# 计算内部学生化残差 (Studentized Residuals)
student_resid = influence.resid_studentized_internal

# 设定极严苛的剔除阈值 (1.2)，剔除对回归面拉扯严重的离群样本
strict_threshold = 1.2
valid_indices = np.abs(student_resid) < strict_threshold

outliers_count = len(y_raw) - np.sum(valid_indices)
print(f"设定学生化残差阈值: |Studentized Resid| < {strict_threshold}")
print(f"原始数据量: {len(y_raw)}，基于严苛统计学诊断剔除了 {outliers_count} 条高干扰数据。")
print(f"剩余有效高质量数据: {np.sum(valid_indices)} 条。")

# --- 提取并保存干净的完整数据（包含 tree_id 等列） ---
df_cleaned = df[valid_indices].reset_index(drop=True)
cleaned_data_save_path = r"I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\train_clear\Cleaned_Traits_Report_No_Outliers.csv"
df_cleaned.to_csv(cleaned_data_save_path, index=False, encoding='gbk')
print(f">> 剔除异常值后的干净数据集已保存至: {cleaned_data_save_path}")
# -----------------------------------------------------------

# 更新 X 和 y 为清洗后的干净数据，重置索引
X = X_raw[valid_indices].reset_index(drop=True)
y = y_raw[valid_indices].reset_index(drop=True)

# ==========================================
# 1.8 对【原始数据】的【所有特征】进行相关性分析以及可视化
# ==========================================
print("\n--- 1.5 正在生成原始数据全量特征的相关性可视化图表 ---")
# 核心修改：将 X 替换为 X_raw，将 y 替换为 y_raw
yield_corr_signed = X_raw.corrwith(y_raw).sort_values()

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20
DATA_LABEL_FONT_SIZE = 20

plt.figure(figsize=(14, 10))

# 设定颜色: 蓝色代表负相关，红色代表正相关
colors = ['#377eb8' if val < 0 else '#e41a1c' for val in yield_corr_signed.values]

# 绘制水平柱状图
bars = plt.barh(yield_corr_signed.index, yield_corr_signed.values, color=colors, height=0.5)

# 在柱状图上添加数值标签
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + 0.02 if width > 0 else width - 0.02
    ha = 'left' if width > 0 else 'right'
    plt.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
             va='center', ha=ha, fontsize=DATA_LABEL_FONT_SIZE, color='black')

# 显著增加 X 轴的范围缓冲
x_min, x_max = min(yield_corr_signed.values), max(yield_corr_signed.values)
plt.xlim(x_min - 0.3, x_max + 0.3)

# 学术风格美化
plt.grid(axis='x', linestyle='--', alpha=0.5, color='gray')
plt.gca().set_axisbelow(True)

plt.xlabel("Pearson Correlation Coefficient with Yield", fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
plt.ylabel("Traits / Features", fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
# 核心修改：更新图表标题，注明是 Raw Data (原始数据)
plt.title("Correlation Analysis of All Traits vs. Target Yield (Raw Data)", fontsize=TITLE_FONT_SIZE, pad=25, fontweight='bold')

plt.axvline(x=0, color='black', linewidth=1.2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

plt.subplots_adjust(left=0.3, right=0.9, top=0.88, bottom=0.12)

# 保存图表 (修改了文件名，加上 _raw 以防覆盖清洗后的图表)
corr_save_path = r'I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\train_clear\yield_correlation_with_values_raw.png'
plt.savefig(corr_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f">> 原始全特征相关性排序可视化结果已保存为: {corr_save_path}")

# ==========================================
# 2. 步骤一：基于皮尔逊相关系数的低相关特征过滤
# ==========================================
print("\n--- 1. 低相关性特征过滤 (基于清洗后的数据) ---")
# 计算所有特征与 yield 的皮尔逊相关系数绝对值，并降序排列
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)

# 设定与目标变量的相关系数阈值
target_corr_threshold = 0.3

selected_features = []
dropped_log = []

for feature, corr_value in corr_with_target.items():
    if corr_value >= target_corr_threshold:
        selected_features.append(feature)
        print(f"保留 [{feature}] (相关系数: {corr_value:.4f})")
    else:
        dropped_log.append(f"剔除 [{feature}] (相关系数 {corr_value:.4f} 低于阈值 {target_corr_threshold})")

print(f"\n设定相关系数阈值: {target_corr_threshold}，过滤后保留了 {len(selected_features)} 个特征。")

if dropped_log:
    print("\n[低相关特征剔除记录]:")
    for log in dropped_log:
        print(" -", log)
else:
    print("\n没有特征被剔除，所有特征的相关系数均高于阈值。")

# 生成过滤后的特征矩阵
X_filtered = X[selected_features]

# ==========================================
# 3. 步骤二：随机森林特征重要性分析 (排序)
# ==========================================
print("\n--- 2. 随机森林核心特征排序 ---")
rf_eval = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf_eval.fit(X_filtered, y)

importances = pd.Series(rf_eval.feature_importances_, index=X_filtered.columns).sort_values(ascending=False)
ranked_features = importances.index.tolist()

print("\n★ 特征重要性排序:")
for feat, imp in importances.items():
    print(f" - {feat}: {imp:.4f}")

# ==========================================
# 4. 步骤三：多模型 5 折 CV 网格搜索 + 增量特征测试
# ==========================================
print("\n--- 3. 结合全局袋外预测 (cross_val_predict) 的增量特征测试 ---")

# 定义 5 折交叉验证
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

model_params = {
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 150, 200],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.02, 0.03, 0.05]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 150, 200],
            'max_depth': [2, 3, 4, 5],
            'min_samples_leaf': [2, 3, 4]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
        "params": {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
    },
    "KNN Regression": {
        "model": Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())]),
        "params": {
            'knn__n_neighbors': [3, 5, 7, 9],
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2]
        }
    },
    "Neural Network (MLP)": {
        "model": Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(max_iter=1300, random_state=42))]),
        "params": {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': [0.001, 0.01]
        }
    },
    "ElasticNet": {
        "model": Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(random_state=42))]),
        "params": {
            'enet__alpha': [0.01, 0.05, 0.1, 1.0],
            'enet__l1_ratio': [0.1, 0.3, 0.5, 0.7]
        }
    },
    "Ridge Regression": {
        "model": Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(random_state=42))]),
        "params": {
            'ridge__alpha': [0.1, 0.5, 1.0, 10.0, 50.0]
        }
    }
}

results = []
scoring = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}

global_best_r2 = -float('inf')
best_overall_model = None
best_overall_features = None
best_overall_name = ""

for model_name, mp in model_params.items():
    print(f"\n>> 正在对 {model_name} 进行增量特征搜索...")

    for i in range(1, len(ranked_features) + 1):
        current_features = ranked_features[:i]
        X_sub = X_filtered[current_features]

        grid = GridSearchCV(mp['model'], mp['params'], cv=cv_strategy,
                            scoring=scoring, refit='r2', n_jobs=-1)
        grid.fit(X_sub, y)

        best_est = grid.best_estimator_
        cv_y_pred = cross_val_predict(best_est, X_sub, y, cv=cv_strategy)

        aligned_cv_r2 = r2_score(y, cv_y_pred)
        aligned_cv_rmse = np.sqrt(mean_squared_error(y, cv_y_pred))
        aligned_cv_mae = mean_absolute_error(y, cv_y_pred)

        results.append({
            'Model': model_name,
            'Num_Features': i,
            'Best_CV_R2': aligned_cv_r2,
            'Best_CV_RMSE': aligned_cv_rmse,
            'Best_CV_MAE': aligned_cv_mae,
            'Best_Params': str(grid.best_params_)
        })
        print(f"   Top {i:<2} 特征 | R²: {aligned_cv_r2:.4f} | RMSE: {aligned_cv_rmse:.4f} | MAE: {aligned_cv_mae:.4f}")

        if aligned_cv_r2 > global_best_r2:
            global_best_r2 = aligned_cv_r2
            best_overall_model = best_est
            best_overall_features = current_features
            best_overall_name = model_name

results_df = pd.DataFrame(results)

# ==========================================
# 提取并保存每个模型的最佳结果
# ==========================================
print("\n--- 提取各模型的最佳性能结果 ---")
best_per_model_df = results_df.loc[results_df.groupby('Model')['Best_CV_R2'].idxmax()].copy()
best_per_model_df = best_per_model_df.sort_values(by='Best_CV_R2', ascending=False).reset_index(drop=True)

best_per_model_df = best_per_model_df[['Model', 'Num_Features', 'Best_CV_R2', 'Best_CV_RMSE', 'Best_CV_MAE', 'Best_Params']]
best_per_model_df.columns = ['Model Name', 'Optimal Num Features', 'Max CV R²', 'Min CV RMSE', 'Min CV MAE', 'Best Hyperparameters']

print(best_per_model_df.to_string())

table_save_path = r"I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\train_clear\models_best_performance_table.csv"
best_per_model_df.to_csv(table_save_path, index=False, encoding='utf-8-sig')
print(f"\n>> 各模型最佳结果汇总表已保存为: {table_save_path}")

print("\n" + "=" * 60)
print(f"🏆 全局最优组合诞生！")
print(f"最强算法: {best_overall_name}")
print(f"最优特征数: Top {len(best_overall_features)}")
print(f"最高袋外 R²: {global_best_r2:.4f}")
print("=" * 60)

# ==========================================
# 新增：最终训练并保存模型权重
# ==========================================
print("\n--- 4. 在全部干净数据上重新拟合并保存模型 ---")
X_best_subset = X_filtered[best_overall_features]
best_overall_model.fit(X_best_subset, y)

model_save_path = r"I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\train_clear\best_yield_model.pkl"
joblib.dump(best_overall_model, model_save_path)
print(f">> 最终最优模型权重已保存至: {model_save_path}")

# ==========================================
# 5. 可视化一：多模型性能对比折线图
# ==========================================
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=results_df, x='Num_Features', y='Best_CV_R2',
    hue='Model', style='Model', markers=True, dashes=False,
    linewidth=2.5, markersize=9
)

plt.title('Yield Prediction R² Trends with Incremental Features (Pooled CV Method)', fontsize=20, pad=15)
plt.xlabel('Number of Features Added (in Descending Order of Importance)', fontsize=20)
plt.ylabel('Best 5-Fold CV R² Score', fontsize=20)
plt.xticks(range(1, len(ranked_features) + 1), fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Algorithms', fontsize=18, loc='lower right')
plt.tight_layout()

lineplot_path = r"I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\train_clear\yield_incremental_cv_lineplot.png"
plt.savefig(lineplot_path, dpi=300)
print(f"\n>> 增量特征性能折线图已保存为: {lineplot_path}")

# ==========================================
# 6. 可视化二：最优模型的预测散点图与指标
# ==========================================
print("\n--- 5. 生成全局最优模型的预测散点图 ---")
y_pred_best = cross_val_predict(best_overall_model, X_best_subset, y, cv=cv_strategy)

final_r2 = r2_score(y, y_pred_best)
final_rmse = np.sqrt(mean_squared_error(y, y_pred_best))
final_mae = mean_absolute_error(y, y_pred_best)

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred_best, alpha=0.7, color='dodgerblue', edgecolors='k', s=60,
            label='CV Out-of-fold Predictions')

min_val = min(y.min(), y_pred_best.min())
max_val = max(y.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')

plt.title(f'Actual vs Predicted Yield ({best_overall_name}, Top {len(best_overall_features)} Features)', fontsize=16)
plt.xlabel('Actual Yield (kg/tree)', fontsize=20)
plt.ylabel('Predicted Yield (kg/tree)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

textstr = f'$R^2$ = {final_r2:.4f}\nRMSE = {final_rmse:.4f}\nMAE = {final_mae:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20,
         verticalalignment='top', bbox=props)

plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()

scatter_save_path = r"I:\pinghemiyou\cailiao\train\new\MCSC\qt\extract\train_clear\yield_prediction_scatter_best.png"
plt.savefig(scatter_save_path, dpi=300)
print(f">> 最优模型预测散点图已保存为: {scatter_save_path}")