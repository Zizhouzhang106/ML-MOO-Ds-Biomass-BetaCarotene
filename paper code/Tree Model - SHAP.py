#Tree Model - SHAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("正在读取数据...")
df = pd.read_csv(r'C:\Users\data.csv', encoding='ISO-8859-1')#import data

df = df.fillna(df.mean())

y = df[['Dry weight ', 'Carotene yield']]
X = df[['light_intensity', 'temperature', 'hormone', 'bicarbonate', 
        'nitrogen_source', 'phosphorus_source', 'cultivation_time', 'nacl']]

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {list(X.columns)}")
print(f"目标变量: {list(y.columns)}")

print("\n正在划分数据集...")
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print(f"训练集大小: {X_train.shape} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"验证集大小: {X_val.shape} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"测试集大小: {X_test.shape} ({X_test.shape[0]/len(X)*100:.1f}%)")

X_test_original = X_test.copy()

print("\n正在归一化数据（无数据泄露）...")
scaler_X = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)

X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

print("数据归一化完成！")
print("注意：归一化器仅使用训练集数据拟合，避免数据泄露。")

print("\n" + "="*60)
print("开始模型训练...")
print("="*60)

print("\n训练随机森林模型...")

random_forest_model1 = RandomForestRegressor(n_estimators=64, random_state=42, max_depth=16)
random_forest_model1.fit(X_train_scaled, y_train['Dry weight '])

random_forest_model2 = RandomForestRegressor(n_estimators=79, random_state=42, max_depth=16)
random_forest_model2.fit(X_train_scaled, y_train['Carotene yield'])

print("随机森林模型训练完成!")

print("\n训练XGBoost模型...")

xgb_model1 = XGBRegressor(n_estimators=310, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model1.fit(X_train_scaled, y_train['Dry weight '])

xgb_model2 = XGBRegressor(n_estimators=243, max_depth=3, learning_rate=0.2, random_state=42)
xgb_model2.fit(X_train_scaled, y_train['Carotene yield'])

print("XGBoost模型训练完成!")

print("\n训练GBDT模型...")

gbdt_model1 = GradientBoostingRegressor(n_estimators=322, max_depth=6, random_state=42)
gbdt_model1.fit(X_train_scaled, y_train['Dry weight '])

gbdt_model2 = GradientBoostingRegressor(n_estimators=349, max_depth=3, random_state=42)
gbdt_model2.fit(X_train_scaled, y_train['Carotene yield'])

print("单目标GBDT模型训练完成!")

print("\n训练多输出GBDT模型...")

multi_gbdt = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        random_state=42
    ),
    n_jobs=-1
)

multi_gbdt.fit(X_train_scaled, y_train)

print("多输出GBDT模型训练完成!")

print("\n" + "="*60)
print("模型性能评估（测试集）")
print("="*60)

def evaluate_model(model, X_test, y_test, model_name, target_name):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name} - {target_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return rmse, r2, mae

def evaluate_multioutput_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    results = {}
    print(f"{model_name} 多输出模型性能:")
    
    for i, target_name in enumerate(['Dry weight', 'Carotene yield']):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        
        print(f"  {target_name}:")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R²: {r2:.4f}")
        print(f"    MAE: {mae:.4f}")
        
        results[f"{target_name}_RMSE"] = rmse
        results[f"{target_name}_R2"] = r2
        results[f"{target_name}_MAE"] = mae
    
    return results

print("\n[Dry weight] 模型性能:")
rmse_rf, r2_rf, mae_rf = evaluate_model(random_forest_model1, X_test_scaled, y_test['Dry weight '], "随机森林", "Dry weight")
rmse_xgb, r2_xgb, mae_xgb = evaluate_model(xgb_model1, X_test_scaled, y_test['Dry weight '], "XGBoost", "Dry weight")
rmse_gbdt, r2_gbdt, mae_gbdt = evaluate_model(gbdt_model1, X_test_scaled, y_test['Dry weight '], "GBDT", "Dry weight")

print("\n[Carotene yield] 模型性能:")
rmse_rf2, r2_rf2, mae_rf2 = evaluate_model(random_forest_model2, X_test_scaled, y_test['Carotene yield'], "随机森林", "Carotene yield")
rmse_xgb2, r2_xgb2, mae_xgb2 = evaluate_model(xgb_model2, X_test_scaled, y_test['Carotene yield'], "XGBoost", "Carotene yield")
rmse_gbdt2, r2_gbdt2, mae_gbdt2 = evaluate_model(gbdt_model2, X_test_scaled, y_test['Carotene yield'], "GBDT", "Carotene yield")

print("\n[多输出GBDT模型性能]:")
multi_gbdt_results = evaluate_multioutput_model(multi_gbdt, X_test_scaled, y_test, "GBDT多输出")

print("\n" + "="*60)
print("开始SHAP分析...")
print("="*60)

def perform_complete_shap_analysis(model, model_name, X_train, X_test, feature_names, target_name, save_path, is_multioutput=False, target_index=None):
    if is_multioutput:
        print(f"\n===== SHAP Analysis for {target_name} ({model_name} Multi-output Model) =====")
        if hasattr(model, 'estimators_'):
            estimator = model.estimators_[target_index]
        else:
            estimator = model
    else:
        print(f"\n===== SHAP Analysis for {target_name} ({model_name} Model) =====")
        estimator = model
    
    if is_multioutput:
        model_save_path = os.path.join(save_path, f"{model_name}_Multi_{target_name.replace(' ', '_')}")
    else:
        model_save_path = os.path.join(save_path, f"{model_name}_{target_name.replace(' ', '_')}")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    test_data = X_test
    
    print(f"Test data shape: {test_data.shape}")
    
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(test_data)
    except Exception as e:
        print(f"TreeExplainer失败，尝试使用其他解释器: {e}")
        try:
            explainer = shap.Explainer(estimator, X_train)
            shap_values = explainer(test_data)
        except Exception as e2:
            print(f"所有解释器都失败: {e2}")
            return None
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    try:
        shap_values_array = np.array(shap_values)
        
        plt.figure(figsize=(12, 8), dpi=300)
        shap.summary_plot(shap_values_array, test_data, feature_names=feature_names, plot_type="dot", show=False)
        title = f'SHAP Feature Impact for {target_name} ({model_name})'
        if is_multioutput:
            title += " (Multi-output)"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if is_multioutput:
            dot_filename = os.path.join(model_save_path, f"SHAP_Dot_{target_name.replace(' ', '_')}_{model_name}_Multi.png")
        else:
            dot_filename = os.path.join(model_save_path, f"SHAP_Dot_{target_name.replace(' ', '_')}_{model_name}.png")
        
        plt.savefig(dot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved SHAP dot plot: {dot_filename}")
        
        plt.figure(figsize=(12, 8), dpi=300)
        shap.summary_plot(shap_values_array, test_data, feature_names=feature_names, plot_type="bar", show=False)
        title = f'Global Feature Importance for {target_name} ({model_name})'
        if is_multioutput:
            title += " (Multi-output)"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if is_multioutput:
            bar_filename = os.path.join(model_save_path, f"SHAP_Bar_{target_name.replace(' ', '_')}_{model_name}_Multi.png")
        else:
            bar_filename = os.path.join(model_save_path, f"SHAP_Bar_{target_name.replace(' ', '_')}_{model_name}.png")
        
        plt.savefig(bar_filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved SHAP bar plot: {bar_filename}")
        
        shap_importance = np.abs(shap_values_array).mean(axis=0)
        top_indices = np.argsort(shap_importance)[-4:]
        
        for idx in top_indices:
            feature = feature_names[idx]
            plt.figure(figsize=(10, 6), dpi=300)
            shap.dependence_plot(feature, shap_values_array, test_data, feature_names=feature_names, show=False)
            title = f'SHAP Dependence Plot for {feature} ({target_name}, {model_name})'
            if is_multioutput:
                title += " (Multi-output)"
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if is_multioutput:
                dep_filename = os.path.join(model_save_path, f"SHAP_Dependence_{feature}_{target_name.replace(' ', '_')}_{model_name}_Multi.png")
            else:
                dep_filename = os.path.join(model_save_path, f"SHAP_Dependence_{feature}_{target_name.replace(' ', '_')}_{model_name}.png")
            
            plt.savefig(dep_filename, bbox_inches='tight', dpi=300)
            plt.close()
        
        print(f"Saved SHAP dependence plots for top 4 features")
        
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return shap_values

def perform_multi_target_shap_comparison(model, model_name, X_train, X_test, feature_names, target_names, save_path):
    print(f"\n===== Multi-target SHAP Comparison for {model_name} =====")
    
    comparison_path = os.path.join(save_path, f"{model_name}_Multi_target_Comparison")
    os.makedirs(comparison_path, exist_ok=True)
    
    shap_values_list = []
    importance_scores = []
    
    for i, target_name in enumerate(target_names):
        print(f"\n计算 {target_name} 的SHAP值...")
        
        if hasattr(model, 'estimators_'):
            estimator = model.estimators_[i]
        else:
            estimator = model
        
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) == 1 else shap_values
            
            shap_values_list.append(shap_values)
            
            importance = np.abs(shap_values).mean(axis=0)
            importance_scores.append(importance)
            
            print(f"  {target_name} SHAP值计算完成，形状: {np.array(shap_values).shape}")
        except Exception as e:
            print(f"  {target_name} SHAP值计算失败: {e}")
            return None
    
    if len(shap_values_list) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for i, (target_name, importance) in enumerate(zip(target_names, importance_scores)):
            sorted_idx = np.argsort(importance)
            sorted_features = [feature_names[idx] for idx in sorted_idx]
            sorted_importance = importance[sorted_idx]
            
            axes[i].barh(range(len(feature_names)), sorted_importance)
            axes[i].set_yticks(range(len(feature_names)))
            axes[i].set_yticklabels(sorted_features)
            axes[i].set_xlabel('SHAP重要性（平均绝对影响）')
            axes[i].set_title(f'{target_name} 特征重要性')
            axes[i].grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'{model_name} 多目标特征重要性比较', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        comparison_chart = os.path.join(comparison_path, f"{model_name}_多目标重要性比较.png")
        plt.savefig(comparison_chart, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved multi-target importance comparison: {comparison_chart}")
        
        if len(importance_scores) == 2:
            importance_diff = importance_scores[0] - importance_scores[1]
            
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(feature_names)), importance_diff)
            
            for bar, diff in zip(bars, importance_diff):
                if diff > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('重要性差异 (目标1 - 目标2)')
            plt.title(f'{model_name}: {target_names[0]} vs {target_names[1]} 特征重要性差异', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            diff_chart = os.path.join(comparison_path, f"{model_name}_多目标重要性差异.png")
            plt.savefig(diff_chart, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved multi-target importance difference: {diff_chart}")
            
            correlation_matrix = np.corrcoef(np.array(shap_values_list).reshape(len(target_names), -1))
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='coolwarm', 
                       center=0,
                       xticklabels=target_names,
                       yticklabels=target_names)
            plt.title(f'{model_name}: SHAP值相关性矩阵', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            corr_chart = os.path.join(comparison_path, f"{model_name}_SHAP相关性矩阵.png")
            plt.savefig(corr_chart, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP correlation matrix: {corr_chart}")
            
            importance_df = pd.DataFrame({
                '特征': feature_names,
                f'{target_names[0]}_重要性': importance_scores[0],
                f'{target_names[1]}_重要性': importance_scores[1],
                '重要性差异': importance_diff,
                '相对差异(%)': (importance_diff / (np.abs(importance_scores[0]) + np.abs(importance_scores[1]) + 1e-10) * 100)
            })
            
            importance_df['绝对差异'] = np.abs(importance_df['重要性差异'])
            importance_df = importance_df.sort_values('绝对差异', ascending=False)
            
            importance_file = os.path.join(comparison_path, f"{model_name}_多目标SHAP重要性差异.csv")
            importance_df.to_csv(importance_file, index=False, encoding='utf-8-sig')
            print(f"Saved multi-target importance differences to CSV: {importance_file}")
    
    print(f"\n多目标SHAP比较分析完成！")
    return shap_values_list

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
shap_results_path = os.path.join(desktop_path, "SHAP_Results_Multi")
os.makedirs(shap_results_path, exist_ok=True)

print(f"\nSHAP结果将保存到: {shap_results_path}")

feature_names = ['light_intensity', 'temperature', 'hormone', 'bicarbonate', 
                 'nitrogen_source', 'phosphorus_source', 'cultivation_time', 'nacl']

targets = ['Dry weight ', 'Carotene yield']

models = {
    'RF': {
        'Dry weight ': random_forest_model1,
        'Carotene yield': random_forest_model2
    },
    'XGB': {
        'Dry weight ': xgb_model1,
        'Carotene yield': xgb_model2
    },
    'GBDT': {
        'Dry weight ': gbdt_model1,
        'Carotene yield': gbdt_model2
    }
}

multi_models = {
    'MultiGBDT': multi_gbdt
}

shap_values_dict = {}

print("\n" + "="*60)
print("执行单目标SHAP分析...")
print("="*60)

for model_name in models:
    for target in targets:
        model = models[model_name][target]
        
        shap_values = perform_complete_shap_analysis(
            model, 
            model_name, 
            X_train_scaled, 
            X_test_scaled, 
            feature_names, 
            target, 
            shap_results_path
        )
        
        if shap_values is not None:
            shap_values_dict[f"{model_name}_{target}"] = shap_values

print("\n" + "="*60)
print("执行多目标SHAP分析...")
print("="*60)

multi_shap_values = perform_multi_target_shap_comparison(
    multi_gbdt,
    "MultiGBDT",
    X_train_scaled,
    X_test_scaled,
    feature_names,
    ['Dry weight', 'Carotene yield'],
    shap_results_path
)

if multi_shap_values is not None:
    for i, target in enumerate(targets):
        shap_values = perform_complete_shap_analysis(
            multi_gbdt,
            "MultiGBDT",
            X_train_scaled,
            X_test_scaled,
            feature_names,
            target,
            shap_results_path,
            is_multioutput=True,
            target_index=i
        )
        
        if shap_values is not None:
            shap_values_dict[f"MultiGBDT_{target}"] = shap_values

print("\n" + "="*60)
print("导出SHAP值到CSV文件...")
print("="*60)

for model_name in models:
    for target in targets:
        model = models[model_name][target]
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            df_shap = pd.DataFrame(shap_values, columns=feature_names)
            
            for i, feature in enumerate(feature_names):
                df_shap[f'{feature}_原始值'] = X_test_original[feature].values
            
            if target == 'Dry weight ':
                df_shap['目标变量_Dry_weight'] = y_test['Dry weight '].values
            else:
                df_shap['目标变量_Carotene_yield'] = y_test['Carotene yield'].values
            
            filename = f"SHAP_{target.replace(' ', '_')}_{model_name}.csv"
            filepath = os.path.join(shap_results_path, filename)
            df_shap.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"Saved SHAP values: {filepath}")
        except Exception as e:
            print(f"导出 {model_name}_{target} SHAP值时出错: {e}")

if multi_shap_values is not None:
    for i, target in enumerate(targets):
        shap_values = multi_shap_values[i]
        
        df_shap = pd.DataFrame(shap_values, columns=feature_names)
        
        for j, feature in enumerate(feature_names):
            df_shap[f'{feature}_原始值'] = X_test_original[feature].values
        
        df_shap[f'目标变量_{target.replace(" ", "_")}'] = y_test.iloc[:, i].values
        
        filename = f"SHAP_{target.replace(' ', '_')}_MultiGBDT.csv"
        filepath = os.path.join(shap_results_path, filename)
        df_shap.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Saved multi-output SHAP values: {filepath}")

print("\n" + "="*60)
print("生成模型性能总结...")
print("="*60)

performance_summary = pd.DataFrame({
    '模型': ['随机森林', 'XGBoost', 'GBDT', 'GBDT多输出'] * 2,
    '目标变量': ['Dry weight'] * 4 + ['Carotene yield'] * 4,
    'RMSE': [rmse_rf, rmse_xgb, rmse_gbdt, multi_gbdt_results.get('Dry weight_RMSE', np.nan), 
             rmse_rf2, rmse_xgb2, rmse_gbdt2, multi_gbdt_results.get('Carotene yield_RMSE', np.nan)],
    'R²': [r2_rf, r2_xgb, r2_gbdt, multi_gbdt_results.get('Dry weight_R2', np.nan), 
           r2_rf2, r2_xgb2, r2_gbdt2, multi_gbdt_results.get('Carotene yield_R2', np.nan)],
    'MAE': [mae_rf, mae_xgb, mae_gbdt, multi_gbdt_results.get('Dry weight_MAE', np.nan), 
            mae_rf2, mae_xgb2, mae_gbdt2, multi_gbdt_results.get('Carotene yield_MAE', np.nan)]
})

performance_file = os.path.join(shap_results_path, "模型性能总结.csv")
performance_summary.to_csv(performance_file, index=False, encoding='utf-8-sig')
print(f"模型性能总结已保存: {performance_file}")

print("\n" + "="*60)
print("最佳模型总结:")
print("="*60)

for target in ['Dry weight', 'Carotene yield']:
    target_data = performance_summary[performance_summary['目标变量'] == target]
    if not target_data.empty:
        best_idx = target_data['R²'].idxmax()
        best_model = target_data.loc[best_idx]
        print(f"对于 {target}: {best_model['模型']} (R² = {best_model['R²']:.4f})")

print("\n" + "="*60)
print("生成SHAP值总结...")
print("="*60)

shap_importance_summary = pd.DataFrame(index=feature_names)

for key, shap_values in shap_values_dict.items():
    if shap_values is not None:
        try:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            shap_importance_summary[f"{key}_SHAP重要性"] = shap_abs_mean
        except:
            pass

if not shap_importance_summary.empty:
    shap_importance_file = os.path.join(shap_results_path, "SHAP重要性总结.csv")
    shap_importance_summary.to_csv(shap_importance_file, encoding='utf-8-sig')
    print(f"SHAP重要性总结已保存: {shap_importance_file}")

print("\n" + "="*60)
print("创建分析报告...")
print("="*60)

multi_target_report = f"""
多目标SHAP分析特别报告
=====================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 多输出GBDT模型概述
-------------------
- 模型类型: MultiOutputRegressor包装的GradientBoostingRegressor
- 参数设置: n_estimators=300, max_depth=5
- 训练方式: 同时训练两个目标变量

2. 多目标SHAP分析亮点
-------------------
- 生成了多目标特征重要性比较图
- 计算了特征对不同目标的重要性差异
- 分析了SHAP值在不同目标间的相关性
- 识别了影响两个目标的共同关键特征

3. 主要发现
-----------
"""

if multi_shap_values is not None and len(multi_shap_values) >= 2:
    importance_target1 = np.abs(multi_shap_values[0]).mean(axis=0)
    importance_target2 = np.abs(multi_shap_values[1]).mean(axis=0)
    
    top_idx1 = np.argsort(importance_target1)[-3:][::-1]
    top_idx2 = np.argsort(importance_target2)[-3:][::-1]
    
    multi_target_report += f"""
对于Dry weight最重要的3个特征:
1. {feature_names[top_idx1[0]]}: {importance_target1[top_idx1[0]]:.4f}
2. {feature_names[top_idx1[1]]}: {importance_target1[top_idx1[1]]:.4f}
3. {feature_names[top_idx1[2]]}: {importance_target1[top_idx1[2]]:.4f}

对于Carotene yield最重要的3个特征:
1. {feature_names[top_idx2[0]]}: {importance_target2[top_idx2[0]]:.4f}
2. {feature_names[top_idx2[1]]}: {importance_target2[top_idx2[1]]:.4f}
3. {feature_names[top_idx2[2]]}: {importance_target2[top_idx2[2]]:.4f}
"""

report_content = f"""
腐胺数据分析报告 - 多目标SHAP版本
================================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 数据概况
-----------
- 总样本数: {len(df)}
- 特征数: {len(feature_names)}
- 目标变量: Dry weight, Carotene yield

2. 模型配置
-----------
- 单目标模型: 随机森林、XGBoost、GBDT（分别训练）
- 多目标模型: GBDT多输出模型（同时训练两个目标）

3. 数据划分
-----------
- 训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)
- 验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(X)*100:.1f}%)
- 测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)

4. 生成文件
-----------
所有结果文件已保存到: {shap_results_path}
包含:
- 单目标SHAP分析图表
- 多目标SHAP比较图表
- SHAP值CSV文件
- 模型性能总结 (RMSE, R², MAE)
- SHAP重要性总结
- 多目标分析特别报告

5. 多目标分析优势
---------------
1. 可以同时分析特征对两个目标的影响
2. 识别对不同目标影响方向相反的特征
3. 发现同时影响两个目标的共同关键特征
4. 为多目标优化提供决策支持

6. 注意事项
-----------
- 多输出模型使用MultiOutputRegressor包装
- SHAP分析使用TreeExplainer
- 所有分析基于测试集数据，确保无数据泄露
{multi_target_report}
"""

report_file = os.path.join(shap_results_path, "分析报告.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_content)

multi_report_file = os.path.join(shap_results_path, "多目标分析特别报告.txt")
with open(multi_report_file, 'w', encoding='utf-8') as f:
    f.write(multi_target_report)

print(f"分析报告已保存: {report_file}")
print(f"多目标特别报告已保存: {multi_report_file}")

print("\n" + "="*60)
print("分析完成！")
print("="*60)
print(f"所有结果已保存到桌面文件夹: {shap_results_path}")
print("\n主要新增内容:")
print("1. 多输出GBDT模型训练和评估")
print("2. 多目标SHAP比较分析（包括重要性比较图、差异图、相关性矩阵）")
print("3. 多目标SHAP值导出")
print("4. 多目标分析特别报告")
print("\n多目标分析优势:")
print("- 可以同时理解特征对两个目标的不同影响")
print("- 识别特征对目标影响的协同和拮抗效应")
print("- 为多目标优化实验设计提供依据")
print("\n注意事项:")
print("- 多输出模型可能会牺牲一些单目标性能，但提供了更好的整体理解")
print("- 多目标SHAP分析可以帮助识别需要权衡的工艺参数")
