#Tree Model Hyperparameter Optimization - A Case Study of XGBoost
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import warnings
warnings.filterwarnings('ignore')

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
results_dir = os.path.join(desktop_path, 'XGBoost_Optimization_Results')

os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存到: {results_dir}")

df = pd.read_csv(r'C:\Users\data.csv', encoding='ISO-8859-1')#import data
df = df.fillna(df.mean())
X = df[['light_intensity', 'temperature', 'hormone', 'bicarbonate', 
        'nitrogen_source', 'phosphorus_source', 'cultivation_time', 'nacl']]
y_dry = df['Dry weight ']
y_car = df['Carotene yield']
y_multi = df[['Dry weight ', 'Carotene yield']]

print(f"数据形状: X={X.shape}, y_dry={y_dry.shape}, y_car={y_car.shape}, y_multi={y_multi.shape}")

def optimize_single_target(X_train, X_val, X_test, y_train, y_val, y_test, target_name):
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
        X_val_array = X_val.values
        X_test_array = X_test.values
    else:
        X_train_array = X_train
        X_val_array = X_val
        X_test_array = X_test
    
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
        y_val_array = y_val.values
        y_test_array = y_test.values
    else:
        y_train_array = y_train
        y_val_array = y_val
        y_test_array = y_test
    
    def objective(params):
        model = XGBRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X_train_array):
            X_train_fold = X_train_array[train_idx]
            X_val_fold = X_train_array[val_idx]
            y_train_fold = y_train_array[train_idx]
            y_val_fold = y_train_array[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            rmse_scores.append(rmse)
        
        avg_rmse = np.mean(rmse_scores)
        
        return {'loss': avg_rmse, 'status': STATUS_OK}
    
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 350, 10),
        'max_depth': hp.quniform('max_depth', 3, 20, 1)
    }
    
    print(f"正在优化 {target_name}...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials,
        show_progressbar=False
    )
    
    final_model = XGBRegressor(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(np.vstack([X_train_array, X_val_array]), 
                    np.concatenate([y_train_array, y_val_array]))
    
    y_pred = final_model.predict(X_test_array)
    rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
    
    print(f"\n{target_name} - 最佳参数:")
    print(f"  n_estimators: {int(best['n_estimators'])}")
    print(f"  max_depth: {int(best['max_depth'])}")
    print(f"  测试集 RMSE: {rmse:.4f}")
    
    return best, final_model, trials

def optimize_multi_target(X_train, X_val, X_test, y_train, y_val, y_test):
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
        X_val_array = X_val.values
        X_test_array = X_test.values
    else:
        X_train_array = X_train
        X_val_array = X_val
        X_test_array = X_test
    
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
        y_val_array = y_val.values
        y_test_array = y_test.values
    else:
        y_train_array = y_train
        y_val_array = y_val
        y_test_array = y_test
    
    def objective(params):
        base_model = XGBRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model = MultiOutputRegressor(base_model)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X_train_array):
            X_train_fold = X_train_array[train_idx]
            X_val_fold = X_train_array[val_idx]
            y_train_fold = y_train_array[train_idx]
            y_val_fold = y_train_array[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            rmse_dry = np.sqrt(mean_squared_error(y_val_fold[:, 0], y_pred[:, 0]))
            rmse_car = np.sqrt(mean_squared_error(y_val_fold[:, 1], y_pred[:, 1]))
            avg_rmse = (rmse_dry + rmse_car) / 2
            rmse_scores.append(avg_rmse)
        
        avg_rmse_cv = np.mean(rmse_scores)
        
        return {'loss': avg_rmse_cv, 'status': STATUS_OK}
    
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 350, 1),
        'max_depth': hp.quniform('max_depth', 3, 20, 1)
    }
    
    print("正在优化多目标模型...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        show_progressbar=False
    )
    
    base_model = XGBRegressor(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    final_model = MultiOutputRegressor(base_model)
    final_model.fit(np.vstack([X_train_array, X_val_array]), 
                    np.vstack([y_train_array, y_val_array]))
    
    y_pred = final_model.predict(X_test_array)
    rmse_dry = np.sqrt(mean_squared_error(y_test_array[:, 0], y_pred[:, 0]))
    rmse_car = np.sqrt(mean_squared_error(y_test_array[:, 1], y_pred[:, 1]))
    
    print(f"\n多目标模型 - 最佳参数:")
    print(f"  n_estimators: {int(best['n_estimators'])}")
    print(f"  max_depth: {int(best['max_depth'])}")
    print(f"  干重 RMSE: {rmse_dry:.4f}")
    print(f"  胡萝卜素 RMSE: {rmse_car:.4f}")
    
    return best, final_model, trials

def save_hyperparameter_search(trials, target_name, model_type):
    try:
        trials_data = []
        for i, t in enumerate(trials.trials):
            trial_info = {
                'trial_id': i,
                'loss': t['result']['loss'],
                'status': t['result']['status'],
                'iteration': t.get('tid', i)
            }
            
            for param_name, param_values in t['misc']['vals'].items():
                if param_values:
                    trial_info[param_name] = param_values[0]
            
            trials_data.append(trial_info)
        
        trials_df = pd.DataFrame(trials_data)
        
        best_idx = trials_df['loss'].idxmin()
        
        trials_df['model_type'] = model_type
        trials_df['target_name'] = target_name
        trials_df['is_best'] = False
        trials_df.loc[best_idx, 'is_best'] = True
        
        filename = f"{target_name.replace(' ', '_')}_{model_type}_trials.csv"
        filepath = os.path.join(results_dir, filename)
        trials_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  优化过程数据已保存: {filepath}")
        print(f"  总试验次数: {len(trials_df)}")
        print(f"  最佳试验ID: {best_idx}")
        print(f"  最佳RMSE: {trials_df.loc[best_idx, 'loss']:.4f}")
        
        return trials_df
    
    except Exception as e:
        print(f"保存优化过程数据时出错: {e}")
        return None

def main():
    print("="*60)
    print("XGBoost超参数优化 - 单目标 vs 多目标")
    print("="*60)
    print("优化参数: n_estimators, max_depth")
    print("评价指标: RMSE")
    print("="*60)
    
    print("\n正在进行数据划分")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_multi, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    y_train_dry = y_train['Dry weight ']
    y_val_dry = y_val['Dry weight ']
    y_test_dry = y_test['Dry weight ']
    
    y_train_car = y_train['Carotene yield']
    y_val_car = y_val['Carotene yield']
    y_test_car = y_test['Carotene yield']
    
    print(f"训练集大小: {len(X_train)} 个样本 (60%)")
    print(f"验证集大小: {len(X_val)} 个样本 (20%)")
    print(f"测试集大小: {len(X_test)} 个样本 (20%)")
    
    print("\n" + "="*50)
    print("1. 单目标优化：干重 (Dry Weight)")
    print("="*50)
    
    best_dry, model_dry, trials_dry = optimize_single_target(
        X_train, X_val, X_test, y_train_dry, y_val_dry, y_test_dry, "干重模型"
    )
    trials_df_dry = save_hyperparameter_search(
        trials_dry, "Dry_Weight", "Single_Objective"
    )
    
    print("\n" + "="*50)
    print("2. 单目标优化：胡萝卜素 (Carotene Yield)")
    print("="*50)
    
    best_car, model_car, trials_car = optimize_single_target(
        X_train, X_val, X_test, y_train_car, y_val_car, y_test_car, "胡萝卜素模型"
    )
    trials_df_car = save_hyperparameter_search(
        trials_car, "Carotene_Yield", "Single_Objective"
    )
    
    print("\n" + "="*50)
    print("3. 多目标优化：同时预测干重和胡萝卜素")
    print("="*50)
    
    best_multi, model_multi, trials_multi = optimize_multi_target(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    trials_df_multi = save_hyperparameter_search(
        trials_multi, "Multi_Target", "Multi_Objective"
    )
    
    print("\n" + "="*50)
    print("优化结果对比")
    print("="*50)
    
    results_summary = []
    
    y_pred_dry = model_dry.predict(X_test.values)
    test_rmse_dry = np.sqrt(mean_squared_error(y_test_dry.values, y_pred_dry))
    
    y_pred_car = model_car.predict(X_test.values)
    test_rmse_car = np.sqrt(mean_squared_error(y_test_car.values, y_pred_car))
    
    y_pred_multi = model_multi.predict(X_test.values)
    test_rmse_dry_multi = np.sqrt(mean_squared_error(y_test.values[:, 0], y_pred_multi[:, 0]))
    test_rmse_car_multi = np.sqrt(mean_squared_error(y_test.values[:, 1], y_pred_multi[:, 1]))
    
    results_summary.append({
        '模型类型': '单目标-干重',
        'n_estimators': int(best_dry['n_estimators']),
        'max_depth': int(best_dry['max_depth']),
        '交叉验证RMSE': trials_df_dry['loss'].min() if trials_df_dry is not None else np.nan,
        '测试集RMSE': test_rmse_dry
    })
    
    results_summary.append({
        '模型类型': '单目标-胡萝卜素',
        'n_estimators': int(best_car['n_estimators']),
        'max_depth': int(best_car['max_depth']),
        '交叉验证RMSE': trials_df_car['loss'].min() if trials_df_car is not None else np.nan,
        '测试集RMSE': test_rmse_car
    })
    
    results_summary.append({
        '模型类型': '多目标',
        'n_estimators': int(best_multi['n_estimators']),
        'max_depth': int(best_multi['max_depth']),
        '交叉验证加权RMSE': trials_df_multi['loss'].min() if trials_df_multi is not None else np.nan,
        '测试集RMSE(干重)': test_rmse_dry_multi,
        '测试集RMSE(胡萝卜素)': test_rmse_car_multi
    })
    
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        print("\n最佳参数组合:")
        print(results_df.to_string(index=False))
        
        results_file = os.path.join(results_dir, 'xgb_optimization_results_summary.csv')
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"\n优化结果汇总已保存到: {results_file}")
        
        all_trials_dfs = []
        if trials_df_dry is not None:
            all_trials_dfs.append(trials_df_dry)
        if trials_df_car is not None:
            all_trials_dfs.append(trials_df_car)
        if trials_df_multi is not None:
            all_trials_dfs.append(trials_df_multi)
        
        if all_trials_dfs:
            all_trials_df = pd.concat(all_trials_dfs, ignore_index=True)
            all_trials_file = os.path.join(results_dir, 'all_optimization_trials.csv')
            all_trials_df.to_csv(all_trials_file, index=False, encoding='utf-8-sig')
            print(f"所有优化过程数据已合并保存到: {all_trials_file}")
    
    print("\n" + "="*60)
    print("优化完成! 所有结果已保存到桌面文件夹。")
    print(f"文件夹路径: {results_dir}")
    print("="*60)
    
    print("\n生成的文件列表:")
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(results_dir, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file} ({file_size/1024:.1f} KB)")
    else:
        print(f"警告: 文件夹 {results_dir} 不存在!")

if __name__ == "__main__":
    main()
