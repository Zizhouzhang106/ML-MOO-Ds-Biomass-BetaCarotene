#Multi-objective model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\data.csv', encoding='ISO-8859-1')#import data

X = df[['light_intensity', 'temperature', 'hormone','bicarbonate', 
        'nitrogen_source', 'phosphorus_source', 'cultivation_time', 'nacl']]
y = df[['Dry weight ', 'Carotene yield']]

feature_names = X.columns.tolist()

print("="*60)
print("数据划分信息")
print("="*60)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"总数据集大小: {len(X)}")
print(f"训练集（含后续验证）大小: {len(X_train_full)} ({len(X_train_full)/len(X)*100:.1f}%)")
print(f"测试集（完全独立）大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, shuffle=True
)

print(f"\n最终训练集大小: {len(X_train_final)} ({len(X_train_final)/len(X)*100:.1f}%)")
print(f"验证集大小: {len(X_val_final)} ({len(X_val_final)/len(X)*100:.1f}%)")
print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print("\n" + "="*60)
print("数据归一化流程（防止数据泄露）")
print("="*60)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

print("1. 仅使用最终训练集计算归一化参数...")
X_train_scaled = scaler_X.fit_transform(X_train_final)
y_train_scaled = scaler_y.fit_transform(y_train_final)

print("2. 使用训练集的归一化参数转换验证集和测试集...")
X_val_scaled = scaler_X.transform(X_val_final)
y_val_scaled = scaler_y.transform(y_val_final)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

print("3. 归一化完成！确保测试集信息未泄露到训练过程中")

print("\n" + "="*60)
print("传统机器学习模型训练数据准备")
print("="*60)

X_train_final_scaled_for_trad = scaler_X.transform(X_train_final)
y_train_final_scaled_for_trad = scaler_y.transform(y_train_final)

print(f"传统模型训练集大小: {len(X_train_final_scaled_for_trad)}")
print(f"传统模型测试集大小: {len(X_test_scaled)}")
print("注意：所有模型都使用相同的最终训练集数据，确保公平比较")

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)))
    return 1 - ss_res / (ss_tot + K.epsilon())

print("\n" + "="*60)
print("执行5折交叉验证")
print("="*60)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_final)):
    print(f"\n折 {fold+1}/5:")
    
    X_fold_train = X_train_final.iloc[train_idx]
    y_fold_train = y_train_final.iloc[train_idx]
    X_fold_val = X_train_final.iloc[val_idx]
    y_fold_val = y_train_final.iloc[val_idx]
    
    fold_scaler_X = MinMaxScaler()
    fold_scaler_y = MinMaxScaler()
    
    X_fold_train_scaled = fold_scaler_X.fit_transform(X_fold_train)
    y_fold_train_scaled = fold_scaler_y.fit_transform(y_fold_train)
    
    X_fold_val_scaled = fold_scaler_X.transform(X_fold_val)
    y_fold_val_scaled = fold_scaler_y.transform(y_fold_val)
    
    def create_fold_model(learning_rate=0.001):
        model = Sequential([
            InputLayer(shape=(X_fold_train_scaled.shape[1],)),
            Dense(49, activation='relu'),
            Dense(2, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), 
            loss='mean_squared_error', 
            metrics=['mean_squared_error']
        )
        return model
    
    model = create_fold_model()
    
    history = model.fit(
        X_fold_train_scaled, y_fold_train_scaled, 
        epochs=120, batch_size=64, 
        verbose=0, validation_data=(X_fold_val_scaled, y_fold_val_scaled)
    )
    
    val_score = model.evaluate(X_fold_val_scaled, y_fold_val_scaled, verbose=0)
    fold_scores.append(np.sqrt(val_score[0]))
    print(f"  验证集RMSE: {np.sqrt(val_score[0]):.6f}")

print(f"\n交叉验证完成!")
print(f"平均验证集RMSE: {np.mean(fold_scores):.6f}")
print(f"交叉验证标准差: {np.std(fold_scores):.6f}")

def create_model(learning_rate=0.0001, shape=(X_train_scaled.shape[1],)):
    model = Sequential()
    model.add(InputLayer(shape=shape))
    model.add(Dense(49, activation='relu'))
    model.add(Dense(2, activation='linear')) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error', root_mean_squared_error])
    return model

def lr_scheduler(epoch, lr):
    return lr * 0.90 if epoch > 50 else lr 

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

def rmse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    mse = mean_squared_error(y, y_pred, multioutput='uniform_average')
    rmse = np.sqrt(mse)
    return -rmse

model = KerasRegressor(model=create_model, epochs=120, batch_size=32, verbose=0)

param_grid = {
    'epochs': [100,150,200],
    'batch_size': [16,32,64],
    'model__learning_rate': [0.001,0.005,0.01,0.05],
}

print("\n" + "="*60)
print("开始嵌套交叉验证进行超参数调优")
print("="*60)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

outer_scores = []
best_models = []

for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_final)):
    print(f"\n外层折 {outer_fold+1}/5:")
    
    X_outer_train = X_train_final.iloc[train_idx]
    y_outer_train = y_train_final.iloc[train_idx]
    X_outer_val = X_train_final.iloc[val_idx]
    y_outer_val = y_train_final.iloc[val_idx]
    
    outer_scaler_X = MinMaxScaler()
    outer_scaler_y = MinMaxScaler()
    
    X_outer_train_scaled = outer_scaler_X.fit_transform(X_outer_train)
    y_outer_train_scaled = outer_scaler_y.fit_transform(y_outer_train)
    X_outer_val_scaled = outer_scaler_X.transform(X_outer_val)
    y_outer_val_scaled = outer_scaler_y.transform(y_outer_val)
    
    print(f"  在内层训练集上进行网格搜索...")
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=inner_cv,
        scoring=rmse_scorer,
        verbose=0
    )
    
    grid_search.fit(X_outer_train_scaled, y_outer_train_scaled)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(X_outer_val_scaled)
    if y_outer_val_scaled.ndim == 1:
        y_outer_val_scaled = y_outer_val_scaled.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    mse = mean_squared_error(y_outer_val_scaled, y_pred, multioutput='uniform_average')
    rmse = np.sqrt(mse)
    outer_scores.append(-rmse)
    
    best_models.append({
        'model': best_model,
        'params': best_params,
        'scaler_X': outer_scaler_X,
        'scaler_y': outer_scaler_y,
        'score': -rmse
    })
    
    print(f"  外层验证集RMSE: {rmse:.6f}")

best_model_info = min(best_models, key=lambda x: x['score'])
best_model = best_model_info['model']
best_params = best_model_info['params']

print(f"\n嵌套交叉验证完成!")
print(f"平均外层验证集负RMSE: {np.mean(outer_scores):.6f}")
print(f"最佳参数组合: {best_params}")
print(f"最佳外层验证集RMSE: {-best_model_info['score']:.6f}")

print("\n使用最佳参数训练最终模型...")
final_model = create_model(
    learning_rate=best_params['model__learning_rate'], 
    shape=(X_train_scaled.shape[1],)
)

early_stopping_final = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

final_model.compile(
    loss='mean_squared_error',
    optimizer=Adam(
        learning_rate=best_params['model__learning_rate'],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    ),
    metrics=[
        'mean_squared_error', 
        root_mean_squared_error, 
        r_squared, 
        tf.keras.metrics.MeanAbsoluteError(),
    ]
)

print("\n开始训练最终模型")
history = final_model.fit(
    X_train_scaled, 
    y_train_scaled,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[early_stopping_final],
    verbose=1
)

def evaluate_model_performance(y_true, y_pred, model_name="", dataset_name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    dry_true = y_true[:, 0]
    dry_pred = y_pred[:, 0]
    dry_rmse = np.sqrt(mean_squared_error(dry_true, dry_pred))
    dry_mae = mean_absolute_error(dry_true, dry_pred)
    dry_r2 = r2_score(dry_true, dry_pred)
    
    car_true = y_true[:, 1]
    car_pred = y_pred[:, 1]
    car_rmse = np.sqrt(mean_squared_error(car_true, car_pred))
    car_mae = mean_absolute_error(car_true, car_pred)
    car_r2 = r2_score(car_true, car_pred)
    
    print(f"\n{model_name} - {dataset_name}集性能:")
    print("  整体:")
    print(f"    ▪ RMSE: {rmse:.6f}")
    print(f"    ▪ MAE: {mae:.6f}")
    print(f"    ▪ R2: {r2:.6f}")
    
    print("\n  干重:")
    print(f"    ▪ RMSE: {dry_rmse:.6f}")
    print(f"    ▪ MAE: {dry_mae:.6f}")
    print(f"    ▪ R2: {dry_r2:.6f}")
    
    print("\n  胡萝卜素:")
    print(f"    ▪ RMSE: {car_rmse:.6f}")
    print(f"    ▪ MAE: {car_mae:.6f}")
    print(f"    ▪ R2: {car_r2:.6f}")
    
    return {
        'overall': {'RMSE': rmse, 'MAE': mae, 'R2': r2},
        'dry_weight': {'RMSE': dry_rmse, 'MAE': dry_mae, 'R2': dry_r2},
        'carotene': {'RMSE': car_rmse, 'MAE': car_mae, 'R2': car_r2}
    }

print(f"\n{'='*60}")
print("深度学习模型评估 (ANN)")
print(f"{'='*60}")

dl_train_pred_scaled = final_model.predict(X_train_scaled)
dl_train_pred = scaler_y.inverse_transform(dl_train_pred_scaled)
dl_train_true = scaler_y.inverse_transform(y_train_scaled)
dl_train_perf = evaluate_model_performance(dl_train_true, dl_train_pred, "ANN", "训练")

dl_val_pred_scaled = final_model.predict(X_val_scaled)
dl_val_pred = scaler_y.inverse_transform(dl_val_pred_scaled)
dl_val_true = scaler_y.inverse_transform(y_val_scaled)
dl_val_perf = evaluate_model_performance(dl_val_true, dl_val_pred, "ANN", "验证")

dl_test_pred_scaled = final_model.predict(X_test_scaled)
dl_test_pred = scaler_y.inverse_transform(dl_test_pred_scaled)
dl_test_true = scaler_y.inverse_transform(y_test_scaled)
dl_test_perf = evaluate_model_performance(dl_test_true, dl_test_pred, "ANN", "测试")

rf_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=236, random_state=42, max_depth=7)
)
rf_model.fit(X_train_final_scaled_for_trad, y_train_final_scaled_for_trad)

print(f"\n{'='*60}")
print("随机森林模型评估 (RF)")
print(f"{'='*60}")

rf_train_pred_scaled = rf_model.predict(X_train_final_scaled_for_trad)
rf_train_pred = scaler_y.inverse_transform(rf_train_pred_scaled)
rf_train_true = scaler_y.inverse_transform(y_train_final_scaled_for_trad)
rf_train_perf = evaluate_model_performance(rf_train_true, rf_train_pred, "RF", "训练")

rf_test_pred_scaled = rf_model.predict(X_test_scaled)
rf_test_pred = scaler_y.inverse_transform(rf_test_pred_scaled)
rf_test_true = scaler_y.inverse_transform(y_test_scaled)
rf_test_perf = evaluate_model_performance(rf_test_true, rf_test_pred, "RF", "测试")

xgb_model = MultiOutputRegressor(
    xgb.XGBRegressor(n_estimators=341, learning_rate=0.1, max_depth=5, random_state=42)
)
xgb_model.fit(X_train_final_scaled_for_trad, y_train_final_scaled_for_trad)

print(f"\n{'='*60}")
print("XGBoost模型评估")
print(f"{'='*60}")

xgb_train_pred_scaled = xgb_model.predict(X_train_final_scaled_for_trad)
xgb_train_pred = scaler_y.inverse_transform(xgb_train_pred_scaled)
xgb_train_true = scaler_y.inverse_transform(y_train_final_scaled_for_trad)
xgb_train_perf = evaluate_model_performance(xgb_train_true, xgb_train_pred, "XGBoost", "训练")

xgb_test_pred_scaled = xgb_model.predict(X_test_scaled)
xgb_test_pred = scaler_y.inverse_transform(xgb_test_pred_scaled)
xgb_test_true = scaler_y.inverse_transform(y_test_scaled)
xgb_test_perf = evaluate_model_performance(xgb_test_true, xgb_test_pred, "XGBoost", "测试")

def create_gbdt_model():
    return GradientBoostingRegressor(
        n_estimators=242, learning_rate=0.1, max_depth=4, random_state=42
    )

gbdt_models = []
for target in y.columns:
    model = create_gbdt_model()
    model.fit(X_train_final_scaled_for_trad, y_train_final_scaled_for_trad[:, list(y.columns).index(target)])
    gbdt_models.append(model)

print(f"\n{'='*60}")
print("GBDT模型评估")
print(f"{'='*60}")

def predict_gbdt(models, X):
    preds = []
    for model in models:
        preds.append(model.predict(X))
    return np.column_stack(preds)

gbdt_train_pred_scaled = predict_gbdt(gbdt_models, X_train_final_scaled_for_trad)
gbdt_train_pred = scaler_y.inverse_transform(gbdt_train_pred_scaled)
gbdt_train_true = scaler_y.inverse_transform(y_train_final_scaled_for_trad)
gbdt_train_perf = evaluate_model_performance(gbdt_train_true, gbdt_train_pred, "GBDT", "训练")

gbdt_test_pred_scaled = predict_gbdt(gbdt_models, X_test_scaled)
gbdt_test_pred = scaler_y.inverse_transform(gbdt_test_pred_scaled)
gbdt_test_true = scaler_y.inverse_transform(y_test_scaled)
gbdt_test_perf = evaluate_model_performance(gbdt_test_true, gbdt_test_pred, "GBDT", "测试")

def export_model_results(model_name, y_true, y_pred, desktop_path):
    try:
        df_overall = pd.DataFrame({
            'Actual_Dry_Weight': y_true[:, 0],
            'Predicted_Dry_Weight': y_pred[:, 0],
            'Actual_Carotene': y_true[:, 1],
            'Predicted_Carotene': y_pred[:, 1]
        })
        
        df_overall.to_excel(os.path.join(desktop_path, f"{model_name}_Overall_Evaluation.xlsx"), index=False)
        print(f"✅ {model_name}模型结果已导出到 {desktop_path}")
        return True
    except Exception as e:
        print(f"❌ 导出{model_name}结果时出错: {str(e)}")
        return False

desktop_path = r'C:\Users\Desktop'#Output the results to the desktop
os.makedirs(desktop_path, exist_ok=True)

print("\n" + "="*60)
print("开始导出模型预测结果...")
print("="*60)

export_success = []

try:
    y_pred_scaled = final_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled)
    export_success.append(export_model_results("ANN", y_test_original, y_pred, desktop_path))
except Exception as e:
    print(f"❌ 获取ANN预测结果时出错: {str(e)}")
    export_success.append(False)

try:
    rf_pred_scaled = rf_model.predict(X_test_scaled)
    rf_pred = scaler_y.inverse_transform(rf_pred_scaled)
    export_success.append(export_model_results("RF", y_test.values, rf_pred, desktop_path))
except Exception as e:
    print(f"❌ 获取RF预测结果时出错: {str(e)}")
    export_success.append(False)

try:
    y_pred_xgb_scaled = xgb_model.predict(X_test_scaled)
    y_pred_xgb = scaler_y.inverse_transform(y_pred_xgb_scaled)
    export_success.append(export_model_results("XGBoost", y_test.values, y_pred_xgb, desktop_path))
except Exception as e:
    print(f"❌ 获取XGBoost预测结果时出错: {str(e)}")
    export_success.append(False)

try:
    gbdt_pred_dry_scaled = gbdt_models[0].predict(X_test_scaled)
    gbdt_pred_carotene_scaled = gbdt_models[1].predict(X_test_scaled)
    gbdt_pred_scaled = np.column_stack((gbdt_pred_dry_scaled, gbdt_pred_carotene_scaled))
    gbdt_pred = scaler_y.inverse_transform(gbdt_pred_scaled)
    export_success.append(export_model_results("GBDT", y_test.values, gbdt_pred, desktop_path))
except Exception as e:
    print(f"❌ 获取GBDT预测结果时出错: {str(e)}")
    export_success.append(False)

success_count = sum(export_success)
print(f"\n导出完成! 成功导出 {success_count}/4 个模型的数据")

print("\n" + "="*80)
print("开始多目标优化过程")
print("="*80)

class MOPSO:
    
    def __init__(self, objective_func, bounds, integer_indices, n_particles=100, max_iter=100, 
                 archive_size=50, w_range=(0.9, 0.4), c1=1.494, c2=1.494, 
                 constriction_factor=0.729):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.integer_indices = integer_indices
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.w_range = w_range
        self.c1 = c1
        self.c2 = c2
        self.constriction_factor = constriction_factor
        
        self.dim = len(bounds)
        
        self.positions = np.random.rand(n_particles, self.dim)
        for d in range(self.dim):
            self.positions[:, d] = self.bounds[d, 0] + (self.bounds[d, 1] - self.bounds[d, 0]) * self.positions[:, d]
        
        self._ensure_integer_values()
        
        self.velocities = np.zeros((n_particles, self.dim))
        
        self.objectives = np.array([objective_func(pos) for pos in self.positions])
        
        self.pbest_positions = self.positions.copy()
        self.pbest_objectives = self.objectives.copy()
        
        self.archive_positions = []
        self.archive_objectives = []
        
        self.gbest_position = None
    
    def _ensure_integer_values(self):
        for idx in self.integer_indices:
            self.positions[:, idx] = np.round(self.positions[:, idx])
            self.positions[:, idx] = np.clip(self.positions[:, idx], self.bounds[idx, 0], self.bounds[idx, 1])
    
    def is_dominated(self, obj1, obj2):
        return (obj2[0] >= obj1[0] and obj2[1] >= obj1[1]) and \
               (obj2[0] > obj1[0] or obj2[1] > obj1[1])
    
    def non_dominated_sorting(self, objectives):
        n = len(objectives)
        domination_counts = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i+1, n):
                if self.is_dominated(objectives[i], objectives[j]):
                    domination_counts[i] += 1
                    dominated_solutions[j].append(i)
                elif self.is_dominated(objectives[j], objectives[i]):
                    domination_counts[j] += 1
                    dominated_solutions[i].append(j)
        
        for i in range(n):
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def crowding_distance(self, objectives, front_indices):
        n = len(front_indices)
        if n == 0:
            return np.zeros(0)
        
        distances = np.zeros(n)
        
        for m in range(2):
            front_objectives = objectives[front_indices]
            
            sorted_indices = sorted(range(n), key=lambda i: front_objectives[i, m])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            if n > 2:
                max_val = front_objectives[sorted_indices[-1], m]
                min_val = front_objectives[sorted_indices[0], m]
                
                if max_val != min_val:
                    for i in range(1, n-1):
                        distances[sorted_indices[i]] += (
                            front_objectives[sorted_indices[i+1], m] - 
                            front_objectives[sorted_indices[i-1], m]
                        ) / (max_val - min_val)
        
        return distances
    
    def update_archive(self):
        if len(self.archive_positions) == 0:
            all_positions = self.positions
            all_objectives = self.objectives
        else:
            all_positions = np.vstack([self.positions, np.array(self.archive_positions)])
            all_objectives = np.vstack([self.objectives, np.array(self.archive_objectives)])
        
        fronts = self.non_dominated_sorting(all_objectives)
        
        self.archive_positions = []
        self.archive_objectives = []
        
        for front in fronts:
            front_indices = list(front)
            
            if len(self.archive_positions) + len(front_indices) <= self.archive_size:
                for idx in front_indices:
                    self.archive_positions.append(all_positions[idx])
                    self.archive_objectives.append(all_objectives[idx])
            else:
                remaining = self.archive_size - len(self.archive_positions)
                if remaining > 0:
                    distances = self.crowding_distance(all_objectives, front_indices)
                    
                    indexed_distances = list(zip(front_indices, distances))
                    sorted_pairs = sorted(indexed_distances, key=lambda x: x[1], reverse=True)
                    
                    for i in range(remaining):
                        idx, _ = sorted_pairs[i]
                        self.archive_positions.append(all_positions[idx])
                        self.archive_objectives.append(all_objectives[idx])
                break
        
        if len(self.archive_positions) > 0:
            self.archive_positions = np.array(self.archive_positions)
            self.archive_objectives = np.array(self.archive_objectives)
        else:
            self.archive_positions = np.array([]).reshape(0, self.dim)
            self.archive_objectives = np.array([]).reshape((0, 2))
    
    def select_leader(self):
        if len(self.archive_objectives) == 0 or len(self.archive_positions) == 0:
            return None
        
        if len(self.archive_objectives) == 1:
            return self.archive_positions[0]
        
        distances = self.crowding_distance(self.archive_objectives, list(range(len(self.archive_objectives))))
        
        if np.all(distances == 0):
            distances = np.ones(len(distances))
        
        if np.sum(distances) == 0:
            probabilities = np.ones(len(distances)) / len(distances)
        else:
            probabilities = distances / np.sum(distances)
        
        probabilities = probabilities / np.sum(probabilities)
        
        if np.any(np.isnan(probabilities)) or np.any(probabilities < 0):
            probabilities = np.ones(len(distances)) / len(distances)
        
        try:
            selected_idx = np.random.choice(len(self.archive_objectives), p=probabilities)
            return self.archive_positions[selected_idx]
        except:
            return self.archive_positions[0]
    
    def optimize(self):
        print(f"MOPSO优化开始: {self.n_particles}个粒子, {self.max_iter}次迭代")
        
        for iteration in range(self.max_iter):
            w = self.w_range[0] - (self.w_range[0] - self.w_range[1]) * (iteration / self.max_iter)
            
            self.update_archive()
            
            self.gbest_position = self.select_leader()
            
            if self.gbest_position is None:
                self.gbest_position = self.positions[np.random.randint(0, self.n_particles)]
            
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                
                self.velocities[i] = self.constriction_factor * (
                    w * self.velocities[i] + cognitive + social
                )
                
                self.positions[i] += self.velocities[i]
                
                for d in range(self.dim):
                    if self.positions[i, d] < self.bounds[d, 0]:
                        self.positions[i, d] = self.bounds[d, 0]
                        self.velocities[i, d] *= -0.5
                    elif self.positions[i, d] > self.bounds[d, 1]:
                        self.positions[i, d] = self.bounds[d, 1]
                        self.velocities[i, d] *= -0.5
                
                for idx in self.integer_indices:
                    self.positions[i, idx] = round(self.positions[i, idx])
                    if self.positions[i, idx] < self.bounds[idx, 0]:
                        self.positions[i, idx] = self.bounds[idx, 0]
                    elif self.positions[i, idx] > self.bounds[idx, 1]:
                        self.positions[i, idx] = self.bounds[idx, 1]
                
                new_objectives = self.objective_func(self.positions[i])
                
                dominated_by_new = self.is_dominated(self.pbest_objectives[i], new_objectives)
                dominated_by_pbest = self.is_dominated(new_objectives, self.pbest_objectives[i])
                
                if dominated_by_new:
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_objectives[i] = new_objectives.copy()
                elif not dominated_by_pbest:
                    if np.random.rand() < 0.5:
                        self.pbest_positions[i] = self.positions[i].copy()
                        self.pbest_objectives[i] = new_objectives.copy()
            
            self.objectives = np.array([self.objective_func(pos) for pos in self.positions])
            
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration+1}/{self.max_iter}, 存档大小: {len(self.archive_objectives)}")
        
        self.update_archive()
        
        if len(self.archive_positions) > 0:
            for idx in self.integer_indices:
                self.archive_positions[:, idx] = np.round(self.archive_positions[:, idx])
                self.archive_positions[:, idx] = np.clip(self.archive_positions[:, idx], 
                                                         self.bounds[idx, 0], self.bounds[idx, 1])
        
        print(f"优化完成! 找到 {len(self.archive_objectives)} 个帕累托最优解")
        
        return self.archive_positions, self.archive_objectives

def objective_function(x):
    x_clipped = np.clip(x, X_train_final.min().values, X_train_final.max().values)
    
    temp_idx = feature_names.index('temperature')
    x_clipped[temp_idx] = round(x_clipped[temp_idx])  
    
    x_df = pd.DataFrame([x_clipped], columns=feature_names)
    x_scaled = scaler_X.transform(x_df) 
    y_pred_scaled = final_model.predict(x_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]  
    
    dry_weight = max(0, y_pred[0])
    carotene = max(0, y_pred[1])
    
    return np.array([dry_weight, carotene])

def pso(cost_func, lb, ub, integer_indices, swarmsize=100, maxiter=100, debug=True):
    dim = len(lb)
    
    swarm = np.random.rand(swarmsize, dim)
    velocity = np.zeros((swarmsize, dim))
    for d in range(dim):
        swarm[:, d] = lb[d] + (ub[d] - lb[d]) * swarm[:, d]
    
    for idx in integer_indices:
        swarm[:, idx] = np.round(swarm[:, idx])
        swarm[:, idx] = np.clip(swarm[:, idx], lb[idx], ub[idx])
    
    pbest_position = swarm.copy()
    pbest_value = np.array([cost_func(swarm[i]) for i in range(swarmsize)])
    gbest_idx = np.argmin(pbest_value)
    gbest_position = swarm[gbest_idx].copy()
    gbest_value = pbest_value[gbest_idx]
    
    for it in range(maxiter):
        w = 0.9 - (0.9 - 0.4) * (it / maxiter)
        
        for i in range(swarmsize):
            for d in range(dim):
                r1, r2 = np.random.rand(2)
                cognitive = 1.494 * r1 * (pbest_position[i, d] - swarm[i, d])
                social = 1.494 * r2 * (gbest_position[d] - swarm[i, d])
                
                velocity[i, d] = 0.729 * (w * velocity[i, d] + cognitive + social)
                
                swarm[i, d] += velocity[i, d]
                
                if swarm[i, d] < lb[d]:
                    swarm[i, d] = lb[d]
                elif swarm[i, d] > ub[d]:
                    swarm[i, d] = ub[d]
            
            for idx in integer_indices:
                swarm[i, idx] = round(swarm[i, idx])
                if swarm[i, idx] < lb[idx]:
                    swarm[i, idx] = lb[idx]
                elif swarm[i, idx] > ub[idx]:
                    swarm[i, idx] = ub[idx]
        
        current_values = np.array([cost_func(swarm[i]) for i in range(swarmsize)])
        
        for i in range(swarmsize):
            if current_values[i] < pbest_value[i]:
                pbest_value[i] = current_values[i]
                pbest_position[i] = swarm[i].copy()
        
        current_best_idx = np.argmin(current_values)
        if current_values[current_best_idx] < gbest_value:
            gbest_value = current_values[current_best_idx]
            gbest_position = swarm[current_best_idx].copy()
        
        if debug and (it + 1) % 10 == 0:
            print(f"迭代 {it+1}/{maxiter} - 最佳适应度: {gbest_value:.6f}")
    
    return gbest_position, gbest_value

def weighted_sum_optimization(alpha, n_particles=100, max_iter=100):
    print(f"\n加权求和法优化: α = {alpha:.2f}")
    
    temp_idx = feature_names.index('temperature')
    integer_indices = [temp_idx]
    
    def weighted_objective(x):
        x_clipped = np.clip(x, X_train_final.min().values, X_train_final.max().values)
        x_clipped[temp_idx] = round(x_clipped[temp_idx])
        
        objectives = objective_function(x_clipped)
        return -(alpha * objectives[0] + (1 - alpha) * objectives[1])
    
    bounds = []
    for col in feature_names:
        min_val = X_train_final[col].min()
        max_val = X_train_final[col].max()
        bounds.append((min_val, max_val))
    
    best_position, best_value = pso(
        weighted_objective,
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        integer_indices=integer_indices,
        swarmsize=n_particles,
        maxiter=max_iter,
        debug=False
    )
    
    best_position = np.clip(best_position, [b[0] for b in bounds], [b[1] for b in bounds])
    
    best_position[temp_idx] = round(best_position[temp_idx])
    
    objectives = objective_function(best_position)
    
    print(f"优化结果: 干重 = {objectives[0]:.4f} g/L, 胡萝卜素 = {objectives[1]:.4f} mg/L")
    
    return {
        'alpha': alpha,
        'conditions': best_position,
        'dry_weight': objectives[0],
        'carotene': objectives[1],
        'type': '加权求和解'
    }

ALPHA_VALUES = np.arange(0, 1.1, 0.1)
print(f"权重因子α数量: {len(ALPHA_VALUES)}个 (从0.0到1.0)")

temp_idx = feature_names.index('temperature')
integer_indices = [temp_idx]

print("\n" + "="*60)
print("开始加权求和法优化")
print("="*60)
weighted_solutions = []

for alpha in ALPHA_VALUES:
    solution = weighted_sum_optimization(alpha, n_particles=100, max_iter=100)
    weighted_solutions.append(solution)

print("\n" + "="*60)
print("开始MOPSO优化帕累托前沿")
print("="*60)

bounds = []
for col in feature_names:
    min_val = X_train_final[col].min()
    max_val = X_train_final[col].max()
    bounds.append((min_val, max_val))

mopso = MOPSO(
    objective_func=objective_function,
    bounds=bounds,
    integer_indices=integer_indices,
    n_particles=100,           
    max_iter=100,              
    archive_size=50,         
    w_range=(0.9, 0.4),      
    c1=1.494,                 
    c2=1.494,                  
)

pareto_positions, pareto_objectives = mopso.optimize()

pareto_positions_clipped = []
for pos in pareto_positions:
    pos_clipped = np.clip(pos, [b[0] for b in bounds], [b[1] for b in bounds])
    pos_clipped[temp_idx] = round(pos_clipped[temp_idx])
    pareto_positions_clipped.append(pos_clipped)
pareto_positions_clipped = np.array(pareto_positions_clipped)

pareto_objectives_clipped = np.array([objective_function(pos) for pos in pareto_positions_clipped])

pareto_solutions = []
for pos, obj in zip(pareto_positions_clipped, pareto_objectives_clipped):
    pareto_solutions.append({
        'conditions': pos,
        'dry_weight': obj[0],
        'carotene': obj[1],
        'type': '帕累托解'
    })

all_solutions = weighted_solutions + pareto_solutions
print(f"\n总解数: {len(all_solutions)} (加权求和解:{len(weighted_solutions)}, 帕累托解:{len(pareto_solutions)})")

dry_weights = []
carotene_yields = []
solution_types = []
alpha_values = []
conditions_list = []

for sol in all_solutions:
    dry_weights.append(sol['dry_weight'])
    carotene_yields.append(sol['carotene'])
    solution_types.append(sol['type'])
    alpha_values.append(sol.get('alpha', 0.5))
    conditions_list.append(sol['conditions'])

plt.figure(figsize=(12, 8), dpi=300)
ax = plt.gca()

norm = plt.Normalize(min(alpha_values), max(alpha_values))
cmap = cm.viridis

weighted_idx = [i for i, t in enumerate(solution_types) if t == '加权求和解']
if weighted_idx:
    scatter1 = ax.scatter(
        [dry_weights[i] for i in weighted_idx],
        [carotene_yields[i] for i in weighted_idx],
        c=[alpha_values[i] for i in weighted_idx],
        cmap=cmap,
        s=200,
        marker='o',
        edgecolor='k',
        alpha=0.9,
        zorder=20,
        label='加权求和解'
    )

pareto_idx = [i for i, t in enumerate(solution_types) if t == '帕累托解']
if pareto_idx:
    scatter2 = ax.scatter(
        [dry_weights[i] for i in pareto_idx],
        [carotene_yields[i] for i in pareto_idx],
        c=[alpha_values[i] for i in pareto_idx],
        cmap=cmap,
        s=80,
        marker='s',
        edgecolor='k',
        alpha=0.7,
        zorder=10,
        label='帕累托解'
    )

if dry_weights:
    max_dry_idx = np.argmax(dry_weights)
    point1 = ax.scatter(
        dry_weights[max_dry_idx], 
        carotene_yields[max_dry_idx],
        s=250,
        marker='*',
        color='gold',
        edgecolor='k',
        alpha=1.0,
        zorder=30,
        label='最大干重'
    )
    ax.annotate(f"干重最高点\n({dry_weights[max_dry_idx]:.4f} g/L)", 
                (dry_weights[max_dry_idx], carotene_yields[max_dry_idx]),
                xytext=(10, 15),
                textcoords="offset points",
                ha='left',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc='gold', alpha=0.5))

if carotene_yields:
    max_caro_idx = np.argmax(carotene_yields)
    point2 = ax.scatter(
        dry_weights[max_caro_idx], 
        carotene_yields[max_caro_idx],
        s=250,
        marker='D',
        color='orange',
        edgecolor='k',
        alpha=1.0,
        zorder=30,
        label='最大胡萝卜素'
    )
    ax.annotate(f"胡萝卜素最高点\n({carotene_yields[max_caro_idx]:.4f} mg/L)", 
                (dry_weights[max_caro_idx], carotene_yields[max_caro_idx]),
                xytext=(-10, -20),
                textcoords="offset points",
                ha='right',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc='orange', alpha=0.5))

for i in weighted_idx:
    alpha_val = alpha_values[i]
    ax.annotate(f"α={alpha_val:.1f}", 
               (dry_weights[i], carotene_yields[i]),
               xytext=(5, 5),
               textcoords="offset points",
               ha='left',
               fontsize=9,
               color='darkblue',
               alpha=0.9)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('权重因子 α (干重权重)', fontsize=12)

ax.set_xlabel('干重产量 (g/L)', fontsize=14)
ax.set_ylabel('胡萝卜素产量 (mg/L)', fontsize=14)
ax.set_title('多目标优化结果: 加权求和解与帕累托前沿解', fontsize=16)
ax.grid(alpha=0.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

image_path = os.path.join(desktop_path, "Pareto_Front.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n优化结果可视化已保存至: {image_path}")

results_data = {
    '类型': solution_types,
    '权重α': alpha_values,
    '干重': dry_weights,
    '胡萝卜素': carotene_yields
}

for i, feature in enumerate(feature_names):
    results_data[feature] = [cond[i] for cond in conditions_list]

results_df = pd.DataFrame(results_data)

weighted_df = results_df[results_df['类型'] == '加权求和解'].copy()
pareto_df = results_df[results_df['类型'] == '帕累托解'].copy()

weighted_path = os.path.join(desktop_path, "加权求和解.csv")
pareto_path = os.path.join(desktop_path, "帕累托前沿解.csv")
all_path = os.path.join(desktop_path, "所有优化解.csv")

weighted_df.to_csv(weighted_path, index=False)
pareto_df.to_csv(pareto_path, index=False)
results_df.to_csv(all_path, index=False)

print("\n优化结果已保存:")
print(f"  加权求和解: {weighted_path}")
print(f"  帕累托前沿解: {pareto_path}")
print(f"  所有优化解: {all_path}")

print("\n" + "="*60)
print("优化结果与训练数据对比分析")
print("="*60)

max_dry_train = y_train_final['Dry weight '].max()
max_caro_train = y_train_final['Carotene yield'].max()

max_dry_optimized = max(dry_weights)
max_caro_optimized = max(carotene_yields)

print(f"训练数据最大干重: {max_dry_train:.4f} g/L")
print(f"优化后最大干重: {max_dry_optimized:.4f} g/L")
print(f"  提升: {(max_dry_optimized - max_dry_train) / max_dry_train * 100:.2f}%")

print(f"\n训练数据最大胡萝卜素: {max_caro_train:.4f} mg/L")
print(f"优化后最大胡萝卜素: {max_caro_optimized:.4f} mg/L")
print(f"  提升: {(max_caro_optimized - max_caro_train) / max_caro_train * 100:.2f}%")

print("\n特殊点优化条件（确保在训练数据范围内）:")
print("-"*60)

def check_integer_values(conditions, solution_type):
    temp_val = conditions[temp_idx]
    temp_int = temp_val == round(temp_val)
    
    result = []
    if temp_int:
        result.append("✅ 温度为整数")
    else:
        result.append(f"❌ 温度{temp_val}不是整数")
    return result

max_dry_sol = all_solutions[max_dry_idx]
print(f"最大干重点 (类型: {max_dry_sol['type']}):")
for feature, value in zip(feature_names, max_dry_sol['conditions']):
    train_min = X_train_final[feature].min()
    train_max = X_train_final[feature].max()
    in_range = "✅ 范围内" if train_min <= value <= train_max else "❌ 超出范围"
    print(f"  {feature}: {value:.4f} (训练范围: {train_min:.4f}-{train_max:.4f}) {in_range}")

int_checks = check_integer_values(max_dry_sol['conditions'], max_dry_sol['type'])
for check in int_checks:
    print(f"  {check}")

print("-"*60)

max_caro_sol = all_solutions[max_caro_idx]
print(f"最大胡萝卜素点 (类型: {max_caro_sol['type']}):")
for feature, value in zip(feature_names, max_caro_sol['conditions']):
    train_min = X_train_final[feature].min()
    train_max = X_train_final[feature].max()
    in_range = "✅ 范围内" if train_min <= value <= train_max else "❌ 超出范围"
    print(f"  {feature}: {value:.4f} (训练范围: {train_min:.4f}-{train_max:.4f}) {in_range}")

int_checks = check_integer_values(max_caro_sol['conditions'], max_caro_sol['type'])
for check in int_checks:
    print(f"  {check}")

print("\n" + "="*60)
print("优化条件边界检查（训练数据范围）")
print("="*60)

out_of_range_count = 0
non_integer_count = 0
for i, sol in enumerate(all_solutions):
    for j, feature in enumerate(feature_names):
        value = sol['conditions'][j]
        train_min = X_train_final[feature].min()
        train_max = X_train_final[feature].max()
        if value < train_min or value > train_max:
            out_of_range_count += 1
            if out_of_range_count <= 5:
                print(f"警告: 解{i}的{feature} = {value:.4f}, 超出训练范围 [{train_min:.4f}, {train_max:.4f}]")
    
    temp_val = sol['conditions'][temp_idx]
    if not (temp_val == round(temp_val)):
        non_integer_count += 1
        if non_integer_count <= 5:
            print(f"警告: 解{i}的温度 = {temp_val:.4f}, 不是整数")

if out_of_range_count == 0:
    print("✅ 所有优化条件都在训练数据范围内!")
else:
    print(f"⚠️  发现 {out_of_range_count} 个超出训练数据范围的优化条件")

if non_integer_count == 0:
    print("✅ 所有温度都是整数!")
else:
    print(f"⚠️  发现 {non_integer_count} 个温度不是整数的情况")

print("\n多目标优化完成!")
print("\n所有功能执行完毕!")

print("\n" + "="*60)
print("早停机制执行结果")
print("="*60)
print(f"最终模型在训练过程中使用了早停机制")
print(f"模型在 epoch {len(history.history['loss'])} 停止训练")
print(f"最佳验证损失: {min(history.history['val_loss']):.6f}")
print(f"最终验证损失: {history.history['val_loss'][-1]:.6f}")
