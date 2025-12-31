#Single-objective model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import make_scorer
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import os

df = pd.read_csv(r'C:\Users\data.csv', encoding='ISO-8859-1')#import data
print("原始数据缺失值数量：", df.isnull().sum())

df = df.fillna(df.mean())
y = df[['Dry weight ', 'Carotene yield']]
print("原始列名：", df.columns)

X = df[['light_intensity', 'temperature', 'hormone','bicarbonate', 
        'nitrogen_source', 'phosphorus_source', 'cultivation_time', 'nacl']]

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
print("5折交叉验证开始")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final)):
    print(f"\n第 {fold+1} 折交叉验证:")
    
    X_train_cv = X_train_final.iloc[train_idx]
    X_val_cv = X_train_final.iloc[val_idx]
    y_train_cv_dry = y_train_final.iloc[train_idx]['Dry weight ']
    y_train_cv_car = y_train_final.iloc[train_idx]['Carotene yield']
    y_val_cv_dry = y_train_final.iloc[val_idx]['Dry weight ']
    y_val_cv_car = y_train_final.iloc[val_idx]['Carotene yield']
    
    scaler_X_cv = MinMaxScaler()
    scaler_y1_cv = MinMaxScaler()
    scaler_y2_cv = MinMaxScaler()
    
    X_train_cv_scaled = scaler_X_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_X_cv.transform(X_val_cv)
    
    y_train_cv_dry_scaled = scaler_y1_cv.fit_transform(y_train_cv_dry.values.reshape(-1, 1))
    y_train_cv_car_scaled = scaler_y2_cv.fit_transform(y_train_cv_car.values.reshape(-1, 1))
    
    cv_results.append({
        'fold': fold+1,
        'train_size': len(X_train_cv),
        'val_size': len(X_val_cv),
        'X_train': X_train_cv_scaled,
        'X_val': X_val_cv_scaled,
        'y_train_dry': y_train_cv_dry_scaled,
        'y_val_dry': scaler_y1_cv.transform(y_val_cv_dry.values.reshape(-1, 1)),
        'y_train_car': y_train_cv_car_scaled,
        'y_val_car': scaler_y2_cv.transform(y_val_cv_car.values.reshape(-1, 1)),
        'y_val_dry_original': y_val_cv_dry.values,
        'y_val_car_original': y_val_cv_car.values,
        'scaler_y1': scaler_y1_cv,
        'scaler_y2': scaler_y2_cv
    })
    
    print(f"  训练集大小: {len(X_train_cv)}, 验证集大小: {len(X_val_cv)}")

print("\n5折交叉验证完成!")

print("\n" + "="*60)
print("数据归一化流程（防止数据泄露）")
print("="*60)

scaler_X = MinMaxScaler()
scaler_y1 = MinMaxScaler()
scaler_y2 = MinMaxScaler()

print("1. 仅使用最终训练集计算归一化参数...")
X_train_scaled = scaler_X.fit_transform(X_train_final)

y_train_dry = y_train_final['Dry weight '].values.reshape(-1, 1)
y_train_car = y_train_final['Carotene yield'].values.reshape(-1, 1)

y_train_scaled_dry = scaler_y1.fit_transform(y_train_dry)
y_train_scaled_car = scaler_y2.fit_transform(y_train_car)

print("2. 使用训练集的归一化参数转换验证集和测试集...")
X_val_scaled = scaler_X.transform(X_val_final)
X_test_scaled = scaler_X.transform(X_test)

y_val_dry = y_val_final['Dry weight '].values.reshape(-1, 1)
y_val_car = y_val_final['Carotene yield'].values.reshape(-1, 1)
y_val_scaled_dry = scaler_y1.transform(y_val_dry)
y_val_scaled_car = scaler_y2.transform(y_val_car)

y_test_dry = y_test['Dry weight '].values.reshape(-1, 1)
y_test_car = y_test['Carotene yield'].values.reshape(-1, 1)
y_test_scaled_dry = scaler_y1.transform(y_test_dry)
y_test_scaled_car = scaler_y2.transform(y_test_car)

print("3. 归一化完成！确保测试集信息未泄露到训练过程中")

print("\n" + "="*60)
print("传统机器学习模型训练数据准备")
print("="*60)

X_train_final_scaled_for_trad = scaler_X.transform(X_train_final)

print(f"传统模型训练集大小: {len(X_train_final_scaled_for_trad)}")
print(f"传统模型测试集大小: {len(X_test_scaled)}")
print("注意：所有模型都使用相同的最终训练集数据，确保公平比较")

def create_single_output_model(target, learning_rate=0.001, shape=(X_train_scaled.shape[1],)):
    model = Sequential()
    
    if target == 'Dry weight ':
        model.add(InputLayer(shape=shape))
        model.add(Dense(57, activation='relu'))
        model.add(Dense(1, activation='linear'))
    elif target == 'Carotene yield':
        model.add(InputLayer(shape=shape))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(1, activation='linear'))
    else:
        raise ValueError("Invalid target name")
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)))
    return 1 - ss_res / (ss_tot + K.epsilon())

def mean_absolute_error_custom(y_true, y_pred):    
    return K.mean(K.abs(y_pred - y_true))

def mean_absolute_deviation(y_true, y_pred):
    return K.mean(K.abs(y_true - K.mean(y_true, axis=0)))

def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return -rmse

rmse_scorer = make_scorer(rmse_score, greater_is_better=True)

def model_builder(target):
    def build_model(learning_rate):
        return create_single_output_model(target, learning_rate=learning_rate)
    return KerasRegressor(model=build_model, epochs=50, batch_size=32, verbose=0)

param_grid = {
    'epochs': [100,150,200],
    'batch_size': [16,32,64],
    'model__learning_rate': [0.001,0.005,0.01,0.05],
}

model1 = model_builder('Dry weight ')
model2 = model_builder('Carotene yield')

grid_search1 = GridSearchCV(estimator=model1, param_grid=param_grid, cv=5, scoring=rmse_scorer)
grid_search2 = GridSearchCV(estimator=model2, param_grid=param_grid, cv=5, scoring=rmse_scorer)

print("正在为干重模型进行超参数优化")
grid_search1.fit(X_train_scaled, y_train_scaled_dry)
print("干重模型超参数优化完成！")

print("正在为胡萝卜素模型进行超参数优化")
grid_search2.fit(X_train_scaled, y_train_scaled_car)
print("胡萝卜素模型超参数优化完成！")

best_model1 = grid_search1.best_estimator_
best_model2 = grid_search2.best_estimator_

print(f"干重模型最佳参数: {grid_search1.best_params_}")
print(f"胡萝卜素模型最佳参数: {grid_search2.best_params_}")
print(f"干重模型最佳交叉验证RMSE: {-grid_search1.best_score_:.6f}")
print(f"胡萝卜素模型最佳交叉验证RMSE: {-grid_search2.best_score_:.6f}")

class HistoryCollector(Callback):
    def __init__(self):
        super(HistoryCollector, self).__init__()
        self.history = {
            'loss': [], 'val_loss': [],
            'mean_squared_error': [], 'val_mean_squared_error': [],
            'root_mean_squared_error': [], 'val_root_mean_squared_error': [],
            'mean_absolute_error': [], 'val_mean_absolute_error': [],
            'r_squared': [], 'val_r_squared': []
        }

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['mean_squared_error'].append(logs.get('mean_squared_error'))
        self.history['val_mean_squared_error'].append(logs.get('val_mean_squared_error'))
        self.history['root_mean_squared_error'].append(logs.get('root_mean_squared_error'))
        self.history['val_root_mean_squared_error'].append(logs.get('val_root_mean_squared_error'))
        self.history['mean_absolute_error'].append(logs.get('mean_absolute_error'))
        self.history['val_mean_absolute_error'].append(logs.get('val_mean_absolute_error'))
        self.history['r_squared'].append(logs.get('r_squared'))
        self.history['val_r_squared'].append(logs.get('val_r_squared'))

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

best_model1.model = create_single_output_model('Dry weight ', learning_rate=grid_search1.best_params_['model__learning_rate'])
best_model1.model.compile(
    optimizer=Adam(grid_search1.best_params_['model__learning_rate']),
    loss='mean_squared_error',
    metrics=['mean_squared_error', root_mean_squared_error, r_squared, mean_absolute_error_custom]
)
history_collector1 = HistoryCollector()
print("训练干重模型")
history1 = best_model1.model.fit(
    X_train_scaled, y_train_scaled_dry,
    epochs=grid_search1.best_params_['epochs'],
    batch_size=grid_search1.best_params_['batch_size'],
    validation_data=(X_val_scaled, y_val_scaled_dry),
    callbacks=[history_collector1, early_stopping],
    verbose=1
)
print(f"干重模型训练完成！在 {len(history1.history['loss'])} 个epoch后停止")

best_model2.model = create_single_output_model('Carotene yield', learning_rate=grid_search2.best_params_['model__learning_rate'])
best_model2.model.compile(
    optimizer=Adam(grid_search2.best_params_['model__learning_rate']),
    loss='mean_squared_error',
    metrics=['mean_squared_error', root_mean_squared_error, r_squared, mean_absolute_error_custom]
)
history_collector2 = HistoryCollector()
print("训练胡萝卜素模型")
history2 = best_model2.model.fit(
    X_train_scaled, y_train_scaled_car,
    epochs=grid_search2.best_params_['epochs'],
    batch_size=grid_search2.best_params_['batch_size'],
    validation_data=(X_val_scaled, y_val_scaled_car),
    callbacks=[history_collector2, early_stopping],
    verbose=1
)
print(f"胡萝卜素模型训练完成！在 {len(history2.history['loss'])} 个epoch后停止")

y_pred_scaled1 = best_model1.predict(X_test_scaled)
y_pred1 = scaler_y1.inverse_transform(y_pred_scaled1.reshape(-1, 1)).flatten()
y_true1 = y_test['Dry weight '].values

y_train_pred_scaled1 = best_model1.predict(X_train_scaled)
y_train_pred1 = scaler_y1.inverse_transform(y_train_pred_scaled1.reshape(-1, 1)).flatten()
r2_train1 = r2_score(y_train_final['Dry weight '].values, y_train_pred1)
rmse_train1 = np.sqrt(mean_squared_error(y_train_final['Dry weight '].values, y_train_pred1))

y_val_pred_scaled1 = best_model1.predict(X_val_scaled)
y_val_pred1 = scaler_y1.inverse_transform(y_val_pred_scaled1.reshape(-1, 1)).flatten()
r2_val1 = r2_score(y_val_final['Dry weight '].values, y_val_pred1)
rmse_val1 = np.sqrt(mean_squared_error(y_val_final['Dry weight '].values, y_val_pred1))

r2_test1 = r2_score(y_true1, y_pred1)
rmse_test1 = np.sqrt(mean_squared_error(y_true1, y_pred1))

print(f"Model1 Training R2: {r2_train1:.4f}, RMSE: {rmse_train1:.4f}")
print(f"Model1 Validation R2: {r2_val1:.4f}, RMSE: {rmse_val1:.4f}")
print(f"Model1 Test R2: {r2_test1:.4f}, RMSE: {rmse_test1:.4f}")

y_pred_scaled2 = best_model2.predict(X_test_scaled)
y_pred2 = scaler_y2.inverse_transform(y_pred_scaled2.reshape(-1, 1)).flatten()
y_true2 = y_test['Carotene yield'].values

y_train_pred_scaled2 = best_model2.predict(X_train_scaled)
y_train_pred2 = scaler_y2.inverse_transform(y_train_pred_scaled2.reshape(-1, 1)).flatten()
r2_train2 = r2_score(y_train_final['Carotene yield'].values, y_train_pred2)
rmse_train2 = np.sqrt(mean_squared_error(y_train_final['Carotene yield'].values, y_train_pred2))

y_val_pred_scaled2 = best_model2.predict(X_val_scaled)
y_val_pred2 = scaler_y2.inverse_transform(y_val_pred_scaled2.reshape(-1, 1)).flatten()
r2_val2 = r2_score(y_val_final['Carotene yield'].values, y_val_pred2)
rmse_val2 = np.sqrt(mean_squared_error(y_val_final['Carotene yield'].values, y_val_pred2))

r2_test2 = r2_score(y_true2, y_pred2)
rmse_test2 = np.sqrt(mean_squared_error(y_true2, y_pred2))

print(f"Model2 Training R2: {r2_train2:.4f}, RMSE: {rmse_train2:.4f}")
print(f"Model2 Validation R2: {r2_val2:.4f}, RMSE: {rmse_val2:.4f}")
print(f"Model2 Test R2: {r2_test2:.4f}, RMSE: {rmse_test2:.4f}")

print("训练随机森林模型...")
random_forest_model1 = RandomForestRegressor(
    n_estimators=64,
    random_state=42,
    max_depth=16
)
random_forest_model1.fit(X_train_scaled, y_train_scaled_dry.ravel())

random_forest_model2 = RandomForestRegressor(
    n_estimators=79,
    random_state=42,
    max_depth=16
)
random_forest_model2.fit(X_train_scaled, y_train_scaled_car.ravel())
print("随机森林模型训练完成！")

print("训练XGBoost模型...")
xgb_model1 = xgb.XGBRegressor(n_estimators=310, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model1.fit(X_train_scaled, y_train_scaled_dry.ravel())
xgb_pred_scaled1 = xgb_model1.predict(X_test_scaled)
xgb_pred1 = scaler_y1.inverse_transform(xgb_pred_scaled1.reshape(-1, 1)).flatten()
xgb_rmse1 = np.sqrt(mean_squared_error(y_test['Dry weight '], xgb_pred1))
xgb_mae1 = mean_absolute_error(y_test['Dry weight '], xgb_pred1)
xgb_r2_1 = r2_score(y_test['Dry weight '], xgb_pred1)

xgb_model2 = xgb.XGBRegressor(n_estimators=243, max_depth=3, learning_rate=0.2, random_state=42)
xgb_model2.fit(X_train_scaled, y_train_scaled_car.ravel())
xgb_pred_scaled2 = xgb_model2.predict(X_test_scaled)
xgb_pred2 = scaler_y2.inverse_transform(xgb_pred_scaled2.reshape(-1, 1)).flatten()
xgb_rmse2 = np.sqrt(mean_squared_error(y_test['Carotene yield'], xgb_pred2))
xgb_mae2 = mean_absolute_error(y_test['Carotene yield'], xgb_pred2)
xgb_r2_2 = r2_score(y_test['Carotene yield'], xgb_pred2)

print(f"XGBoost Model1 RMSE: {xgb_rmse1:.4f}")
print(f"XGBoost Model1 MAE: {xgb_mae1:.4f}")
print(f"XGBoost Model1 R2: {xgb_r2_1:.4f}")
print(f"XGBoost Model2 RMSE: {xgb_rmse2:.4f}")
print(f"XGBoost Model2 MAE: {xgb_mae2:.4f}")
print(f"XGBoost Model2 R2: {xgb_r2_2:.4f}")

print("训练GBDT模型...")
gbdt_model1 = GradientBoostingRegressor(n_estimators=322, max_depth=6, random_state=42)
gbdt_model1.fit(X_train_scaled, y_train_scaled_dry.ravel())
gbdt_pred_scaled1 = gbdt_model1.predict(X_test_scaled)
gbdt_pred1 = scaler_y1.inverse_transform(gbdt_pred_scaled1.reshape(-1, 1)).flatten()
gbdt_rmse1 = np.sqrt(mean_squared_error(y_test['Dry weight '], gbdt_pred1))
gbdt_mae1 = mean_absolute_error(y_test['Dry weight '], gbdt_pred1)
gbdt_r2_1 = r2_score(y_test['Dry weight '], gbdt_pred1)

gbdt_model2 = GradientBoostingRegressor(n_estimators=349, max_depth=3, random_state=42)
gbdt_model2.fit(X_train_scaled, y_train_scaled_car.ravel())
gbdt_pred_scaled2 = gbdt_model2.predict(X_test_scaled)
gbdt_pred2 = scaler_y2.inverse_transform(gbdt_pred_scaled2.reshape(-1, 1)).flatten()
gbdt_rmse2 = np.sqrt(mean_squared_error(y_test['Carotene yield'], gbdt_pred2))
gbdt_mae2 = mean_absolute_error(y_test['Carotene yield'], gbdt_pred2)
gbdt_r2_2 = r2_score(y_test['Carotene yield'], gbdt_pred2)

print(f"GBDT Model1 RMSE: {gbdt_rmse1:.4f}")
print(f"GBDT Model1 MAE: {gbdt_mae1:.4f}")
print(f"GBDT Model1 R2: {gbdt_r2_1:.4f}")
print(f"GBDT Model2 RMSE: {gbdt_rmse2:.4f}")
print(f"GBDT Model2 MAE: {gbdt_mae2:.4f}")
print(f"GBDT Model2 R2: {gbdt_r2_2:.4f}")

def evaluate_model(model, X_train, y_train_scaled, X_val, y_val_scaled, y_val_original, X_test, y_test_original, scaler_y, feature_name):
    try:
        cv_scores = cross_val_score(model, X_train, y_train_scaled.ravel(), cv=5, scoring='r2')
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)
    except Exception as e:
        print(f"交叉验证时出错: {e}")
        cv_r2_mean = None
        cv_r2_std = None
    
    try:
        y_val_pred_scaled = model.predict(X_val)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
        val_r2 = r2_score(y_val_original, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_original, y_val_pred))
    except Exception as e:
        print(f"验证集预测时出错: {e}")
        val_r2 = None
        val_rmse = None
    
    try:
        y_test_pred_scaled = model.predict(X_test)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
        mae = mean_absolute_error(y_test_original, y_test_pred)
        r2 = r2_score(y_test_original, y_test_pred)
    except Exception as e:
        print(f"测试集预测时出错: {e}")
        rmse, mae, r2 = None, None, None
    
    if cv_r2_mean is not None:
        print(f"{model.__class__.__name__} {feature_name} Validation CV R2: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    if val_r2 is not None:
        print(f"{model.__class__.__name__} {feature_name} Validation R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
    if rmse is not None:
        print(f"{model.__class__.__name__} {feature_name} Test RMSE: {rmse:.4f}")
        print(f"{model.__class__.__name__} {feature_name} Test MAE: {mae:.4f}")
        print(f"{model.__class__.__name__} {feature_name} Test R2: {r2:.4f}\n")
    
    return {
        'cv_r2': cv_r2_mean,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2
    }

print("==== Dry weight ====")
rf_dw_metrics = evaluate_model(random_forest_model1, X_train_scaled, y_train_scaled_dry, 
                              X_val_scaled, y_val_scaled_dry, y_val_final['Dry weight '].values, 
                              X_test_scaled, y_test['Dry weight '].values, scaler_y1, "Dry weight")

xgb_dw_metrics = evaluate_model(xgb_model1, X_train_scaled, y_train_scaled_dry, 
                               X_val_scaled, y_val_scaled_dry, y_val_final['Dry weight '].values,
                               X_test_scaled, y_test['Dry weight '].values, scaler_y1, "Dry weight")

gbdt_dw_metrics = evaluate_model(gbdt_model1, X_train_scaled, y_train_scaled_dry, 
                                X_val_scaled, y_val_scaled_dry, y_val_final['Dry weight '].values,
                                X_test_scaled, y_test['Dry weight '].values, scaler_y1, "Dry weight")

print("==== Carotene yield ====")
rf_cy_metrics = evaluate_model(random_forest_model2, X_train_scaled, y_train_scaled_car, 
                              X_val_scaled, y_val_scaled_car, y_val_final['Carotene yield'].values,
                              X_test_scaled, y_test['Carotene yield'].values, scaler_y2, "Carotene yield")

xgb_cy_metrics = evaluate_model(xgb_model2, X_train_scaled, y_train_scaled_car, 
                               X_val_scaled, y_val_scaled_car, y_val_final['Carotene yield'].values,
                               X_test_scaled, y_test['Carotene yield'].values, scaler_y2, "Carotene yield")

gbdt_cy_metrics = evaluate_model(gbdt_model2, X_train_scaled, y_train_scaled_car, 
                                X_val_scaled, y_val_scaled_car, y_val_final['Carotene yield'].values,
                                X_test_scaled, y_test['Carotene yield'].values, scaler_y2, "Carotene yield")

dl_rmse1 = np.sqrt(mean_squared_error(y_true1, y_pred1))
dl_mae1 = mean_absolute_error(y_true1, y_pred1)
dl_rmse2 = np.sqrt(mean_squared_error(y_true2, y_pred2))
dl_mae2 = mean_absolute_error(y_true2, y_pred2)

rf_test_scaled1 = random_forest_model1.predict(X_test_scaled)
rf_test1 = scaler_y1.inverse_transform(rf_test_scaled1.reshape(-1, 1)).flatten()
rf_rmse1 = np.sqrt(mean_squared_error(y_true1, rf_test1))
rf_mae1 = mean_absolute_error(y_true1, rf_test1)
rf_r2_1 = r2_score(y_true1, rf_test1)

rf_test_scaled2 = random_forest_model2.predict(X_test_scaled)
rf_test2 = scaler_y2.inverse_transform(rf_test_scaled2.reshape(-1, 1)).flatten()
rf_rmse2 = np.sqrt(mean_squared_error(y_true2, rf_test2))
rf_mae2 = mean_absolute_error(y_true2, rf_test2)
rf_r2_2 = r2_score(y_true2, rf_test2)

def evaluate_and_save(model, model_name, target_name, X_train, y_train_original, 
                      X_val, y_val_original, X_test, y_test_original, scaler_X, scaler_y):
    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    train_pred_scaled = model.predict(X_train_scaled)
    val_pred_scaled = model.predict(X_val_scaled)
    test_pred_scaled = model.predict(X_test_scaled)
    
    train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
    val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    
    y_train_original = y_train_original.values.flatten()
    y_val_original = y_val_original.values.flatten()
    y_test_original = y_test_original.values.flatten()
    
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2
    
    train_rmse, train_mae, train_r2 = calculate_metrics(y_train_original, train_pred)
    val_rmse, val_mae, val_r2 = calculate_metrics(y_val_original, val_pred)
    test_rmse, test_mae, test_r2 = calculate_metrics(y_test_original, test_pred)
    
    train_df = pd.DataFrame({
        'Actual': y_train_original,
        'Predicted': train_pred
    })
    
    val_df = pd.DataFrame({
        'Actual': y_val_original,
        'Predicted': val_pred
    })
    
    test_df = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': test_pred
    })
    
    desktop_path = r'C:\Users\Desktop\模型输出结果'
    os.makedirs(desktop_path, exist_ok=True)
    
    safe_target = target_name.replace(' ', '_')
    
    train_df.to_csv(os.path.join(desktop_path, f'{model_name}_{safe_target}_train_predictions.csv'), index=False)
    val_df.to_csv(os.path.join(desktop_path, f'{model_name}_{safe_target}_val_predictions.csv'), index=False)
    test_df.to_csv(os.path.join(desktop_path, f'{model_name}_{safe_target}_test_predictions.csv'), index=False)
    
    return {
        'train': {'RMSE': train_rmse, 'MAE': train_mae, 'R2': train_r2},
        'val': {'RMSE': val_rmse, 'MAE': val_mae, 'R2': val_r2},
        'test': {'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2}
    }

print("\n正在评估所有模型并保存预测结果...")

dl_dw_metrics = evaluate_and_save(
    best_model1, 'DL', 'Dry_weight', 
    X_train_final, y_train_final['Dry weight '],
    X_val_final, y_val_final['Dry weight '],
    X_test, y_test['Dry weight '],
    scaler_X, scaler_y1
)

dl_cy_metrics = evaluate_and_save(
    best_model2, 'DL', 'Carotene_yield', 
    X_train_final, y_train_final['Carotene yield'],
    X_val_final, y_val_final['Carotene yield'],
    X_test, y_test['Carotene yield'],
    scaler_X, scaler_y2
)

rf_dw_metrics = evaluate_and_save(
    random_forest_model1, 'RF', 'Dry_weight', 
    X_train_final, y_train_final['Dry weight '],
    X_val_final, y_val_final['Dry weight '],
    X_test, y_test['Dry weight '],
    scaler_X, scaler_y1
)

rf_cy_metrics = evaluate_and_save(
    random_forest_model2, 'RF', 'Carotene_yield', 
    X_train_final, y_train_final['Carotene yield'],
    X_val_final, y_val_final['Carotene yield'],
    X_test, y_test['Carotene yield'],
    scaler_X, scaler_y2
)

xgb_dw_metrics = evaluate_and_save(
    xgb_model1, 'XGB', 'Dry_weight', 
    X_train_final, y_train_final['Dry weight '],
    X_val_final, y_val_final['Dry weight '],
    X_test, y_test['Dry weight '],
    scaler_X, scaler_y1
)

xgb_cy_metrics = evaluate_and_save(
    xgb_model2, 'XGB', 'Carotene_yield', 
    X_train_final, y_train_final['Carotene yield'],
    X_val_final, y_val_final['Carotene yield'],
    X_test, y_test['Carotene yield'],
    scaler_X, scaler_y2
)

gbdt_dw_metrics = evaluate_and_save(
    gbdt_model1, 'GBDT', 'Dry_weight', 
    X_train_final, y_train_final['Dry weight '],
    X_val_final, y_val_final['Dry weight '],
    X_test, y_test['Dry weight '],
    scaler_X, scaler_y1
)

gbdt_cy_metrics = evaluate_and_save(
    gbdt_model2, 'GBDT', 'Carotene_yield', 
    X_train_final, y_train_final['Carotene yield'],
    X_val_final, y_val_final['Carotene yield'],
    X_test, y_test['Carotene yield'],
    scaler_X, scaler_y2
)

print(f"所有预测结果已保存到桌面文件夹: {r'C:\Users\Desktop'}")#Output the results to the desktop

def print_metrics(metrics, model_name, target_name):
    print(f"\n===== {model_name} - {target_name} =====")
    print("训练集:")
    print(f"  RMSE: {metrics['train']['RMSE']:.4f}, MAE: {metrics['train']['MAE']:.4f}, R²: {metrics['train']['R2']:.4f}")
    
    print("验证集:")
    print(f"  RMSE: {metrics['val']['RMSE']:.4f}, MAE: {metrics['val']['MAE']:.4f}, R²: {metrics['val']['R2']:.4f}")
    
    print("测试集:")
    print(f"  RMSE: {metrics['test']['RMSE']:.4f}, MAE: {metrics['test']['MAE']:.4f}, R²: {metrics['test']['R2']:.4f}")

print("\n\n===== 模型评估指标汇总 =====")
print_metrics(dl_dw_metrics, "深度模型", "干重")
print_metrics(dl_cy_metrics, "深度模型", "胡萝卜素")
print_metrics(rf_dw_metrics, "随机森林", "干重")
print_metrics(rf_cy_metrics, "随机森林", "胡萝卜素")
print_metrics(xgb_dw_metrics, "XGBoost", "干重")
print_metrics(xgb_cy_metrics, "XGBoost", "胡萝卜素")
print_metrics(gbdt_dw_metrics, "GBDT", "干重")
print_metrics(gbdt_cy_metrics, "GBDT", "胡萝卜素")

print("\n所有模型评估完成！")
print("\n" + "="*60)
print("早停机制执行结果")
print("="*60)
print(f"干重模型在 {len(history1.history['loss'])} 个epoch后停止训练")
print(f"胡萝卜素模型在 {len(history2.history['loss'])} 个epoch后停止训练")
