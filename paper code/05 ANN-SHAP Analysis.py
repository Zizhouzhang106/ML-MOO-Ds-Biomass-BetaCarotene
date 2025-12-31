#ANN-SHAP Analysis
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import math

import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'C:\Users\data.csv', encoding='ISO-8859-1')#data import

target1 = 'Dry weight '
target2 = 'Carotene yield'
features = ['light_intensity', 'temperature', 'hormone','bicarbonate', 
        'nitrogen_source', 'phosphorus_source', 'cultivation_time', 'nacl']

X = df[features]
y1 = df[target1]
y2 = df[target2]
y_multi = df[[target1, target2]]


X_train, X_test, y1_train, y1_test, y2_train, y2_test, y_multi_train, y_multi_test = train_test_split(
    X, y1, y2, y_multi, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test) 

scaler_y1 = StandardScaler()
y1_train_scaled = scaler_y1.fit_transform(y1_train.values.reshape(-1, 1)).ravel()
y1_test_scaled = scaler_y1.transform(y1_test.values.reshape(-1, 1)).ravel()

scaler_y2 = StandardScaler()
y2_train_scaled = scaler_y2.fit_transform(y2_train.values.reshape(-1, 1)).ravel()
y2_test_scaled = scaler_y2.transform(y2_test.values.reshape(-1, 1)).ravel()

scaler_y_multi = StandardScaler()
y_multi_train_scaled = scaler_y_multi.fit_transform(y_multi_train)
y_multi_test_scaled = scaler_y_multi.transform(y_multi_test)

grid1 = GridSearchCV(MLPRegressor(max_iter=5000, random_state=42), {
    'hidden_layer_sizes': [(57,)],
    'activation': ['relu'],
    'alpha': [0.0005],
    'learning_rate': ['constant', 'adaptive']
}, cv=5)
grid1.fit(X_train_scaled, y1_train_scaled)  
best_model1 = grid1.best_estimator_

grid2 = GridSearchCV(MLPRegressor(max_iter=5000, random_state=42), {
    'hidden_layer_sizes': [(40,)],
    'activation': ['relu'],
    'alpha': [0.005],
    'learning_rate': ['constant', 'adaptive']
}, cv=5)
grid2.fit(X_train_scaled, y2_train_scaled) 
best_model2 = grid2.best_estimator_

def build_multi_output_model(input_dim):
    model = Sequential([
        Dense(49, activation='relu', input_shape=(input_dim,)),
        Dense(2)  
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

keras_model = build_multi_output_model(X_train_scaled.shape[1])
history = keras_model.fit(
    X_train_scaled, 
    y_multi_train_scaled, 
    epochs=250, 
    batch_size=32, 
    verbose=0,
    validation_data=(X_test_scaled, y_multi_test_scaled)  
)

def evaluate_model(model, X_test_scaled, y_test_scaled, scaler_y, model_name):
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} RMSE: {rmse:.4f}")
    return rmse

print("\n模型性能评估:")
rmse1 = evaluate_model(best_model1, X_test_scaled, y1_test_scaled, scaler_y1, "干重模型")
rmse2 = evaluate_model(best_model2, X_test_scaled, y2_test_scaled, scaler_y2, "胡萝卜素模型")

y_multi_pred_scaled = keras_model.predict(X_test_scaled)
y_multi_pred = scaler_y_multi.inverse_transform(y_multi_pred_scaled)
y_multi_true = scaler_y_multi.inverse_transform(y_multi_test_scaled)

rmse_multi_dry = math.sqrt(mean_squared_error(y_multi_true[:, 0], y_multi_pred[:, 0]))
rmse_multi_car = math.sqrt(mean_squared_error(y_multi_true[:, 1], y_multi_pred[:, 1]))
print(f"多目标模型-干重 RMSE: {rmse_multi_dry:.4f}")
print(f"多目标模型-胡萝卜素 RMSE: {rmse_multi_car:.4f}")

explainer1 = shap.Explainer(best_model1.predict, X_train_scaled) 
shap_values1 = explainer1(X_test_scaled)  

explainer2 = shap.Explainer(best_model2.predict, X_train_scaled)
shap_values2 = explainer2(X_test_scaled)

k_explainer = shap.Explainer(keras_model, X_train_scaled)
k_shap_values = k_explainer(X_test_scaled)

def save_shap_summary(shap_values, X_data, name, rmse_value=None):
    shap.summary_plot(shap_values, features=X_data, feature_names=features, show=False)
    if rmse_value:
        plt.title(f"SHAP - {name} (测试集 RMSE: {rmse_value:.4f})")
    else:
        plt.title(f"SHAP - {name} (测试集)")
    plt.tight_layout()
    plt.savefig(f"shap_{name}.png")
    plt.clf()

    shap_df = pd.DataFrame(shap_values.values, columns=features)
    shap_df.to_csv(f"shap_{name}.csv", index=False)

save_shap_summary(shap_values1, X_test, "干重", rmse1)
save_shap_summary(shap_values2, X_test, "胡萝卜素产量", rmse2)

for i, name in enumerate(['干重', '胡萝卜素产量']):
    sv = k_shap_values[:, :, i]
    shap.summary_plot(sv, features=X_test, feature_names=features, show=False)
    rmse_value = rmse_multi_dry if i == 0 else rmse_multi_car
    plt.title(f"SHAP - 多目标模型: {name} (测试集 RMSE: {rmse_value:.4f})")
    plt.tight_layout()
    plt.savefig(f"shap_multi_{name}.png")
    plt.clf()

    df_shap = pd.DataFrame(sv.values, columns=features)
    df_shap.to_csv(f"shap_multi_{name}.csv", index=False)

joblib.dump(best_model1, 'model_dry_weight.pkl')
joblib.dump(best_model2, 'model_carotenoid.pkl')
keras_model.save('model_multi_output_keras.h5')

print("\n模型与图像、CSV 已保存。")
print(f"测试集大小: {X_test_scaled.shape[0]} 个样本")
