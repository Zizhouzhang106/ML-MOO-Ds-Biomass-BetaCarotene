#ANN Network Structure Hyperparameter Optimization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import gc
import warnings
import logging

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class DataProcessor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, encoding='ISO-8859-1')
        self._preprocess()
    
    def _preprocess(self):
        self.df = self.df.fillna(self.df.mean())       
        features = ['light_intensity', 'temperature', 'hormone', 
                   'bicarbonate', 'nitrogen_source', 'phosphorus_source',
                   'cultivation_time', 'nacl']
        self.X = self.df[features].copy()        
        self.y_raw = self.df[['Dry weight ', 'Carotene yield']].copy()        
        y_transformed = self.y_raw.copy()
        y_transformed['Dry weight '] = np.sqrt(y_transformed['Dry weight '])
        y_transformed['Carotene yield'] = np.log1p(y_transformed['Carotene yield'].clip(lower=0))        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_transformed, test_size=0.2, random_state=42
        )
        
        self.X_train_raw = X_train.copy()
        self.X_test_raw = X_test.copy()
        self.y_train_raw = y_train.copy()
        self.y_test_raw = y_test.copy()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.scalers = {
            'X': MinMaxScaler().fit(self.X_train),
            'dry': MinMaxScaler().fit(self.y_train[['Dry weight ']]),
            'car': MinMaxScaler().fit(self.y_train[['Carotene yield']])
        }
        self.X_train_scaled = self.scalers['X'].transform(self.X_train)
        self.X_test_scaled = self.scalers['X'].transform(self.X_test)
        
        self.y_train_scaled = np.hstack([
            self.scalers['dry'].transform(self.y_train[['Dry weight ']]),
            self.scalers['car'].transform(self.y_train[['Carotene yield']])
        ])
        self.y_test_scaled = np.hstack([
            self.scalers['dry'].transform(self.y_test[['Dry weight ']]),
            self.scalers['car'].transform(self.y_test[['Carotene yield']])
        ])
        print(f"训练集: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"测试集: X={self.X_test.shape}, y={self.y_test.shape}")

class AdvancedPSO:
    def __init__(self, model_type, input_dim, num_particles=15, max_iter=10, n_folds=5):
        self.model_type = model_type
        self.input_dim = input_dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.n_folds = n_folds  
        self.activation_functions = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
        self.global_best = {
            'loss': np.inf, 
            'topology': None, 
            'activations': None, 
            'lr': None,
            'cv_rmse': np.inf  
        }
        
        self.topology_history = []
        self.total_neurons_history = []
        self.learning_rate_history = []
        self.activations_history = []
        self.cv_rmse_history = []
        self.iterations_history = []
        self._init_particles()
    
    def _init_particles(self):
        self.particles = []
        for _ in range(self.num_particles):
            depth = np.random.randint(1, 4)  
            topology = np.random.randint(10, 101, size=depth)
            activations = [np.random.choice(self.activation_functions) for _ in range(depth)]
            lr = 10 ** np.random.uniform(-4, -1)
            self.particles.append({
                'depth': depth,
                'topology': topology,
                'activations': activations,
                'lr': lr,
                'velocity_topology': np.zeros_like(topology, dtype=float),
                'velocity_lr': 0.0,
                'best_cv_rmse': np.inf,
                'best_topology': topology.copy(),
                'best_activations': activations.copy(),
                'best_lr': lr
            })
        print(f"初始化粒子数: {self.num_particles}")

    def _dynamic_adjust(self, particle, iteration):
        w = 0.9 - (0.5 * (iteration / self.max_iter))        
        if np.random.rand() < 0.3:
            current_depth = particle['depth']
            new_depth = np.clip(current_depth + np.random.choice([-1, 1]), 1, 3)
            if new_depth > current_depth:
                for _ in range(new_depth - current_depth):
                    new_neurons = np.random.randint(10, 101)
                    particle['topology'] = np.append(particle['topology'], new_neurons)
                    particle['activations'].append(np.random.choice(self.activation_functions))
                    particle['velocity_topology'] = np.append(particle['velocity_topology'], 0)
            elif new_depth < current_depth:
                particle['topology'] = particle['topology'][:new_depth]
                particle['activations'] = particle['activations'][:new_depth]
                particle['velocity_topology'] = particle['velocity_topology'][:new_depth]
                particle['depth'] = new_depth        
        for i in range(len(particle['topology'])):
            if np.random.rand() < 0.2:
                adjustment = np.random.randint(-5, 6)  
                particle['topology'][i] = np.clip(
                    particle['topology'][i] + adjustment,
                    10, 100
                )
        for i in range(len(particle['activations'])):
            if np.random.rand() < 0.15:
                particle['activations'][i] = np.random.choice(self.activation_functions)
        particle['lr'] = np.clip(
            particle['lr'] * np.random.normal(1.0, 0.05),
            0.0001, 0.1
        )

    def _prepare_target(self, y_train_scaled):
        if self.model_type == 'dry':
            return y_train_scaled[:, 0].reshape(-1, 1)
        elif self.model_type == 'car':
            return y_train_scaled[:, 1].reshape(-1, 1)
        else:
            return y_train_scaled

    def _build_model(self, topology, activations, lr):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        for i, n in enumerate(topology):
            model.add(Dense(
                n, 
                activation=activations[i] if i < len(activations) else 'relu',
                kernel_regularizer=l2(0.01)
            ))
            model.add(Dropout(0.5))
        output_size = 1 if self.model_type != 'multi' else 2
        model.add(Dense(output_size, activation='linear'))
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='mse'
        )
        return model

    def _evaluate_with_cv(self, particle, X_scaled, y_target):
        try:
            np.random.seed(42)
            tf.random.set_seed(42)
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y_target[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_val_fold = y_target[val_idx]
                model = self._build_model(
                    particle['topology'], 
                    particle['activations'], 
                    particle['lr']
                )
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=0
                )
                history = model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=30,
                    batch_size=32,
                    verbose=0,
                    callbacks=[early_stop]
                )
                if len(history.history['val_loss']) == 0:
                    fold_rmse = np.inf
                else:
                    last_val_loss = history.history['val_loss'][-1]
                    fold_rmse = math.sqrt(max(last_val_loss, 0))
                fold_scores.append(fold_rmse)
                del model
                gc.collect()
                tf.keras.backend.clear_session()
            cv_rmse = np.mean(fold_scores)
            return cv_rmse
        except Exception as e:
            print(f"评估出错: {str(e)[:80]}")
            return np.inf

    def optimize(self, X_train_scaled, y_train_scaled):
        print(f"\n开始PSO优化 ({self.model_type} 模型)...")
        y_target = self._prepare_target(y_train_scaled)
        for iteration in range(self.max_iter):
            print(f"迭代 {iteration+1}/{self.max_iter}:")
            for i, particle in enumerate(self.particles):
                self._dynamic_adjust(particle, iteration)
                cv_rmse = self._evaluate_with_cv(particle, X_train_scaled, y_target)
                if cv_rmse < particle['best_cv_rmse']:
                    particle['best_cv_rmse'] = cv_rmse
                    particle['best_topology'] = particle['topology'].copy()
                    particle['best_activations'] = particle['activations'].copy()
                    particle['best_lr'] = particle['lr']
                if cv_rmse < self.global_best['cv_rmse']:
                    self.global_best['loss'] = cv_rmse
                    self.global_best['cv_rmse'] = cv_rmse
                    self.global_best['topology'] = particle['topology'].copy()
                    self.global_best['activations'] = particle['activations'].copy()
                    self.global_best['lr'] = particle['lr']
                self.topology_history.append(particle['topology'].copy())
                self.total_neurons_history.append(sum(particle['topology']))
                self.learning_rate_history.append(particle['lr'])
                self.activations_history.append(particle['activations'].copy())
                self.cv_rmse_history.append(cv_rmse)
                self.iterations_history.append(iteration)
                if (i + 1) % 5 == 0:
                    print(f"  粒子: {i+1}/{self.num_particles}, CV RMSE: {cv_rmse:.4f}")
            print(f"  当前最佳CV RMSE: {self.global_best['cv_rmse']:.4f}")
        print(f"\nPSO优化完成!")
        print(f"拓扑结构: {self.global_best['topology']}")
        print(f"激活函数: {self.global_best['activations']}")
        print(f"学习率: {self.global_best['lr']:.6f}")
        print(f"五折交叉验证RMSE: {self.global_best['cv_rmse']:.4f}")
        return self.global_best

class ModelEvaluator:
    def __init__(self, data_processor):
        self.processor = data_processor
    
    def build_final_model(self, best_config, model_name):
        print(f"\n构建 {model_name} 模型...")
        output_size = 1 if model_name != 'Multi-Target' else 2
        model = Sequential()
        model.add(Input(shape=(self.processor.X_train.shape[1],)))
        for i, n in enumerate(best_config['topology']):
            activation = best_config['activations'][i] if i < len(best_config['activations']) else 'relu'
            model.add(Dense(
                n, 
                activation=activation,
                kernel_regularizer=l2(0.01)
            ))
            model.add(Dropout(0.5))
        model.add(Dense(output_size, activation='linear'))
        model.compile(
            optimizer=Adam(learning_rate=best_config['lr']),
            loss='mse'
        )
        print(f"  拓扑结构: {best_config['topology']}")
        print(f"  激活函数: {best_config['activations']}")
        print(f"  学习率: {best_config['lr']:.6f}")
        return model
    
    def train_and_evaluate(self, model, model_name, epochs=50):
        print(f"训练 {model_name} 模型...")
        if model_name == 'Dry Weight':
            y_target = self.processor.y_train_scaled[:, 0].reshape(-1, 1)
        elif model_name == 'Carotene':
            y_target = self.processor.y_train_scaled[:, 1].reshape(-1, 1)
        else:
            y_target = self.processor.y_train_scaled
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        history = model.fit(
            self.processor.X_train_scaled,
            y_target,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )
        print(f"  训练完成，最终损失: {history.history['loss'][-1]:.4f}")
        print(f"评估 {model_name} 模型...")
        y_pred_scaled = model.predict(self.processor.X_test_scaled, verbose=0)
        if model_name == 'Dry Weight':
            y_pred = self._inverse_transform_dry(y_pred_scaled)
            y_true = self._get_true_dry()
            metrics = self._calculate_metrics(y_true, y_pred)
        elif model_name == 'Carotene':
            y_pred = self._inverse_transform_car(y_pred_scaled)
            y_true = self._get_true_car()
            metrics = self._calculate_metrics(y_true, y_pred)
        else:
            y_pred_dry = self._inverse_transform_dry(y_pred_scaled[:, 0].reshape(-1, 1))
            y_true_dry = self._get_true_dry()
            y_pred_car = self._inverse_transform_car(y_pred_scaled[:, 1].reshape(-1, 1))
            y_true_car = self._get_true_car()
            metrics_dry = self._calculate_metrics(y_true_dry, y_pred_dry)
            metrics_car = self._calculate_metrics(y_true_car, y_pred_car)
            metrics = {
                'Dry': metrics_dry,
                'Car': metrics_car
            }
        return metrics
    
    def _inverse_transform_dry(self, y_scaled):
        y_normalized = self.processor.scalers['dry'].inverse_transform(y_scaled)
        y_original = y_normalized ** 2
        return y_original.flatten()
    
    def _inverse_transform_car(self, y_scaled):
        y_normalized = self.processor.scalers['car'].inverse_transform(y_scaled)
        y_original = np.expm1(y_normalized)
        return y_original.flatten()
    
    def _get_true_dry(self):
        y_true_normalized = self.processor.scalers['dry'].transform(
            self.processor.y_test_raw[['Dry weight ']]
        )
        y_true_original = self.processor.scalers['dry'].inverse_transform(y_true_normalized) ** 2
        return y_true_original.flatten()
    
    def _get_true_car(self):
        y_true_normalized = self.processor.scalers['car'].transform(
            self.processor.y_test_raw[['Carotene yield']]
        )
        y_true_original = np.expm1(self.processor.scalers['car'].inverse_transform(y_true_normalized))
        return y_true_original.flatten()
    
    def _calculate_metrics(self, y_true, y_pred):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

def main():
    print("=" * 70)
    print("ANN超参数优化")
    print("=" * 70)
    
    print("\n1. 数据预处理阶段")
    try:
        processor = DataProcessor(r'C:\Users\data.csv')#data Import
        print("数据预处理完成")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    print("\n2. 配置优化任务")
    optimization_tasks = {
        'Dry Weight': {'type': 'dry', 'input_dim': processor.X_train.shape[1]},
        'Carotene': {'type': 'car', 'input_dim': processor.X_train.shape[1]},
        'Multi-Target': {'type': 'multi', 'input_dim': processor.X_train.shape[1]}
    }
    
    best_results = {}
    
    print("\n3. PSO超参数优化阶段")
    
    for name, config in optimization_tasks.items():
        print(f"\n优化 {name} 模型:")
        optimizer = AdvancedPSO(
            model_type=config['type'],
            input_dim=config['input_dim'],
            num_particles=15,
            max_iter=10,
            n_folds=5
        )
        best = optimizer.optimize(processor.X_train_scaled, processor.y_train_scaled)
        if best['loss'] == np.inf:
            print(f"警告: {name} 模型优化失败，使用默认配置")
            best = {
                'loss': 0.1,
                'topology': [50, 30],
                'activations': ['relu', 'relu'],
                'lr': 0.001,
                'cv_rmse': 0.1
            }
        best_results[name] = best
    
    print("\n4. 最优超参数配置")
    for model, params in best_results.items():
        print(f"\n{model}:")
        print(f"  拓扑结构: {params['topology']}")
        print(f"  激活函数: {params['activations']}")
        print(f"  学习率: {params['lr']:.6f}")
        print(f"  五折交叉验证RMSE: {params['cv_rmse']:.4f}")
    
    print("\n5. 最终模型训练和评估")
    evaluator = ModelEvaluator(processor)
    all_metrics = {}
    for name, best_config in best_results.items():
        print(f"\n处理 {name} 模型:")
        model = evaluator.build_final_model(best_config, name)
        if name == 'Multi-Target':
            metrics = evaluator.train_and_evaluate(model, name)
            all_metrics[f'{name}-Dry'] = metrics['Dry']
            all_metrics[f'{name}-Car'] = metrics['Car']
        else:
            metrics = evaluator.train_and_evaluate(model, name)
            all_metrics[name] = metrics
    
    print("\n6. 最终评估结果")
    print("\n性能指标汇总:")
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  R²:   {metrics['R2']:.6f}")
    
    print("\n" + "=" * 70)
    print("程序执行完毕!")
    print("=" * 70)

if __name__ == "__main__":
    main()
