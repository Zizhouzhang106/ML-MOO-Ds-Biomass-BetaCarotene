# ML-MOO-Ds-Biomass-BetaCarotene
Code for Machine learning-driven multi-objective optimization of Dunaliella salina cultivation for enhanced biomass and β-carotene production

一、项目概述
本项目针对盐生杜氏藻（Dunaliella salina）生物量（干重-DCW）与β-胡萝卜素产量的协同优化问题，构建了“神经网络模型结构及树模型关键参数超参数优化→单/多目标预测→→SHAP可解释性分析”的完整机器学习框架。通过系统优化随机森林（RF）、极端梯度提升（XGBoost）、梯度提升决策树（GBDT）和人工神经网络（ANN）四大算法，实现了对盐藻培养关键指标的高精度预测，结合多目标粒子群优化（MOPSO）及帕累托前沿解生成实用培养策略，并通过SHAP分析揭示了特征影响，最终可提升生物量和β-胡萝卜素产量，为微藻工业化生产提供解决方案。

二、核心流程与关键模块
（一）超参数优化
1. 优化目标
针对单目标/多目标模型中的树模型关键参数和神经网络模型拓扑结构，以最小化预测误差（RMSE）为目标优化，提升模型泛化能力，确定最适结构与超参数。
2. 优化工具与算法
树模型（RF/XGBoost/GBDT）：采用贝叶斯优化（Hyperopt）
人工神经网络（ANN）：采用改进粒子群优化（Advanced PSO）
交叉验证策略：5折交叉验证，确保无数据泄露
3. 优化参数与最佳配置
模型类型	优化参数	最佳配置示例（单目标）	最佳配置示例（多目标）
RF：n_estimators、max_depth	单目标（DCW：64棵树、深度16；β - 胡萝卜素：79棵树、深度16）；多目标（236棵树、深度7）
XGBoost：n_estimators、max_depth 单目标（DCW：310棵树、深度6；β - 胡萝卜素 243棵树、深度3）；多目标（341棵树、深度5）
GBDT：n_estimators、max_depth	单目标（DCW：322棵树、深度6；β - 胡萝卜素：349棵树、深度3）；多目标（242棵树、深度4）
ANN：拓扑结构（隐藏层层数、神经元数） 单目标（DCW：（1，57）；β - 胡萝卜素：（1.40））；多目标（1，49）
4. 输出结果
各模型优化后的超参数配置文件（CSV格式）
优化过程记录（迭代损失、最优试验ID）
交叉验证性能报告（CV - RMSE）

（二）单目标预测模型
1. 数据基础
数据集规模：637个培养条件×4个采样时间点共计1494条有效观测（数据处于合同保护状态，暂不公布）
输入特征（8个）：光照强度、温度、盐度（NaCl）、碳酸氢钠、硝酸钠、磷酸氢二钾、激素浓度、培养时间
目标变量（2个）：干重（DCW，g/L）、β - 胡萝卜素产量（mg/L）
数据划分：训练集80%（含25%验证集）、测试集20%，随机种子42保证可重复性
预处理：缺失值填充、MinMax归一化
2. 模型构建与性能
目标变量	最优模型	测试集性能（R²/RMSE/MAE）核心优势
干重（DCW）ANN	捕捉非线性生长关系，残差无系统偏差
β - 胡萝卜素产量	GBDT适应胁迫响应驱动的代谢规律
3. 输出结果
单目标最优模型文件
训练/验证/测试集预测结果
模型性能对比信息

（三）多目标优化模型
1. 模型构建
基于单目标最优模型扩展，构建双输出统一框架：
多目标ANN：单隐藏层49神经元，同时预测DCW和β - 胡萝卜素产量
多目标GBDT：MultiOutputRegressor包装，独立优化双目标预测逻辑
2. 优化算法
帕累托最优解：MOPSO算法（粒子数100、迭代100次、惯性权重0.9→0.4）
加权求和解：权重α∈[0,1]（0优先β - 胡萝卜素，1优先DCW），共11组目标组合
3. 性能与验证
多目标ANN整体测试R²
实验验证：12组优化方案（1个帕累托解+11个加权解）
性能提升：较非ML优化对照组，生物量提升百分比，β - 胡萝卜素提升百分比
4. 输出结果
多目标预测模型文件
帕累托前沿解与加权解数据集（含优化培养参数）
优化结果可视化图表

（四）SHAP可解释性分析
1. 分析目标
量化特征对预测结果的贡献度
揭示特征与目标变量的非线性关系
验证模型决策逻辑与生理机制的一致性
2. 核心发现
关键特征影响：
培养时间：生物量和β - 胡萝卜素积累的首要驱动因素，随时间先增后稳
盐度：高盐胁迫显著促进β - 胡萝卜素合成
激素：低浓度促进双目标，高浓度抑制
3. 输出结果
SHAP值数据集（CSV格式，含特征原始值与贡献度）
可视化图表（特征重要性条形图、依赖关系散点图、多目标对比热图）
特征影响机制分析报告

三、环境配置与使用说明
1.依赖库安装
# 基础数据处理与科学计算
import numpy as np
import pandas as pd
import math
import gc
import warnings
import logging
import os
# 机器学习与预处理
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from scikeras.wrappers import KerasRegressor
# 深度学习框架
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, InputLayer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import backend as K
# 树模型与优化
import xgboost as xgb
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
# 可解释性分析
import shap
# 可视化
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib
# 模型保存与工具
import joblib

2.版本要求建议
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
xgboost>=1.5.0
hyperopt>=0.2.7
shap>=0.41.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikeras>=0.10.0

（二）数据准备
1.数据文件：将培养数据整理为data.csv，编码为ISO - 8859 - 1
2.列名规范：输入特征需与“光照强度（light_intensity）、温度（temperature）”等一致，目标变量为“Dry weight ”“Carotene yield”
3.数据预处理：代码内置缺失值填充与归一化流程，无需手动处理

（三）运行步骤
1.超参数优化-生成最优参数配置
2.单目标模型训练-输出单目标最优模型
3.多目标模型训练与优化-生成帕累托解与加权解
4.SHAP分析-输出可解释性结果
5.结果验证：基于优化参数进行实验室培养，对比预测值与实测值

（四）输出文件目录
├─ 超参数优化结果/ # 超参数配置、迭代记录
├─ 模型文件/ # .pkl（树模型）、.h5（ANN模型）
├─ 预测结果/ # 训练/测试集预测CSV、性能指标汇总
├─ SHAP分析结果/ # SHAP值CSV、可视化图表、分析报告
└─ 优化方案/ # 帕累托解、加权解、培养参数表

四、注意事项
1.数据路径：代码中数据读取路径为C:\Users\data.csv，使用时需修改为实际路径
2.可重复性：所有模型均固定随机种子（42），确保结果可复现
3.硬件要求：ANN训练建议使用GPU加速，超参数优化过程可根据算力调整迭代次数
4.实验验证：优化后的培养参数需控制在实验可行范围
5.结果解读：SHAP分析结果需结合微藻生理机制（如盐胁迫、营养代谢）进行解读

五、项目创新点
1.首次将植物激素作为量化输入特征，揭示其浓度依赖的调控作用
2.把培养时间视为动态优化变量，突破固定采样点的局限
3.构建“预测 - 优化 - 解释”闭环框架，模型精度与可解释性兼顾
4.多目标优化支持灵活目标配置，适配不同生产需求（生物量优先/β - 胡萝卜素优先/均衡）
