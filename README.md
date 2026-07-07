# 💳 Credit Card Fraud Detection

基于机器学习的信用卡欺诈检测系统，集成模型训练、SHAP 可解释性分析与交互式 Web 演示。

---

## 📋 项目结构

```
.
├── AGENTS.md                # 项目规范与目录约定
├── README.md                # 项目说明文档
├── data/
│   ├── raw/                 # 原始数据集（只读）
│   └── processed/           # 清洗后的数据
├── src/
│   ├── download_data.py     # 从 Kaggle 下载数据集
│   ├── data_loader.py       # 数据加载与 EDA 分析
│   ├── preprocess.py        # 数据预处理（标准化 + SMOTE 过采样）
│   ├── model.py             # 模型训练（LR / RF / SVM + 5 折交叉验证）
│   ├── evaluate.py          # 模型评估（ROC、PR、混淆矩阵）
│   ├── shap_analysis.py     # SHAP 可解释性分析
│   └── save_scaler.py       # 保存 StandardScaler 供推理使用
├── notebooks/               # Jupyter EDA 笔记
├── reports/                 # 输出图表与结果
│   └── shap/                # SHAP 分析图表
├── models/                  # 训练好的模型与 scaler
└── app/
    └──app.py                # Streamlit 交互式演示应用
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
# source .venv/bin/activate   # Linux / macOS
 .venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
python src/download_data.py
```

> 数据集来源：[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 3. 完整训练流程

```bash
# Step 1: 数据探索（可选）
python src/data_loader.py

# Step 2: 训练模型（自动执行 5 折交叉验证）
python src/model.py

# Step 3: 保存 scaler（推理前必须执行）
python src/save_scaler.py

# Step 4: SHAP 可解释性分析
python src/shap_analysis.py
```

### 4. 启动 Web 应用

```bash
streamlit run app.py
```

---

## 🔬 方法论

### 数据处理

| 步骤 | 说明 |
|------|------|
| **特征标准化** | 对 `Amount` 和 `Time` 列进行 `StandardScaler` 标准化 |
| **分层抽样** | 按 8:2 比例分层划分训练集 / 测试集，保持类别分布 |
| **SMOTE 过采样** | 对训练集少数类（欺诈）进行合成过采样，缓解类别不平衡 |

### 模型方案

采用三种基线模型，通过 **5 折分层交叉验证** 评估：

| 模型 | 特点 |
|------|------|
| **Logistic Regression** | 线性基线，训练快，可解释性强 |
| **Random Forest** | 集成树模型，自动特征重要性排序，最终部署模型 |
| **SVM (Linear)** | 高维空间线性分类，适合大规模稀疏数据 |

### 评估指标

针对欺诈检测的**类别不平衡**特点，重点关注：

- **Precision**（精确率）：减少误报，避免正常交易被错误拦截
- **Recall**（召回率）：减少漏报，尽可能捕获真实欺诈
- **F1-Score**：Precision 与 Recall 的调和均值
- **ROC-AUC**：整体排序能力
- **PR-AUC**（Average Precision）：不平衡场景下的综合指标

---

## 📊 SHAP 可解释性

项目集成 [SHAP](https://shap.readthedocs.io/) 进行模型解释：

- **Beeswarm Summary Plot**：全局特征影响力概览
- **Force Plot**：单个样本的预测驱动力分解
- **Top-3 Feature Bar Plot**：误分类样本的关键特征可视化

SHAP 分析帮助回答：**为什么模型做出了这个预测？**

---

## 🖥️ Web 演示

Streamlit 应用提供交互式欺诈检测演示：

1. 在侧边栏输入交易特征（`Amount`、`Time`、PCA 特征 `V1`–`V28`）
2. 实时查看模型预测结果（正常 / 欺诈）及欺诈概率
3. 可视化 Top-5 SHAP 特征贡献，理解模型决策依据
4. 项目已在Streamlit社区部署，可直接访问网址 https://zjut-credit-card-fraud-detection-rgujrsyb5rnevoydwbjq2u.streamlit.app/

---

## 📝 注意事项

- 原始数据 (`data/raw/`) 为**只读**，预处理后的数据写入 `data/processed/`
- 所有路径使用相对路径，保证跨平台兼容
- 网络较慢时，可手动下载数据集后放入 `data/raw/creditcard.csv`
- 运行 `app.py` 前必须执行 `python src/save_scaler.py` 生成 scaler

---
