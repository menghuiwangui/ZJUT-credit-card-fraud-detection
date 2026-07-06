# Credit Card Fraud Detection Project

## Environment
- Python virtual env: .venv (activate before running scripts)
- Required packages: kagglehub, pandas, numpy, scikit-learn, matplotlib, seaborn

## Directory Convention
- data/raw/        # 原始数据集 (creditcard.csv)
- data/processed/  # 清洗后数据
- src/             # Python 脚本 (download, train, etc.)
- notebooks/       # Jupyter EDA
- reports/         # 图片、输出结果

## Rules
- 使用相对路径，兼容跨平台
- 不要硬编码绝对路径
- 原始数据只读，不修改 data/raw/
- 若网络慢可提示手动下载链接