import pandas as pd  # conventional alias
from sklearn.datasets import load_boston

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# df= pd.read_csv("data/data.csv")

# 查看数据集大小
instance_count, attr_count = df.shape


