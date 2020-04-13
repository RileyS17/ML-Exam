from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("set2.csv")
X = data.iloc[:, :].values
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()