from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# n equals the desired components
def apply_pca(n, data):
    ss = StandardScaler()
    data = ss.fit_transform(data)
    pca = PCA(n_components=n)
    return pca.fit_transform(data)