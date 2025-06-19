from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import data_prep as prep

def plot_explained_variance_ratio(pca):
    plt.figure()
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Explained Variance by different principal components")
    plt.show()

def plot_eigenvalues(pca):
    plt.xlabel("Number of components")
    plt.ylabel("Eigenvalues")
    plt.title("PCA Eigenvalues")
    plt.ylim(0, max(pca.explained_variance_))
    plt.axhline(y=1, color="r", linestyle="--")
    plt.plot(pca.explained_variance_)
    plt.show()

def find_n():
    data = prep.NORMALIZED_DATA
    train_data, _ = prep.split_data(data)
    pca = PCA().fit(train_data)
    plot_explained_variance_ratio(pca)
    plot_eigenvalues(pca)

MY_PCA = PCA(n_components=3)
DATA_AFTER_PCA = MY_PCA.fit_transform(prep.NORMALIZED_DATA)
TRAIN_DATA, TEST_DATA = prep.split_data(DATA_AFTER_PCA)