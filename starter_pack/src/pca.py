import numpy as np
import matplotlib.pyplot as plt




def plot_scree(X_d_train):
    X_centered = X_d_train - np.mean(X_d_train,axis=0,keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = (S**2) / (X_centered.shape[0] - 1)
    total_var = np.sum(explained_variance)
    var_ratio = explained_variance / total_var

    plt.title("Scree Plot")
    plt.plot(np.cumsum(var_ratio))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()


def pca_dimensions(X_train):
    X_mean = np.mean(X_train,axis=0,keepdims=True)
    X_centered = X_train - X_mean
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_mean,X_centered,Vt

def plot_2d_pca(X, y):
    X_mean, X_centered, Vt = pca_dimensions(X)
    X_2d = X_centered @ Vt[:2].T  
    
    plt.figure(figsize=(6,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10', alpha=0.7)
    plt.title("2D PCA Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Class')
    plt.show()


    
