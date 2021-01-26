from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.cluster import homogeneity_score
from sklearn.model_selection import GridSearchCV


def main():
    nbClusters = 6 #number of clusters
    #load the data
    X = np.loadtxt("./OutputDir/window_matrix.csv", delimiter=",")
    Y = np.loadtxt("./OutputDir/window_matrix_terms_labeled.txt", delimiter=",", dtype=str)
    y_true = Y[:,1].astype(np.int)
    #data shuffling
    # np.random.shuffle(X)
    #apply k-means
    ldaResult = LatentDirichletAllocation(n_components=nbClusters, random_state=0).fit_transform(X)
    # grid = GridSearchCV(sgd, param_grid=param_grid, cv=5, verbose=5, n_jobs=-1)
    # grid.fit(X, Y)

    print(ldaResult)

    # plt.figure()
    # X = PCA(n_components=2).fit_transform(X)
    # plt.scatter(X[:, 0], X[:, 1], c=lda)
    # plt.show()

    return printClustersAndComputeHomogeneity(ldaResult, y_true, nbClusters)


def printClustersAndComputeHomogeneity(ldaResult, y_true, nbClusters):
    clusters = dict()
    for i in range(0, nbClusters):
        clusters[i] = []
    i = 0
    #get the clusters
    y_pred = []
    with open("./OutputDir/window_matrix_terms.txt", "r") as f:
        for line in f:
            term = line.strip()
            clusterNb = np.argmax(ldaResult[i])
            i += 1
            y_pred.append(clusterNb)
            clusters[clusterNb].append(term)
    for cluster_id, cluster in clusters.items():
        print("cluster: " + str(cluster_id))
        print(cluster)
    homo = homogeneity_score(y_true, y_pred)
    print("Homogeneity : " + str(homo))
    return homo
if __name__ == '__main__':
    print(main())