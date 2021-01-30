from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.cluster import homogeneity_score
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
from common_functions import GroupedColorFunc


def main():
    nbClusters = 6 #number of clusters
    #load the data
    X = np.loadtxt("./OutputDir/window_matrix.csv", delimiter=",")
    Y = np.loadtxt("./OutputDir/window_matrix_terms_labeled.txt", delimiter=",", dtype=str)
    y_true = Y[:,1].astype(np.int)
    terms =  Y[:,0]
    #data shuffling
    # np.random.shuffle(X)
    #apply k-means
    ldaResult = LatentDirichletAllocation(n_components=nbClusters, random_state=0).fit_transform(X)
    # grid = GridSearchCV(sgd, param_grid=param_grid, cv=5, verbose=5, n_jobs=-1)
    # grid.fit(X, Y)

    # plt.figure()
    # X = PCA(n_components=2).fit_transform(X)
    # plt.scatter(X[:, 0], X[:, 1], c=lda)
    # plt.show()
    homo, y_pred, clusters = printClustersAndComputeHomogeneity(ldaResult, y_true, nbClusters)
    # y_vals = [max(predForATerm) for predForATerm in ldaResult]
    # wordcloud = WordCloud(width=800, height=800,
    #                       background_color='white',
    #                       min_font_size=10).generate_from_frequencies(dict(zip(terms, y_vals)))
    colors = ["red", "blue", "green", "black", "cyan", "yellow"]
    # grouped_color_func = GroupedColorFunc(dict(zip(colors, clusters.values())), "grey")
    # wordcloud.recolor(color_func=grouped_color_func)
    #
    # # plot the WordCloud image
    # plt.figure(figsize=(8, 8), facecolor=None)
    # plt.imshow(wordcloud)
    # plt.axis("off")
    # plt.tight_layout(pad=0)
    # plt.title("Wordcloud for the LDA clustering")
    #
    # plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=ListedColormap(colors, name='from_list', N=None))
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)
    plt.title("Term Classification with LDA", size=20)
    plt.colorbar()
    # vocab = list(Y[:,0])
    # for i, word in enumerate(vocab):
    #     if i%30 == 0:
    #         plt.annotate(word, xy=(X[i, 0], X[i, 1]))
    plt.show()

    return homo

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
        print("cluster: " + str(cluster_id) + " " + str(len(cluster)))
        print(cluster)
    homo = homogeneity_score(y_true, y_pred)
    print("Homogeneity : " + str(homo))
    return homo, y_pred, clusters

if __name__ == '__main__':
    print(main())