import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

def main():
    #load the data and labels
    X = np.loadtxt("./OutputDir/window_matrix.csv", delimiter=",", dtype=float)
    Y = np.loadtxt("./OutputDir/window_matrix_terms_labeled.txt", delimiter=",", dtype=str)
    #split the data between training and testing
    from sklearn.model_selection import train_test_split
    data_train, data_test, labels_train, labels_test, couples_train, couples_test = train_test_split(X, Y[:, 1], Y[:, 0], test_size=0.20)
    #build the Knn model
    from sklearn.linear_model import SGDClassifier
    param_grid = {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "penalty": ["l2", "l1", "elasticnet"],
        "max_iter": [1,5,10,50,100,250,500,750,1000,2000],
        "alpha": [0.5,0.1,0.01,0.001,0.0001]
    }
    sgd = SGDClassifier(alpha=0.01, loss='modified_huber', max_iter=100, penalty='elasticnet', random_state=10)
    # grid = GridSearchCV(sgd, param_grid=param_grid, cv=5, verbose=5, n_jobs=-1)
    # grid.fit(data_train, labels_train)
    sgd.fit(data_train, labels_train)
    # best_estimator = grid.best_estimator_
    # print(best_estimator)
    # best_estimator.fit(data_train, labels_train)

    #predict for the testing data
    y = sgd.predict(data_test)
    #pring the results
    # print()
    scores = cross_val_score(sgd, X, Y[:, 1], cv=5, scoring=make_scorer(accuracy_score))
    accuracy = np.mean(scores)
    print("Accuracy " + str(accuracy))
    scores = cross_val_score(sgd, X, Y[:, 1], cv=5, scoring=make_scorer(precision_score, average="micro"))
    print("Precision " + str(precision_score(labels_test, y,average="micro")))
    scores = cross_val_score(sgd, X, Y[:, 1], cv=5, scoring=make_scorer(recall_score, average="weighted"))
    print("Recall " + str(recall_score(labels_test, y, average="micro")))
    print("Confusion matrix " + str(confusion_matrix(labels_test, y)))
    y = np.array(sgd.predict(data_test)).astype(int)
    X = PCA(n_components=2).fit_transform(X)
    plt.scatter(data_test[:, 0], data_test[:, 1], c=y)
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)
    plt.title("Term Classification", size=20)
    plt.colorbar()
    vocab = list(couples_test)
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(data_test[i, 0], data_test[i, 1]))
    plt.show()

    # return accuracy
    # i = 0
    # for couple in couples_test:
    #     if int(y[i]) - int(labels_test[i]) == 0:
    #         print(couple + ", " + str(labels_test[i]) + ", " + str(y[i]))
    #     else:
    #         print(couple + ", " + str(labels_test[i]) + ", " + str(y[i]))
    #     i += 1

    plot_confusion_matrix(sgd, data_test, labels_test)
    plt.savefig("../images/finalPlotLDA.png")
    plt.show()

if __name__ == '__main__':
    main()

