import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
    sgd = SGDClassifier(alpha=0.1, max_iter=100)
    # grid = GridSearchCV(sgd, param_grid=param_grid, cv=5, verbose=5, n_jobs=-1)
    # grid.fit(data_train, labels_train)
    sgd.fit(data_train, labels_train)

    # best_estimator = grid.best_estimator_
    # print(best_estimator)

    #predict for the testing data
    y = sgd.predict(data_test)
    #pring the results
    print("accuracy ")
    accForComb = accuracy_score(labels_test, y)
    print(accForComb)

    # i = 0
    # for couple in couples_test:
    #     if int(y[i]) - int(labels_test[i]) == 0:
    #         print(couple + ", " + str(labels_test[i]) + ", " + str(y[i]))
    #     else:
    #         print(couple + ", " + str(labels_test[i]) + ", " + str(y[i]))
    #     i += 1


if __name__ == '__main__':
    main()

