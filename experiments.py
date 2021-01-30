from TermsExtraction import main as termExtraction
from TermsExtractionCoreRelated import main as termCoreRelatedWindow
from TermsExtractionDepRelated import  main as termAncestorExtraction
from CorpusParsing import main as corpusParsing
from termsLabelling import main as autoLabelling
from window_based_matrix_creation import main as windowGeneration
from subject_verb_based_matrix_creation import main as SVmatrixCreation
from SGD import main as exeSGD
from LDAClustering import main as LDAclustering
from common_functions import cartesian
from matrix_sparsity import  main as matrixSparcity
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    minFreqs = [i for i in range(1, 10)]
    accuracies = []
    accuraciesVerb = []
    homogeneities = []
    homogeneitiesVerb = []
    for minFreq in minFreqs:
        windowGeneration(window_size=minFreq)
        print(autoLabelling())
        homogeneities.append(LDAclustering())
        accuracies.append(exeSGD())
        SVmatrixCreation(window_size=minFreq)
        print(autoLabelling())
        homogeneitiesVerb.append(LDAclustering())
        accuraciesVerb.append(exeSGD())
    print(accuracies)
    print(accuraciesVerb)
    plt.plot(minFreqs, accuracies, label="Window based")
    plt.plot(minFreqs, accuraciesVerb, label="Subject Verb")
    plt.xlabel("minFreq")
    plt.ylabel("Accuracy")
    plt.title("Accuracy depending on the window size.")
    plt.legend()
    print("Best Value for :"+str(minFreqs[np.argmax(accuracies)]))
    plt.show()

    plt.plot(minFreqs, homogeneities, label="Window based")
    plt.plot(minFreqs, homogeneitiesVerb, label="Subject Verb")
    plt.xlabel("minFreq")
    plt.ylabel("Homogeneity")
    plt.title("Homogeneity depending on the window size.")
    plt.legend()
    print("Best Value for :"+str(minFreqs[np.argmax(homogeneities)]))
    plt.show()
