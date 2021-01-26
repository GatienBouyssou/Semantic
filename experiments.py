from TermsExtraction import main as termExtraction
from TermsExtractionCoreRelated import main as termCoreRelatedWindow
from CorpusParsing import main as corpusParsing
from termsLabelling import main as autoLabelling
from window_based_matrix_creation import main as windowGeneration
from subject_verb_based_matrix_creation import main as SVmatrixCreation
from SGD import main as exeSGD
from LDAClustering import main as LDAclustering
from common_functions import cartesian
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    windowSizes = [i for i in range(1,15, 1)]
    fracIrrTerms = []
    nbrOfWords = []
    for windowSize in windowSizes:
        termCoreRelatedWindow(5, windowSize)
        windowGeneration()
        freqIrrTerm, nbrWords = autoLabelling()
        fracIrrTerms.append(freqIrrTerm)
        nbrOfWords.append(nbrWords)
    plt.plot(windowSizes, fracIrrTerms)
    plt.xlabel("MinFreq")
    plt.ylabel("Frequency Irrelevant Terms")
    plt.title("Frequency of Irrelevant terms depending on the minFreq value.")
    print("Best Value for :"+str(windowSizes[np.argmin(fracIrrTerms)]))
    plt.show()

    plt.plot(windowSizes, nbrOfWords)
    plt.xlabel("MinFreq")
    plt.ylabel("Number of extracted terms")
    plt.title("Number of extracted terms depending on the minFreq")
    plt.show()