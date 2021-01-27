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
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    proportionsData = np.linspace(0.1,0.8, 8)
    fracIrrTerms = []
    nbrOfWords = []
    for proportionData in proportionsData:
        termCoreRelatedWindow(5, window=1, proportionData=proportionData)
        windowGeneration()
        freqIrrTerm, nbrWords = autoLabelling()
        fracIrrTerms.append(freqIrrTerm)
        nbrOfWords.append(nbrWords)
    plt.plot(proportionsData, fracIrrTerms)
    plt.xlabel("Proportion of data used")
    plt.ylabel("Frequency Irrelevant Terms")
    plt.title("Frequency of Irrelevant terms depending on the proportion of data.")
    print("Best Value for :"+str(proportionsData[np.argmin(fracIrrTerms)]))
    plt.show()

    plt.plot(proportionsData, nbrOfWords)
    plt.xlabel("Proportion of data used")
    plt.ylabel("Number of extracted terms")
    plt.title("Number of extracted terms depending on the proportion of data")
    plt.show()