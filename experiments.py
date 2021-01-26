from TermsExtraction import main as termExtraction
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
    minFreqValues = [i for i in range(5,30, 2)]
    fracIrrTerms = []
    for minFreq in minFreqValues:
        termExtraction(minFreq=minFreq)
        windowGeneration()
        fracIrrTerms.append(autoLabelling())
    plt.plot(minFreqValues, fracIrrTerms)
    plt.xlabel("MinFreq")
    plt.ylabel("Homogeneity")
    plt.title("Homogeneity score depending on the minFreq value. Best Value for : "
              +str(minFreqValues[np.argmin(fracIrrTerms)]))
    plt.show()