import spacy
from os import listdir
from os.path import isfile, join
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopWords = set(STOP_WORDS)

def main(minFreq=5, proportionData=0.1):
    """
    document terms extraction
    """
    print("Starting term extraction")
    #inputs
    corpus_dir = r"./Corpus" #directory for corpus documents
    output_dir = r"./OutputDir" #result files output directory
    output_file = output_dir + r"/ExtractedTerms.txt" #the path to save the extracted terms

    terms_file = open(output_file, "w", errors='ignore')
    #compute tf for each term in the corpus
    tf = computerTf(corpus_dir, proportionData)
    #if tf of the term is greater than minimum freq save it to the output file
    for term, score in tf.items():
        if score >= minFreq:
            terms_file.write(str(term) + "\n")
    print("Finish term extraction")


def removeArticles(text):
    #remove stop words from the begining of a NP
    words = text.split()
    if words[0] in stopWords:
        return text.replace(words[0]+ " ", "")
    return text


def textContainsConcept(text, concepts):
    for concept in concepts:
        if concept in text:
            return True
    return False


def getAncestorsOfCoreConcept(relatedTerms, allTerms):
    for relatedTerm in relatedTerms:
        np = removeArticles(relatedTerm.text.lower())
        if np in stopWords:
            continue
        if np in allTerms.keys():
            allTerms[np] += 1
        else:
            allTerms[np] = 1


def computerTf(dir, proportionData=0.1):
    CoreConcepts = {"music", "musician", "album", "genre", "instrument", "performance", 'song', 'release', 'band'}
    alldocs = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    AllTerms = dict()
    alldocs = alldocs[:int(len(alldocs) * proportionData)]
    for i, doc in enumerate(alldocs):
        # if i % 10 == 0: print(i / nbrOfDocs)
        docText = open(doc, "r", errors='ignore').read()
        docParsing = nlp(docText)
        for noun_chunks in docParsing.noun_chunks:
            if not textContainsConcept(noun_chunks.text.lower(), CoreConcepts):
                continue
            conjucts = list(noun_chunks.conjuncts)
            conjucts.append(noun_chunks)
            getAncestorsOfCoreConcept(conjucts, allTerms=AllTerms)
    return AllTerms

if __name__ == '__main__':
    main()
