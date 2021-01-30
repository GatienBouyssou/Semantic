import heapq

import spacy
from os import listdir
from os.path import isfile, join
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopWords = set(STOP_WORDS)


def main(min_freq = 3, isDep = True, proportionData=0.1):
    """
        corpus parsing + creation of frequent terms file above given minimum threshold
    """
    # inputs
    corpus_dir = r"./Corpus"  #directory for corpus documents
    output_dir = r"./OutputDir" # result file output directory
    output_file = output_dir + r"/processedTestCorpus.txt" # name of the output file (the processed corpus)
    freqTerms_output_file = output_dir + r"/freqTerms.txt" # name of the output file (the frequent terms)
    print("Start Corpus extraction")
    #Process Result: create the processed corpus file and the frequent terms file
    lemmas = dict()
    alldocs = [join(corpus_dir, f) for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]
    f2 = open(output_file, "w", errors='ignore')
    alldocs = alldocs[:int(len(alldocs)*proportionData)]
    nbrOfDocs = len(alldocs)
    for i, doc in enumerate(alldocs):
        # if i%10 ==0: print(i/nbrOfDocs)
        f2.write("<text>" + "\n") # new document
        with open(doc, "rb")as docText:
            for line in docText:
                f2.write("<s>" + "\n") #new sentence
                sent = line.strip().decode("utf-8", "ignore")
                if isDep:
                    parsedSent = nlp(sent) #dependency parsing
                else:
                    parsedSent = nlp(sent, disable=['parser']) #shallow parsing
                index = 0
                for token in parsedSent:
                    if token.pos_ in {"PUNCT"} or token.lemma_ in stopWords:
                        continue
                    if isDep:
                        w = token.text + "\t" + token.lemma_ + "\t" + token.pos_ + "\t" + str(
                            index) + "\t" + token.head.text + "\t" + token.dep_ + "\n"
                    else:
                        w = token.text + "\t" + token.lemma_  + "\t" + token.pos_ + "\t" + str(
                            index) + "\tparent\tdep\n"
                    f2.write(w) #sentence word
                    index += 1
                    lemma = token.lemma_
                    if lemma in lemmas.keys():
                        lemmas[lemma] = lemmas[lemma] + 1
                    else:
                        lemmas[lemma] = 1
                f2.write("</s>" + "\n")
        f2.write("</text>" + "\n")

    #write the frequent lemmas into frequnet file
    freq_file = open(freqTerms_output_file, "w")
    for key in lemmas.keys():
        if lemmas[key] >= min_freq:
            freq_file.write(key + "\n")
    freq_file.close()
    f2.close()
    # print(heapq.nlargest(10, lemmas, key=lemmas.get))
    print("Finish Corpus extraction")

if __name__ == '__main__':
    main(100, True, 0.7)