import gzip
import string
import numpy as np
from collections import defaultdict
from wordcloud import get_single_color_func


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

def cartesian(arrays, out=None):
    arraysCopy = [np.asarray(x) for x in arrays]
    dtype = arraysCopy[0].dtype

    n = np.prod([x.size for x in arraysCopy])
    if out is None:
        out = np.zeros([n, len(arraysCopy)], dtype=dtype)

    m = int(n / arraysCopy[0].size)
    out[:,0] = np.repeat(arraysCopy[0], m)
    if len(arrays) > 1:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arraysCopy[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def get_sentences(corpus_file):
    """
    Returns all the (content) sentences in a corpus file
    :param corpus_file: the corpus file
    :return: the next sentence (yield)
    """

    # Read all the sentences in the file
    with open(corpus_file, 'r', errors='ignore') as f_in:

        s = []

        for line in f_in:
            line = line

            # Ignore start and end of doc
            if '<text' in line or '</text' in line or '<s>' in line:
                continue
            # End of sentence
            elif '</s>' in line:
                yield s
                s = []
            else:
                try:
                    word, lemma, pos, index, parent, dep = line.split()
                    s.append((word, lemma, pos, int(index), parent, dep))
                # One of the items is a space - ignore this token
                except Exception as e:
                    print(str(e))
                    continue

def save_file_as_Matrix12(cooc_mat, frequent_contexts, output_Dir, MatrixFileName):
    writeMatrix12intoFile(cooc_mat, frequent_contexts, output_Dir+MatrixFileName)


def writeMatrix12intoFile(cooc_mat, frequent_contexts, MatrixFile):
    cnts = ["Terms/contexts"]
    list_of_lists = []
    for context in frequent_contexts:
        cnts.append(context)
    list_of_lists.append(cnts)
    for target, contexts in cooc_mat.items():
        targets = []
        targets.append(target)
        for context in frequent_contexts:
            if context in contexts.keys():
                targets.append(str(contexts[context]))
            else:
                targets.append(str(0))
        list_of_lists.append(targets)
    res = np.array(list_of_lists)
    np.savetxt(MatrixFile, res, delimiter=",", fmt='%s')


def save_file_as_Matrix1(cooc_mat, frequent_contexts, output_Dir, MatrixFileName, MatrixSamplesFileName):
    f = open(output_Dir + MatrixSamplesFileName, "w")
    writeMatrix1intoFile(cooc_mat, frequent_contexts, output_Dir+MatrixFileName, f)
    return

def writeMatrix1intoFile(cooc_mat, frequent_contexts, matrixFile, matrixSamplesFile):
    list_of_lists = []
    for target, contexts in cooc_mat.items():
        targets = []
        matrixSamplesFile.write(target+"\n")
        for context in frequent_contexts:
            if context in contexts.keys():
                targets.append(contexts[context])
            else:
                targets.append(0)
        list_of_lists.append(targets)
    res = np.array(list_of_lists)
    np.savetxt(matrixFile, res, delimiter=",")
    return

def save_file_as_Matrix2(cooc_mat, frequent_contexts, output_Dir, MatrixFileName, frequent_couples):
    list_of_lists = []
    cnts = ["NP_couples/contexts"]
    for context in frequent_contexts:
        cnts.append(context)
    list_of_lists.append(cnts)
    for target, contexts in cooc_mat.items():
        if target in frequent_couples:
            targets = []
            targets.append(str(target).replace("\t", " ## "))
            for context in frequent_contexts:
                if context in contexts.keys():
                    targets.append(str(contexts[context]))
                else:
                    targets.append(str(0))
            list_of_lists.append(targets)
    res = np.array(list_of_lists)
    np.savetxt(output_Dir + MatrixFileName, res, delimiter=",", fmt='%s')
    return

def save_file_as_Matrix(cooc_mat, frequent_contexts, output_Dir, MatrixFileName, MatrixSamplesFileName, frequent_couples):
    f = open(output_Dir + MatrixSamplesFileName, "w")
    list_of_lists = []
    for target, contexts in cooc_mat.items():
        if target in frequent_couples:
            targets = []
            f.write(target+"\n")
            for context in frequent_contexts:
                if context in contexts.keys():
                    targets.append(contexts[context])
                else:
                    targets.append(0)
            list_of_lists.append(targets)
    res = np.array(list_of_lists)
    np.savetxt(output_Dir + MatrixFileName, res, delimiter=",")
    return

def filter_couples(cooc_mat, min_occurrences):
    """
    Returns the couples that occurred at least min_occurrences times
    :param cooc_mat: the co-occurrence matrix
    :param min_occurrences: the minimum number of occurrences
    :return: the frequent couples
    """

    couple_freq = []
    for target, contexts in cooc_mat.items():
        occur = 0
        for context, freq in contexts.items():
            occur += freq
        if occur >= min_occurrences:
            couple_freq.append(target)
    return couple_freq

def filter_contexts(cooc_mat, min_occurrences):
    """
    Returns the contexts that occurred at least min_occurrences times
    :param cooc_mat: the co-occurrence matrix
    :param min_occurrences: the minimum number of occurrences
    :return: the frequent contexts
    """
    context_freq = defaultdict(int)
    for target, contexts in cooc_mat.items():
        for context, freq in contexts.items():
            try:
                str(context)
                context_freq[context] = context_freq[context] + freq
            except:
                continue

    frequent_contexts = set([context for context, frequency in context_freq.items() if frequency >= min_occurrences and context not in string.punctuation])
    return frequent_contexts

def getSentence(strip_sentence):
    """
            Returns sentence (space seperated tokens)
            :param strip_sentence: the list of tokens with other information for each token
            :return: the sentence as string
    """
    sent = ""
    for i, (t_word, t_lemma, t_pos, t_index, t_parent, t_dep) in enumerate(strip_sentence):
        sent += t_word.lower() + " "
    return sent