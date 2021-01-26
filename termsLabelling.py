
def main():
    mapET_label = {}
    WMT_label = ""
    with open("./OutputDir/ExtractedTermsLabeled.txt", "r") as f:
        for line in f:
            mapET_label[line.split(",")[0]] = int(line.split(",")[1])
    totalWords = 0
    irrelevantTerms = 0
    with open("./OutputDir/window_matrix_terms.txt", "r") as f:
        for word in f:
            newWord = word.replace("\n", "")
            if newWord in mapET_label:
                cluster_id = mapET_label[newWord]
                if cluster_id == 5:
                    irrelevantTerms += 1
                WMT_label += newWord + ", " + str(cluster_id) + "\n"
            else:
                irrelevantTerms += 1
                WMT_label += newWord + ", 5\n"
            totalWords += 1

    with open("./OutputDir/window_matrix_terms_labeled.txt", "w") as f:
        f.write(WMT_label)
    print("Finished labelling")

    return irrelevantTerms/totalWords

if __name__ == '__main__':
    main()