
def main():
    mapET_label = {}
    WMT_label = ""
    with open("./OutputDir/ExtractedTermsLabeled.txt", "r") as f:
        for line in f:
            mapET_label[line.split(",")[0]] = int(line.split(",")[1])

    with open("./OutputDir/window_matrix_terms.txt", "r") as f:
        for word in f:
            newWord = word.replace("\n", "")
            WMT_label += newWord + ", " + str(mapET_label[newWord]) + "\n"

    with open("./OutputDir/window_matrix_terms_labeled.txt", "w") as f:
        f.write(WMT_label)

if __name__ == '__main__':
    main()