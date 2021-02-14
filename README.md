# Semantic

The idea of this project was to create an ontology automatically from a music corpus. Basically, the idea here is to :
- Extract the frequent terms in the documents,
- Extract the terms related to pre-defined core concepts (Musician, Album, Genre, Instrument, Perfomance)
- Build a co-occurence matrix based on the extracted terms and the frequent terms.
- Use LDA (Latent Dirichlet allocation) and SGD (Stochastic gradient descent) to classify the extracted terms under the core concepts to build a Gold Ontology.


**You can find all our experiments in the notebook Experiments.ipynb**

Compared to the initial files given, we have added the files :
- SDG.py that provides function to run the SGD model on the co-occurrence matrix, 
- LDAClustering.py that runs LDA, 
- TermExtractionCoreRelated.py that provides a window based term extraction centered around core concepts
- and TermExtractionDepRelated.py that is taking the cunjuncts terms of core concepts.
