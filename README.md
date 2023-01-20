Anonymous code repository for the paper submitted in ACL 2023.

## Main Codes are based on JupyterNotebook
Note that these main codes are uploaded before the deadline :)
1) RelatedWordExtractor.ipynb : Semantically related words extractor for Self-Extrofitting
2) ExtroScriptNotebook.ipynb : Script-like codes for Deep-Extrofitting

1+2) SelfExtro.ipynb : Combination of 1) and 2) in GPU version (Updated)

3) TextClassification.ipynb : Codes for downstream tasks.

## update 230121
Due to the lastest update of scikit-learn, vector dimension after extrofitting could be weird.
- In the case of the ‘svd’ solver, the shape is (n_samples, min(rank, n_components)).

The possible solutions are:

- use old version of scikit-learn

- use old version of LDA;

  - change sklearn.discriminant_analysis.LinearDiscriminantAnalysis to sklearn.lda.LDA

Few modifications (maybe data_path) are only required to run them by yourself.

## Other data resources
We are delayed for finding external storage to upload such large datasets (e.g., 4.7GB per a Word Embedding File) in Anonymous.

In GoogleDrive, detailed information includes my name and affiliation.

So... it will be linked very soon!
