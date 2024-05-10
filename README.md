# ELEC0141_NLP
Created for completing a final assignment about Natural Language Processing NLP.

The project involves building RAG models for a policy compliance generation task, using gemma 2b it as LLM;

FAISS as similarity search function;

TF-IDF, BERT (Longformer) and Nomic embeddings are tested, with default selection strategy.

# Files in the repository:

**Dataset** folder: contains the original dataset, i.e. policy.xlsx, including policy scripts and 200 different queries regarding of different clauses of the policy, with true labels of what clauses is expected to be retrieved. The dataset is created by our MSc project team and permission from all the teammates and supervisors are obtained.
Results files, i.e. data_bert.csv and data_tfidf.csv for plotting is also stored here.

**Models_notebooks** folder: contains one .ipynb file for implementing each model, tf-idf, longformer, nomic, respectively. A meaningful copy for final generation comparison is also included.

**Images** folder: contains all images produced and potentially useful to be presented in the report.

**requirements.txt**: lists for all dependencies of the implementation evironment.

**main.py**: give a run of the project workflow, going through tf-idf, FAISS, missing inspection and default selection augmentation, and gemma model generation.
