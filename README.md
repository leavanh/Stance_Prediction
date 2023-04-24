# Enhancing Stance Prediction by Utilizing Party Manifestos

This project aims to explore the effectiveness of integrating information from political party manifestos to enhance the accuracy of large language models in predicting party stances. This is done by comparing the performance of two NLP models, ELECTRA and BERT, in order to assess their suitability for this specific task, as well as by using a variety of semantic search techniques and input patterns.


## Contents of the Repository

In this repository you will find the following files:
```bash
    .
    ├── code
    │   ├── Data                                # contains all the data sets which are needed to fit BERT and ELECTRA 
    │   │   ├── total.xlsx                      # dataset that contains party, query and agreement (from Maximilian Witte)
    │   │   ├── total_extended_complete.xlsx    # contains also the context (done by get_data.py)
    │   │   ├── total_extended_input.xlsx       # contains also the different input patterns (done by get_data.py)
    │   │   ├── train_extended_input.xlsx       # training part of total_extended_input (done by get_data.py)
    │   │   └── test_extended_input.xlsx        # testing part of total_extendend_input (done by get_data.py)
    │   ├── Embeddings                          # contains the embeddings for all sentences in the party manifestos
    │   │   ├── all_bertflow
    │   │   ├── all_isbert
    │   │   └── ...                             # for each semantic search method a separate file
    │   ├── isbert_utils                        # folder that contains all the utils needed for ISBERT.py
    │   ├── Plots                               # folder that contains the plots produced by visualize_results.py
    │   ├── Results                             # folder that contains the txt files with the model results
    │   ├── Wahlprogramme                       # contains the party manifestos
    │   │   ├── afd 2021.xlsx
    │   │   ├── cdu 2021.xlsx
    │   │   ├── ...                             # for all the parties
    │   │   └──  all_sentences.xlsx             # random shuffle of all sentences in all party manifestos (done by get_data.py)
    │   ├── Bert_Flow_utils.py                  # utils needed for Bert_Flow.py
    │   ├── Bert_Flow.py                        # semantic search technique BERT-Flow applied to BERT embeddings
    │   ├── bert.py                             # function used in fit_models.py to fit all BERT models
    │   ├── electra.py                          # function used in fit_models.py to fit all ELECTRA models
    │   ├── fit_models.py                       # fits all models and saves the accuracy and f1 score of the models on the test data
    │   ├── get_data.py                         # makes all the datasets used for training (apart from the summarization experiment)
    │   ├── ISBERT.py                           # semantic search technique ISBERT applied to BERT embeddings
    │   ├── new_embedding.py                    # function: sentence -> embedding
    │   ├── SBERT_Bert_Flow.py                  # semantic search technique BERT-Flow applied to SBERT embeddings
    │   ├── SBERT_whitening.py                  # semantic search technique Whitening applied to SBERT embeddings
    │   ├── SBERT.py                            # semantic search technique SBERT
    │   ├── SBERTWK_utils.py                    # utils needed for SBERTWK.py
    │   ├── SBERTWK.py                          # semantic search technique SBERTWK applied to BERT embeddings
    │   ├── similarity_search.py                # function: sentence -> 5 most semantically similar sentences
    │   ├── summarize.py                        # makes the data used for the summarization experiment
    │   ├── visualize_results.py                # plots the tables of the results for all models
    │   ├── whitening.py                        # semantic search technique Whitening applied to BERT embeddings
    │   ├── requirements_fit_models.yaml        # contains the requirements for the virtual environment for fitting the models
    │   └── requirements_sem_search.yaml        # contains the requirements for the semantic search virtual environment
    └── thesis
        ├── thesis.tex                          # or .Rmd, .Rnw or similar 
        ├── thesis.pdf                          # pdf-file of your thesis
        └── bibliography.bib                    # bibtex entries for the references
``` 

## How to use this repository

There are two virtual environments needed as some packages have different dependencies. The following files use the packages from the semantic search environment: ``isbert_utils, BERT_Flow_utils.py, BERT_Flow.py, get_data.py, ISBERT.py, new_embedding.py, SBERT_BERT_Flow.py, SBERT_whitening.py, SBERT.py, SBERTWK_utils.py, SBERTWK.py, similarity_search.py, summarize.py, whitening.py``. The scripts that are used to fit the ELECTRA and BERT models ``bert.py, electra.py, fit_models.py`` and to visualize the results ``visualize_results.py`` use the other virtual environment.

In order to apply the semantic search methods to the party manifestos first the models have to be trained. To do this run the scripts ``BERT_Flow.py, ISBERT.py, SBERT_BERT_Flow.py, SBERT_whitening.py, SBERT.py, SBERTWK.py, whitening.py``. This trains the models and computes the embeddings for all the sentences in the party manifestos. In ``get_data.py`` these embeddings are then used to find the 5 most semantically similar sentences for each query. ``get_data.py`` also prepares the different inputs. Only the summary with IGEL is done in a different script, ``summarize.py``. ``fit_models.py`` fits the BERT and ELECTRA models using the different inputs (please remember to change to a different venv). In order to plot the results run ``visualize_results.py``.