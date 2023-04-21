In this repository you will find the following files
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
    │   └── requirements.yaml                   # contains the conda environment
    └── thesis
        ├── thesis.tex                          # or .Rmd, .Rnw or similar 
        ├── thesis.pdf                          # pdf-file of your thesis
        └── bibliography.bib                    # bibtex entries for the references
``` 






























# Template for Bachelor and Master Theses (Dept. of Statistics, LMU)

This repository is intended to serve as a really basic template for theses, containing merely empty folders for pre-defining the structure for a repository. You are not obligated to work within GitLab or use git at all (despite we highly recommend it), but for the final submission you need to provide a repository with a similar structure (see below) and running code.  
 
Use it via `New Project > Create from template > Group > thesis-template`.  
(_Important:_ Make sure to adjust the path in the project url to the current semester)

A template for the tex-file can be found here: https://www.overleaf.com/latex/templates/lmu-slds-thesis-template/mhnhsykmqpvm

# Some more detailed instructions on how to kick things off

- You should create the directory for your project in the current semester's directory (e.g. for the summer semester 2022 your directory should be in https://gitlab.lrz.de/statistics/summer22). 
- Your directory shouldbe named with the type of thesis/project and your surname (e.g. for a Bachelor thesis by Ludwig Meier it should be named "ba_meier").
- Once you created your directory, you should clone it to your computer with the help of a git client using the option "clone". You can use any git client for this (if you have no experience and you are working on a WIndows machine, we can recommend Github Desktop as it is very user friendly).
- After you cloned the repo, you'll be able to automatically syncronize the work on your computer by using the commit and push options. Please do so regularly, as this will greatly help you (and your supervisors/collaborator) with version control!
- You can immediately update the template (found at the above link) by adding your name and the title of your project, and then push the updates to the repo to check if everything is working as intended.
- You can also add your project supervisor to the git when you want, so that they can easily review your work and provide feedback.


# Requirements and recommendations on how to organize your work

General structure

In general your repository should (in the end) look somehow like this:

```bash
    .
    ├── thesis
    │   ├── thesis.tex                  # or .Rmd, .Rnw or similar 
    │   ├── thesis.pdf                  # pdf-file of your thesis
    │   ├── bibliography.bib            # bibtex entries for the references
    ├── data
    │   ├── train_data                  # these data sets can be excerpts from the original data
    │   ├── validation_data             # or even simulated data sets (simply provide them so that
    │   ├── test_data                   # your code can be executed)
    ├── code
    │   ├── main.R/.py                  # a file which puts together all the pieces
    │   ├── 01_preprocessing.R/.py            
    │   ├── 02_descriptives.R/.py              
    │   ├── ...            
    │   └── requirements.txt            # for reproducibility
    ├── results                         # raw data results of experiments
    └── ...
``` 

Further, you should provide

- a short explanation of your thesis project in the README
- explanation of the folder structure and the different scripts

More in general, remember that the idea is for everything to be reproducible as easily as possible: you can add anything that you think would be helpful for the purpose. Good luck!
