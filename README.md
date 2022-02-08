# Template for Bachelor and Master Theses (Dept. of Statistics, LMU)

This repository is intended to serve as a really basic template for theses, containing merely empty folders for pre-defining the structure for a repository. You are not obligated to work within GitLab or use git at all (despite we highly recommend it), but for the final submission you need to provide a repository with a similar structure (see below) and running code.  
 
Use it via `New Project > Create from template > Group > thesis-template`.  

## General structure

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

## Minimum requirements

Further you should provide

- a short explanation of your thesis project in the README
- explanation of the folder structure and the different scripts
