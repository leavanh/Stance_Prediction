# Template for Bachelor and Master Theses (Dept. of Statistics, LMU)

## General structure

In general your repository should (in the end) look somehow like this:

```bash
    .
    ├── thesis
    │   ├── thesis.tex                  # or .Rmd, .Rnw or similar 
    │   ├── thesis.pdf                  # pdf-file of your thesis
    ├── data
    │   ├── train_data                  # if applicable
    │   ├── validation_data             # if applicable
    │   ├── test_data                   # if applicable
    ├── code
    │   ├── main.R                      # a file which puts together all the pieces
    │   ├── 01_preprocessing            
    │   ├── 02_descriptives              
    │   ├── ...            
    │   └── requirements.txt            # for reproducibility
    └── ...
``` 

## Minimum requirements

