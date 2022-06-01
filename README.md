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
