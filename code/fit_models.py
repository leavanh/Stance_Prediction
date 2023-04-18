# fits all the BERT and ELECTRA models and saves the accuracy and f1 score of the models on the test data


path = '/home/ubuntu/lrz/thesis/Stance_prediction'

import sys
sys.path.append(path)
import os

from electra import fit_electra
from bert import fit_bert

print("##################################### electra #################################")
print("##################################### no_semsearch #################################")
acc, f1 = fit_electra("input_no_semsearch")
with open(path+'/Results/results_electra.txt', 'a') as f:
    f.write("input_no_semsearch: accuracy = " + str(acc) + ", f1 = " +str(f1)+"\n")

for model in ["sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening", "rankbm25"]:
    print("##################################### " + str(model) + " #################################")
    inputs = ["input11_", "input12_", "input21_", "input22_", "input3_", "input4_"]
    for input in inputs:
        input_name = str(input) + str(model)
        acc, f1 = fit_electra(input_name)
        with open(path+'/Results/results_electra.txt', 'a') as f:
            f.write(str(input_name) + ": accuracy = " + str(acc) + ", f1 = " +str(f1)+"\n")
        
        # remove temporary files to clean up space
        path_delete = '/tmp/'
        temp_files = (file for file in os.listdir(path_delete) if os.path.isfile(os.path.join(path_delete, file)))
        for file in temp_files:
            try:
                os.remove(path_delete + file)
            except OSError as e:
                print("Error: %s : %s" % (file, e.strerror))
        print("TMP FILES REMOVED")

model = "summary_isbert"
print("##################################### " + str(model) + " #################################")
inputs = ["input11_", "input12_", "input21_", "input22_", "input3_", "input4_"]
for input in inputs:
    input_name = str(input) + str(model)
    acc, f1 = fit_electra(input_name)
    with open(path+'/Results/results_electra_summary.txt', 'a') as f:
        f.write(str(input_name) + ": accuracy = " + str(acc) + ", f1 = " +str(f1)+"\n")
    
    # remove temporary files to clean up space
    path_delete = '/tmp/'
    temp_files = (file for file in os.listdir(path_delete) if os.path.isfile(os.path.join(path_delete, file)))
    for file in temp_files:
        try:
            os.remove(path_delete + file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))
    print("TMP FILES REMOVED")

print("##################################### bert #################################")
print("##################################### no_semsearch #################################")
acc, f1 = fit_bert("input_no_semsearch")
with open(path+'/Results/results_bert.txt', 'a') as f:
    f.write("input_no_semsearch: accuracy = " + str(acc) + ", f1 = " +str(f1)+"\n")

for model in ["sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening", "rankbm25"]:
    print("##################################### " + str(model) + " #################################")
    inputs = ["input11_", "input12_", "input21_", "input22_", "input3_", "input4_"]
    for input in inputs:
        input_name = str(input) + str(model)
        acc, f1 = fit_bert(input_name)
        with open(path+'/Results/results_bert.txt', 'a') as f:
            f.write(str(input_name) + ": accuracy = " + str(acc) + ", f1 = " +str(f1)+"\n")
        
        # remove temporary files to clean up space
        path_delete = '/tmp/'
        temp_files = (file for file in os.listdir(path_delete) if os.path.isfile(os.path.join(path_delete, file)))
        for file in temp_files:
            try:
                os.remove(path_delete + file)
            except OSError as e:
                print("Error: %s : %s" % (file, e.strerror))
        print("TMP FILES REMOVED")