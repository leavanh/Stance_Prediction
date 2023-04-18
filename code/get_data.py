# Prepares the data, makes the files that are used to fit the semantic search models
# Apply the semantic search models, make the different input patterns
# Split into train and test

path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import pandas as pd
import os
import nltk
import regex as re

from similarity_search import find_sentences

# -----------------------------
# Prepare the party manifestos

# read all manifestos and split into sentences
path_manifestos = path + "Wahlprogramme/"

manifestos = {}
for party in ["gruene", "fdp", "linke", "spd", "cdu", "afd"]:
    path = "{0}{1} 2021.xlsx".format(path_manifestos, party)
    manifesto = pd.read_excel(path)['text']
    manifesto = manifesto.str.cat(sep = " ") # add all paragraphs together
    sentences = nltk.sent_tokenize(manifesto) # split into sentences
    manifestos[party] =  sentences # save in dictionary

# make one excel with all sentences
all_sentences = [*manifestos["gruene"], *manifestos["fdp"], *manifestos["linke"], *manifestos["spd"], *manifestos["cdu"], *manifestos["afd"]]
all_sentences = pd.DataFrame(all_sentences)
party = ["grüne"]*len(manifestos["gruene"])+["fdp"]*len(manifestos["fdp"])+["linke"]*len(manifestos["linke"])+["spd"]*len(manifestos["spd"])+["cdu"]*len(manifestos["cdu"])+["afd"]*len(manifestos["afd"])
all_sentences["party"] = party
all_sentences = all_sentences.sample(frac=1, random_state=910) #shuffle
all_sentences.to_excel(os.path.join(path_manifestos, 'all_sentences.xlsx'), header=True, index=False)

# --------------------------
# Prepare the input data for the model

# read the dataframe that was input for already trained model
data = pd.read_excel(path + r'Data/Input/total.xlsx')
pd.set_option('display.max_columns', None)
data.head(2)

# extract the text, the party and agreement
text = data["text"].str.extract(r":\s(.*)") # gets the text after first : followed bywhitespace
party = data["party"]
party = party.replace({'cdu/csu':'cdu', 'die linke':'linke'}) # replace party names so they fit other code
# data["yes"].eq(data["no"]).any() # check if yes 1 means no 0 and other way around
agreement = data["yes"]
source = data["dataset"]
df = pd.DataFrame({"text": text[0], "party": party, "agreement": agreement, "source": source})

# for each text get 5 manifesto sentences using semantic search

for model in ["sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening", "rankbm25"]:
    model_name = str(model)
    print(model_name)
    results = df.apply(lambda x: find_sentences(x.text, model_name, x.party, 5), axis = 1, result_type ='expand')
    df[model_name] = results.iloc[:,0]
    df[str(model_name+"_scores")] = results.iloc[:,1]
    df.to_excel(path + r'Data/Input/total_extended_complete.xlsx', header=True, index=False)

#------------------------------------

df = pd.read_excel(path + r'Data/Input/total_extended_complete.xlsx')

input_name = "input_no_semsearch"
df[input_name] = df.apply(lambda x: str(x.party)+": "+x.text, axis = 1)

for model in ["sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening", "rankbm25"]:
    print(str(model))
    if model in ["isbert", "sbert-bertflow"]: # only 3 sentences, otherwise too long
        print("3 sentences")
        df["sentences"] = df[str(model)].apply(lambda x: ' '.join(re.findall(r".+(?=',\s'.+',\s')", x)) + "']")
    elif model in ["bertflow"]: # only 4 sentences, otherwise too long
        print("4 sentences")
        df["sentences"] = df[str(model)].apply(lambda x: ' '.join(re.findall(r".+(?=',\s')", x)) + "']")
    else:
        print("5 sentences")
        df["sentences"] = df[str(model)]
    input_name11 = "input11_" + str(model)
    df[input_name11] = df.apply(lambda x: str(x.party)+": "+x["sentences"]+ ", '"+x.text+"'", axis = 1)
    input_name12 = "input12_" + str(model)
    df[input_name12] = df.apply(lambda x: str(x.party)+": '"+x.text+"', "+x["sentences"], axis = 1)
    input_name21 = "input21_" + str(model)
    df[input_name21] = df.apply(lambda x: str(x.party)+": Satz: '"+x.text+"', Kontext: "+x["sentences"], axis = 1)
    input_name22 = "input22_" + str(model)
    df[input_name22] = df.apply(lambda x: str(x.party)+": Kontext: "+x["sentences"]+ ", Satz: '"+x.text+"'", axis = 1)
    input_name3 = "input3_" + str(model)
    df[input_name3] = df.apply(lambda x: "Passt '"+x.text+"' zur "+str(x.party)+" und "+x["sentences"]+"?", axis = 1)
    input_name4 = "input4_" + str(model)
    df[input_name4] = df.apply(lambda x: str(x.party)+" sagt: "+x["sentences"]+", passt '"+x.text+"' dazu?", axis = 1)
    df = df.drop("sentences", axis = 1)

df.to_excel(path + r'Data/Input/total_extended_input.xlsx', header=True, index=False)

# check if the input is too long for the models

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

i = pd.read_excel(path + r'Data/Input/total_extended_input.xlsx')

for model in ["sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening", "rankbm25"]:
    print(str(model))
    for input in ["input11_", "input21_", "input3_", "input4_"]:
        input_name = str(input) + str(model)
        t = i.apply(lambda x: len(tokenizer(x[str(input_name)])["input_ids"]), axis = 1)
        print(t.max())

# isbert, bertflow and sbert-bertflow are too long -> use only the first 4 best sentences for bertflow and 3 for isbert and sbert-bertflow

# split into train, test and validation data

# equal sample all three sources (code from https://drive.google.com/drive/folders/1JBJC9-WzkbhcacEg2RCbij1jeA34mf-x?usp=share_link)
# test makes about about 30% of data
prolific_data = df[df['source'] == "prolific"].sample(frac=1, random_state=853).iloc[:2000, :]
wahlomat_data = df[df['source'] == "wahlomat"].sample(frac=1, random_state=853).iloc[:2000, :]
wahlprogramm_data = df[df['source'] == "program"].sample(frac=1, random_state=853).iloc[:2000, :]
test = pd.concat([prolific_data, wahlprogramm_data, wahlomat_data])

train = df
train = train.drop(test.index)
train = train.sample(frac=1, random_state=853)

train.to_excel(path + r'Data/Input/train_extended_input.xlsx', header=True, index=False)
test.to_excel(path + r'Data/Input/test_extended_input.xlsx', header=True, index=False)