# Use IGEL model (https://huggingface.co/philschmid/instruct-igel-001) to summarize the 5 ISBERT input sentences into 1
# apply the patterns and save so models can be trained on the new input

path = '/home/ubuntu/lrz/thesis/Stance_prediction/'


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001", cache_dir = path + "cache")

model = AutoModelForCausalLM.from_pretrained("philschmid/instruct-igel-001", cache_dir = path + "cache")

summarizer = pipeline(task="text-generation", model=model, tokenizer=tokenizer)


# read data
df_train = pd.read_excel(path + f"Data/Input/train_extended_input.xlsx", index_col=None)
df_test = pd.read_excel(path + f"Data/Input/test_extended_input.xlsx", index_col=None)

model_name = "isbert"
for data in ["train", "test"]:
    if data == "train":
        df = df_train
    else:
        df = df_test

    # get correct column
    input = df.loc[:,[str(model_name)]].values.tolist()

    # add command in front
    input = ["### Anweisung:\nAls eins Satz zusammenfassen: " + str(s) + "\n\n### Antwort:" for s in input]

        # get summary
    results = summarizer(input, return_full_text=False)
    results_flattened = [d['generated_text'] for inner_list in results for d in inner_list]

    # add to dataframe
    column_name = "summary_" + str(model_name)
    df[column_name] = results_flattened

    # add columns for the different input patterns
    df["sentences"] = df[column_name].apply(lambda x: '"' + x.lstrip('\n') + '"') # remove /n and add "" around text
    input_name11 = "input11_" + str(column_name)
    df[input_name11] = df.apply(lambda x: str(x.party)+": "+x["sentences"]+ ", '"+x.text+"'", axis = 1)
    input_name12 = "input12_" + str(column_name)
    df[input_name12] = df.apply(lambda x: str(x.party)+": '"+x.text+"', "+x["sentences"], axis = 1)
    input_name21 = "input21_" + str(column_name)
    df[input_name21] = df.apply(lambda x: str(x.party)+": Satz: '"+x.text+"', Kontext: "+x["sentences"], axis = 1)
    input_name22 = "input22_" + str(column_name)
    df[input_name22] = df.apply(lambda x: str(x.party)+": Kontext: "+x["sentences"]+ ", Satz: '"+x.text+"'", axis = 1)
    input_name3 = "input3_" + str(column_name)
    df[input_name3] = df.apply(lambda x: "Passt '"+x.text+"' zur "+str(x.party)+" und "+x["sentences"]+"?", axis = 1)
    input_name4 = "input4_" + str(column_name)
    df[input_name4] = df.apply(lambda x: str(x.party)+" sagt: "+x["sentences"]+", passt '"+x.text+"' dazu?", axis = 1)
    df = df.drop("sentences", axis = 1)

    # save
    df.to_excel(path + f'Data/Input/{str(data)}_extended_input.xlsx', header=True, index=False)
