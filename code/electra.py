# function to fit ELECTRA models

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import set_seed
import evaluate
import datasets
from datasets import Dataset, DatasetDict
from datetime import datetime
import torch
import wandb


def fit_electra(input_name, batches = 16, epochs = 13, lr = 2.21e-5): # defaults (except batch size) are from original project (Maximilian Witte)
  path = '/home/ubuntu/lrz/thesis/ma_schulzvanheyden/code/'
  # path = "/home/ubuntu/lrz/thesis/Stance_prediction/"

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  set_seed(42)

  wandb.init(project="electra", name = str(input_name))

  # read data
  df_train = pd.read_excel(path + f"Data/Input/train_extended_input.xlsx", index_col=None)
  df_test = pd.read_excel(path + f"Data/Input/test_extended_input.xlsx", index_col=None)

  df_train = df_train.loc[:,[str(input_name), "agreement"]].rename({'agreement': 'label', str(input_name): 'text'}, axis=1)
  df_test = df_test.loc[:,[str(input_name), "agreement"]].rename({'agreement': 'label', str(input_name): 'text'}, axis=1)

  train = Dataset.from_pandas(df_train)
  test = Dataset.from_pandas(df_test)

  data = DatasetDict()
  data['train'] = train
  data['test'] = test
  

  # preprocess data
  tokenizer = AutoTokenizer.from_pretrained("german-nlp-group/electra-base-german-uncased", cache = path + "cache")

  def preprocess_function(data):
    return tokenizer(data['text'], truncation=True)
  
  tokenized_data = data.map(preprocess_function, batched=True)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # prepare evaluation
  accuracy = evaluate.load("accuracy")
  f1_metric = evaluate.load("f1")

  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      acc = accuracy.compute(predictions=predictions, references=labels)
      f1 = f1_metric.compute(predictions=predictions, references=labels)
      return {"accuracy": acc, "f1": f1}
  
  # train
  id2label = {0: "NEGATIVE", 1: "POSITIVE"}
  label2id = {"NEGATIVE": 0, "POSITIVE": 1}

  model = AutoModelForSequenceClassification.from_pretrained("german-nlp-group/electra-base-german-uncased", num_labels=2, id2label=id2label, label2id=label2id)
  model = model.to(device)

  model_path = path + "Models/stance_prediction/" + f'electra_{str(input_name)}'
  training_args = TrainingArguments(output_dir = model_path, 
                                  per_device_train_batch_size = batches, 
                                  per_device_eval_batch_size = batches,
                                  num_train_epochs = epochs, learning_rate = lr,
                                  evaluation_strategy="epoch",
                                  save_strategy="no")

  trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_data['train'], eval_dataset=tokenized_data['test'], tokenizer=tokenizer,
                     data_collator=data_collator, compute_metrics=compute_metrics)

  trainer.train()

  trainer.save_model()

  # evaluate

  results = trainer.evaluate()

  wandb.finish()

  acc = round(results['eval_accuracy']['accuracy'], 3)
  f1 = round(results['eval_f1']['f1'], 3)

  now = datetime.now()

  current_time = now.strftime("%H:%M:%S")
  print(f"{current_time}--- New model electra_{str(input_name)} saved with mean acc of {acc} % and f1 of {f1} %. Parameter {batches}, {lr}, {epochs} ---")

  return acc, f1
