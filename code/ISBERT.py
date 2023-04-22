# Fit ISBERT model using BERT embeddings
# utils are in a seperate folder: isbert_utils
# Code is from https://github.com/yanzhangnlp/IS-BERT with a few changes to use different base model and data

path = '/home/ubuntu/lrz/thesis/ma_schulzvanheyden/code/'
# path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import sys
sys.path.append(path)
import torch
from torch.utils.data import DataLoader
import math
from isbert_utils import models, losses
from isbert_utils import SentencesDataset, ISBERT, InputExample
import pandas as pd
import numpy as np
import pickle
from transformers import AutoTokenizer

model_name = 'bert-base-german-cased'
train_batch_size = 32
num_epochs = 4
model_save_path = path+'Models/isbert-model'
sentences = list(pd.read_excel(path+r'Wahlprogramme/all_sentences.xlsx')[0])

# Set device
torch.cuda.set_device(-1)
device = torch.device("cuda", 0)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path+"cache")
word_embedding_model = models.Transformer(model_name, cache_dir=path+"cache")

cnn = models.CNN(in_word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(cnn.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = ISBERT(modules=[word_embedding_model, cnn, pooling_model])


# Load data

train_samples = []

for s in sentences:
    sentence = s.strip().split('\t')[0]
    label_id = 1
    train_samples.append(InputExample(texts=[sentence], label=1))


train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss = losses.MutualInformationLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension())

# Configure the training
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

model.save(model_save_path) # added because no evaluator

# get the embeddings

model = ISBERT(model_save_path)
model.train(False)

embeddings = []
for sentence in sentences:
    embedding = model.encode([sentence], convert_to_numpy = True, is_pretokenized = False)[0]
    embeddings.append(embedding)

embeddings = np.array(embeddings)

with open(path+"Embeddings/all_isbert", "wb") as fp:   # Save the embeddings
    pickle.dump(embeddings, fp)
