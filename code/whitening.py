# Use whitening transformation on BERT embeddings


path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import pandas as pd
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.models import Pooling
from torch.utils.data import Dataset, DataLoader



# get BERT embeddings

sentences = list(pd.read_excel(path+r'Wahlprogramme/all_sentences.xlsx')[0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", cache_dir=path+"cache")
model = AutoModel.from_pretrained("bert-base-german-cased", cache_dir=path+"cache")
pooling = Pooling(512, pooling_mode="mean")

embeddings = []
for i, sentence in enumerate(sentences):
    print("##################################  "+str(i)+"    ####################################")
    inputs = tokenizer(sentence, return_tensors='pt', padding='max_length')
    hidden_state = model(**inputs)["last_hidden_state"]
    pooling_inputs = {"token_embeddings": hidden_state, "attention_mask": inputs.attention_mask}
    embedding = pooling(pooling_inputs)["sentence_embedding"]
    embedding = embedding.detach().numpy()
    embeddings.extend(np.array(embedding))


with open(path+"Embeddings/mean_pooling", "wb") as fp:   # Save the embeddings
    pickle.dump(embeddings, fp)

# apply whitening

def whitening_torch(embeddings):
    mu = np.mean(embeddings, axis=0, keepdims=True)
    cov = np.matmul(np.transpose((embeddings - mu)), embeddings - mu)
    u, s, vh = np.linalg.svd(cov)
    W = np.matmul(u, np.diag(1/np.sqrt(s)))
    embeddings = np.matmul(embeddings - mu, W)
    return embeddings, mu, W

sentences_white, mu, W = whitening_torch(embeddings)
sentences_white.shape

with open(path+"Models/whitening", "wb") as fp:   # Save the mu and W
    pickle.dump([mu, W], fp)

with open(path+"Embeddings/all_whitening", "wb") as fp:   # Save the embeddings
    pickle.dump(sentences_white, fp)