# Use whitening transformation on SBERT embeddings

path = '/home/ubuntu/lrz/thesis/ma_schulzvanheyden/code/'
# path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle

sentences = list(pd.read_excel(path+r'Wahlprogramme/all_sentences.xlsx')[0])

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

embeddings = model.encode(sentences)
embeddings.shape

def whitening(embeddings): #check if does the same as with tensors, check dims
    mu = np.mean(embeddings, axis=0, keepdims=True)
    cov = np.matmul(np.transpose((embeddings - mu)), embeddings - mu)
    u, s, vh = np.linalg.svd(cov)
    W = np.matmul(u, np.diag(1/np.sqrt(s)))
    embeddings = np.matmul(embeddings - mu, W)
    return embeddings, mu, W

sentences_white, mu, W = whitening(embeddings)

with open(path+"Models/sbert_whitening", "wb") as fp:   # Save the mu and W
    pickle.dump([mu, W], fp)

with open(path+"Embeddings/all_sbertwhitening", "wb") as fp:   # Save the embeddings
    pickle.dump(sentences_white, fp)
