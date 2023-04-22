# Use existing SBERT model to get embeddings

path = '/home/ubuntu/lrz/thesis/ma_schulzvanheyden/code/'
# path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

sentences = list(pd.read_excel(path+r'Wahlprogramme/all_sentences.xlsx')[0])
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(sentences)

with open(path+"Embeddings/all_sbert", "wb") as fp:   # Save the embeddings
    pickle.dump(embeddings, fp)
