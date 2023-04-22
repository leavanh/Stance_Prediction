# Function that takes a sentence and returns an embedding depending on the semantic search method used

path = '/home/ubuntu/lrz/thesis/ma_schulzvanheyden/code/'
# path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import torch
import pickle
import os
import sys
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from transformers import AutoTokenizer, AutoModel
sys.path.append(path)
from isbert_utils import ISBERT
from Bert_Flow_utils import TransformerGlow

def trun_pad(sentences):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", cache_dir="./cache")
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > 512:
            sent_ids = sent_ids[: 512]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = 512 - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == 512
        assert len(sent_mask) == 512

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)
    features_mask = np.array(features_mask)
    return features_input_ids, features_mask


def embedding(sentence: str, model: str, path_models: str = path + "Models/") -> list:
    """
    Return an embedding for a new query sentence using a pretrained model
    sentence: the query sentence
    model: the pretrained model to use, either "sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening"
    path_models: a path to a folder that contains the pretrained models
    """

    # check if a valid model is selected
    valid_models = {"sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening"}
    if model not in valid_models:
        raise ValueError("embedding: model must be one of %r." % valid_models)

    # check if sentence is a string and only one sentence
    if not isinstance(sentence, str):
        raise TypeError("embedding: sentence is not a string")

    if len(nltk.sent_tokenize(sentence)) > 1:
        raise ValueError("embedding: sentence has to be one sentence and not multiple")

    # Set device
    torch.cuda.set_device(-1)
    device = torch.device("cuda", 0)
    

    if model == "sbert":
        sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        embedding = sbert.encode(sentence)
        embedding = np.array(embedding)

    if model == "isbert":
        isbert = ISBERT(os.path.join(path_models, "isbert-model"))  # Load model
        isbert.train(False)
        
        embedding = isbert.encode(sentence, convert_to_numpy = True, is_pretokenized = False)
        embedding = np.array(embedding)

    if model == "sbertwk":
        with open(os.path.join(path_models, "sbert-wk-model"), "rb") as fp:    # Load model
            sbertwk = torch.load(fp)
        sbertwk.eval()

        input_ids, mask = trun_pad([sentence]) # sentence as list, because trun_pad deals with lists not str
        inputs = {"input_ids": torch.tensor(input_ids, dtype=torch.long).to(device), "attention_mask": torch.tensor(mask, dtype=torch.long).to(device)}

        features = sbertwk(**inputs)[1]

        all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().detach().numpy()

        # use method "ave_last_hidden" as for training (from SBERTWK_utils)
        unmask_num = np.sum(mask, axis=1) - 1 # Not considering the last item
        
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1,:,:]
            embedding.append(np.mean(hidden_state_sen[:sent_len,:], axis=0))

        embedding = np.array(embedding[0])

    if model == "bertflow":
        bertflow = TransformerGlow.from_pretrained(os.path.join(path_models, "bert-flow-model"))  # Load model
        bertflow.train(False)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", cache_dir=path+"cache")
        model_input = tokenizer([sentence], add_special_tokens=True, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        embedding = bertflow(model_input['input_ids'], model_input['attention_mask'])[0]
        embedding = embedding.detach().numpy()

    if model == "sbert-bertflow":
        sbert_bertflow = TransformerGlow.from_pretrained(os.path.join(path_models, "sbert-bert-flow-model"))  # Load model
        sbert_bertflow.train(False)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", cache_dir=path+"cache")
        model_input = tokenizer([sentence], add_special_tokens=True, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        embedding = sbert_bertflow(model_input['input_ids'], model_input['attention_mask'])[0]
        embedding = embedding.detach().numpy()

    if model == "whitening":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", cache_dir=path+"cache")
        model = AutoModel.from_pretrained("bert-base-german-cased", cache_dir=path+"cache")
        pooling = Pooling(512, pooling_mode="mean")
        inputs = tokenizer(sentence, return_tensors='pt', padding='max_length')
        hidden_state = model(**inputs)["last_hidden_state"]
        pooling_inputs = {"token_embeddings": hidden_state, "attention_mask": inputs.attention_mask}
        pooled_embedding = pooling(pooling_inputs)["sentence_embedding"]
        pooled_embedding = pooled_embedding.detach().numpy()

        with open(os.path.join(path_models, "whitening"), "rb") as fp:   # Open the parameters
            mu, W = pickle.load(fp)

        embedding = np.matmul(pooled_embedding - mu, W)[0]

    if model == "sbertwhitening":
        sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        sbert_embedding = sbert.encode(sentence)

        with open(os.path.join(path_models, "sbert_whitening"), "rb") as fp:   # Open the parameters
            mu, W = pickle.load(fp)

        embedding = np.matmul(sbert_embedding - mu, W)[0]

    return embedding