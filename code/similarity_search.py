# Function that takes a sentence and returns the most semantically similar sentences

path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import sys
sys.path.append(path)
import pickle
from sentence_transformers import util
import torch
import nltk
import numpy as np
import pandas as pd
import os

from rank_bm25 import BM25Plus

from new_embedding import embedding


def find_sentences(sentence: str, model: str, party: str, top_k: int, path_embeddings: str = path + "Embeddings/") -> list:
    """
    Return the top_k sentences that are the most similar to the input sentence according to model for the party
    sentence: the query sentence
    model: the pretrained model to use, either "sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening" or "rankbm25"
    party: the party to get the results for, either "grüne", "linke", "spd", "fdp", "cdu" or "afd"
    top_k: how many similar sentences the model should return
    path_embeddings: a path to a folder that contains the embeddings
    """

    # check if a valid model is selected
    valid_models = {"sbert", "isbert", "sbertwk", "bertflow", "sbert-bertflow", "whitening", "sbertwhitening", "rankbm25"}
    if model not in valid_models:
        raise ValueError("embedding: model must be one of %r." % valid_models)

    # check if a valid party is selected
    valid_parties = {"grüne", "linke", "spd", "fdp", "cdu", "afd"}
    if party not in valid_parties:
        raise ValueError("embedding: party must be one of %r." % valid_parties)

    # check if top_k is an integer
    if not isinstance(top_k, int):
        raise TypeError("embedding: top_k is not a integer")

    # check if sentence is a string and only one sentence
    if not isinstance(sentence, str):
        raise TypeError("embedding: sentence is not a string")

    if len(nltk.sent_tokenize(sentence)) > 1:
        raise ValueError("embedding: sentence has to be one sentence and not multiple")

    if model == "rankbm25":
        sentences = pd.read_excel(path + r'Wahlprogramme/all_sentences.xlsx') # read sentences
        party_indices = np.where(sentences.loc[:,"party"]==party) # get the indices of the sentences from the selected party
        party_sentences = sentences.loc[party_indices,0]
        party_sentences = list(party_sentences)

        tokenized_party_sentences = [s.split(" ") for s in party_sentences]

        # TODO: think about stemming and change algorithm to BM25Plus

        bm25 = BM25Plus(tokenized_party_sentences)

        tokenized_sentence = sentence.split(" ")

        results = bm25.get_top_n(tokenized_sentence, party_sentences, n=top_k)
        scores = bm25.get_scores(tokenized_sentence)[:top_k]

        return [results, scores]
    else:
        # get the embedding for the input sentence
        query_embedding = embedding(sentence, model)

        # get all embeddings for the party manifesto
        with open(os.path.join(path_embeddings, "all_" + model), "rb") as fp:   # Open the saved embeddings
            all_embeddings = pickle.load(fp)

        sentences = pd.read_excel(path + r'Wahlprogramme/all_sentences.xlsx') # read sentences
        party_indices = np.where(sentences.loc[:,"party"]==party) # get the indices of the sentences from the selected party
        party_sentences = sentences.loc[party_indices,0]
        party_sentences = list(party_sentences)
        party_embeddings = all_embeddings[party_indices]
        party_embeddings = np.array(party_embeddings)

        # compute the most similar sentences
        similarity = util.dot_score(query_embedding, party_embeddings)
        value, index = torch.topk(similarity, top_k)
        results = [party_sentences[i] for i in index.tolist()[0]]

        return [results, value[0].numpy()]

