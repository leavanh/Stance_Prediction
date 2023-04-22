# Fit BERT-Flow model using SBERT embeddings
# Code is from https://github.com/UKPLab/pytorch-bertflow with a few changes to use different base model and data

path = '/home/ubuntu/lrz/thesis/ma_schulzvanheyden/code/'
# path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import sys
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

sys.path.append(path)
from Bert_Flow_utils import TransformerGlow, AdamWeightDecayOptimizer

batch_size = 64 # as in paper
num_epochs = 2
max_length = 512 
model_name_or_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
bertflow = TransformerGlow(model_name_or_path, pooling='first-last-avg')  # pooling could be 'mean', 'max', 'cls' or 'first-last-avg' (mean pooling over the first and the last layers)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=path + "cache")
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters= [
    {
        "params": [p for n, p in bertflow.glow.named_parameters()  \
                        if not any(nd in n for nd in no_decay)],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in bertflow.glow.named_parameters()  \
                        if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamWeightDecayOptimizer(
    params=optimizer_grouped_parameters, 
    lr=1e-3, 
    eps=1e-6,
)

# ------------------------
# Load data
class Manifestos(Dataset):
    def __init__(self):
        # data loading
        self.sentences = list(pd.read_excel(path+r'Wahlprogramme/all_sentences.xlsx')[0])
        self.n_samples = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return self.n_samples

dataset = Manifestos()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# -----------------------------------------------
# Train model
bertflow.train()
for epoch in range(num_epochs):
    for i, sentences in enumerate(dataloader):
        print("############# "+str(i)+" #############")
        model_inputs = tokenizer(
            sentences,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
        z, loss = bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=True)  # Here z is the sentence embedding
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


bertflow.save_pretrained(path+'Models/sbert-bert-flow-model')  # Save model

# -------------------------
# Compute embeddings

bertflow = TransformerGlow.from_pretrained(path+'Models/sbert-bert-flow-model')  # Load model
bertflow.train(False)

embeddings = []

for i, sentences in enumerate(dataloader):
    model_inputs = tokenizer(
        sentences,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    z = bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=False)  # Here z is the sentence embedding
    embeddings.append(z)

embeddings = [item for sublist in embeddings for item in sublist] # flatten the different batches
embeddings = torch.stack(embeddings).cpu().detach() # otherwise the file is too big

with open(path+"Embeddings/all_sbert-bertflow", "wb") as fp:   # Save the embeddings
    pickle.dump(embeddings, fp)
