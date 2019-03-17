from evaluation import encode_data
from model import VSE
import pickle
import torch
import os
import numpy as np

from vocab import Vocabulary
from data_resnet import get_loaders

# Encode all training text to joint embedding space

checkpoint = torch.load('runs/runX/model_best.pth.tar')
opt = checkpoint['opt']
model = VSE(opt)

vocab = pickle.load(open("../vocab/vocab.pkl", "rb"))

print ('loading text...')
train_loader, val_loader = get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)
print (val_loader)
print ('encoding text...')
_, cap_embs = encode_data(model, train_loader)

print ('outputing embeddings...')
np.save('../Text_encoded.npy', cap_embs)
print (cap_embs.shape)
print ('done!')
