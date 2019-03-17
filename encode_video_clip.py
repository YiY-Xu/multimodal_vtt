from evaluation import encode_data
from model import VSE
import pickle
import torch
import os
import numpy as np

from vocab import Vocabulary
from data_resnet import get_loaders

# Encode all training text to joint embedding space
def encode_video_clip(img_id):

    checkpoint = torch.load('runs/runX/model_best.pth.tar')
    opt = checkpoint['opt']
    model = VSE(opt)
    vocab = pickle.load(open("../vocab/vocab.pkl", "rb"))

    print ('loading image...')
    img = np.load('../resnet_feature/video' + img_id + '.npy')
    img_vector = np.mean(img, axis = 0)
    img_tensor = torch.Tensor([img_vector]).cuda()

    print (img_tensor, img_tensor.size())
    model.img_enc.cuda()

    print ('encoding image...')
    img_embs = model.img_enc.forward(img_tensor)

    print ('outputing...')
    print (img_embs[0], img_embs[0].shape)
    print ('done!')

    return(img_embs[0])

encode_video_clip('84')
