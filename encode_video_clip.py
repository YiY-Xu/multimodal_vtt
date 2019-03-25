from evaluation import encode_data
from model import VSE
import pickle
import torch
import os
import numpy as np
import faiss 

#from vocab import Vocabulary
from data_resnet import get_loaders


# Encode all training text to joint embedding space
def encode_video_clip(img_id):

    checkpoint = torch.load('runs/runX/model_best.pth.tar')
    opt = checkpoint['opt']
    model = VSE(opt)
    vocab = pickle.load(open("../vocab/vocab.pkl", "rb"))

    text_embedding = np.load('../Text_encoded.npy').astype('float32')

    test_imgs = np.load('../how2-300h-v1/features/resnext101-action-avgpool-300h/val.npy').astype('float32')

    print ('loading image...')
    #img = np.load('../resnet_feature/video' + img_id + '.npy')
    #img_vector = np.mean(img, axis = 0)

    img_vector = test_imgs[img_id]
    img_tensor = torch.Tensor([img_vector]).cuda()

    print (img_tensor, img_tensor.size())
    model.img_enc.cuda()

    print ('encoding image...')
    img_embs = model.img_enc.forward(img_tensor).detach().numpy()

    print ('outputing...')
    print (img_embs[0], img_embs[0].shape)
    print ('done!')

    index = faiss.IndexFlatL2(text_embedding.shape[1])
    index.add(text_embedding)

    k = 3
    
    D, I = index.search(img_embs[0].reshape(-1, 1), k)

    print(D)
    print(I)

    return(D, I)


if __name__ == '__main__':
    encode_video_clip(15)
