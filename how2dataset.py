import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import pickle
from tqdm import tqdm
from kaldiio import ReadHelper


class How2(Dataset):

    def __init__(self, video_feat_path, audio_feat_path, text_feat_path, ids, data_type, use_aud):
        self.use_aud = use_aud
        self.ids = ids
        # load video features
        self.img = torch.from_numpy(np.load(video_feat_path + '/' + data_type + '.npy'))

        # load text features
        self.cap = [torch.from_numpy(x) for x in np.load(text_feat_path + '/' + data_type + '.npy')]

        self.data_type = data_type

        # load audio features
        self.aud = torch.from_numpy(np.load(audio_feat_path + '/' + data_type + '.npy'))

        # audio = []
        # for i in tqdm(range(10)):
        #     file = 'scp:'+ audio_feat_path + '/raw_fbank_pitch_all_181506.' + str(i+1) + '.scp'
        #     print(file)
        #     with ReadHelper(file) as reader:
        #         for key, array in reader:
        #             if key in self.ids:
        #                 audio.append(np.mean(array, axis=0))
        #     print(len(audio))
        # self.aud = np.array(audio)
        # print(len(self.img), len(self.cap), len(self.aud))

    def __getitem__(self, index):
        if self.use_aud:
            img_aud = torch.cat([self.img[index], self.aud[index]])
        else:
            img_aud = self.img[index]
        return img_aud, self.cap[index], index

    def __len__(self):
        return len(self.img)

    def dump_file(self):
        # with open('../how2-300h-v1/features/audio/' + self.data_type + '.npy', 'wb') as f:
        #     pickle.dump(self.aud, f)
        np.save('../how2-300h-v1/features/audio/' + self.data_type + '.npy', self.aud)

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    img_aud, captions, ids = zip(*data)

    img_aud = torch.stack(img_aud, 0)

    #images, captions = zip(*data)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return img_aud, targets, lengths, ids

def get_how2_loader(video_feat_path, audio_feat_path, text_feat_path, ids, data_type, opt, batch_size=10, shuffle=True, num_workers=2):
    how2 = How2(video_feat_path, audio_feat_path, text_feat_path, ids, data_type, opt.use_aud)
    data_loader = torch.utils.data.DataLoader(dataset=how2,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              drop_last=True
                                              )
    return data_loader
    

def get_loaders(data_name, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.use_aud:
        opt.img_dim = 2091
    if opt.data_name.startswith('how2'):

        train_ids = []
        with open(dpath+'/train_id.lst') as f:
            for line in f.readlines():
                train_ids.append(line[:-1])

        val_ids = []
        with open(dpath+'/val_id.lst') as f:
            for line in f.readlines():
                val_ids.append(line[:-1])

        test_ids = []
        with open(dpath+'/test_id.lst') as f:
            for line in f.readlines():
                test_ids.append(line[:-1]) 

        text_feat_path = dpath + '/features/scripts'
        video_feat_path = dpath + '/features/resnext101-action-avgpool-300h'
        audio_feat_path = dpath + '/features/audio'

        train_loader = get_how2_loader(video_feat_path, audio_feat_path, text_feat_path, train_ids, 'train', opt, batch_size, True, workers)
        val_loader = get_how2_loader(video_feat_path, audio_feat_path, text_feat_path, val_ids, 'val', opt, batch_size, False, workers)
        test_loader = get_how2_loader(video_feat_path, audio_feat_path, text_feat_path, test_ids, 'dev5', opt, batch_size, True, workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/ubuntu/las',
                        help='path to datasets')
    parser.add_argument('--data_name', default='how2-300h-v1',
                        help='msr-vtt|msvd')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    opt = parser.parse_args()
    print("this is for test dataset")
    video_feat_path = '../how2-300h-v1/features/resnext101-action-avgpool-300h'
    audio_feat_path = '../how2-300h-v1/features/audio'
    text_feat_path = '../how2-300h-v1/features/scripts'

    train_loader, val_loader, test_loader = get_loaders('how2-300h-v1', 224, 100, 1, opt)


    dataset = how2(video_feat_path, audio_feat_path, text_feat_path, ids, 'dev5')
    dataset.__getitem__(0)


