from tqdm import tqdm
from kaldiio import ReadHelper
import pickle
import sys


def read_feature_data(file):
    keys = []
    audio = []
    with ReadHelper(file) as reader:
        for key, array in reader:
            audio.append(array)
            keys.append(key)
    return keys, audio

if __name__ == '__main__':

    base_path = '/home/ubuntu/data_wsj/'

    audio_feat_paths = ['train_si284', 'test_eval92', 'test_dev93']
    types = ['train', 'eval', 'dev']

    for data_type, audio_feat_path in zip(types, audio_feat_paths):

        file = 'scp:'+ base_path + audio_feat_path + '/feats.scp'
        print(file)

        keys, audio_feat = read_feature_data(file)
        with open(base_path + data_type+'_data.pkl', 'wb') as f:
            pickle.dump(audio_feat, f)
        with open(base_path + data_type+'_key.pkl', 'wb') as f:
            pickle.dump(keys, f)



    
