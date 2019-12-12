import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import pandas
from models import InferSent
from tqdm import tqdm

from utils import OrderedCounter

from models import InferSent
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = 'fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)
infersent.build_vocab_k_words(K=100000)
infersent = infersent.cuda()

class PoetryDataset(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'poems.csv')
        self.data_file = 'poems.{}.json'.format(self.split)
        self.vocab_file = 'poems.vocab.json'
        self.categories = [
            ['love', 'relationships', 'marriage'],
            ['grieving', 'death', 'sorrow'],
            ['religion', 'faith', 'spiritual'],
            ['animals', 'pets'],
            ['politics', 'conflict'],
            ['nature', 'trees', 'flowers', 'rivers'],
            ['travel', 'journeys']
        ]
        self.categories_len = len(self.categories)

        if create_data:
            print("Creating new %s poem data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'topic': np.asarray(self.data[idx]['topic']),
            'category': np.asarray(self.data[idx]['category']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _find_category(self, tags):
        tags = tags.lower()
        category_vector = [0] * self.categories_len
        for idx, category_strs in enumerate(self.categories):
            for s in category_strs:
                if s in tags:
                    category_vector[idx] = 1
                    break
        return category_vector

    def _create_data(self):

        if self.split == 'train' and not os.path.exists(os.path.join(self.data_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        df = pandas.read_csv(self.raw_data_path)
        
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.8
        if self.split == 'train':
            df = df[mask]
        elif self.split == 'valid':
            df = df[~mask]
        else:
            raise ValueError("Invalid split {}".format(self.split))

        more_than_2_newlines = 0

        max_len = 0
        lens = 0
        for i in tqdm(range(len(df))):
            poem = df.iloc[i]["Poem"]
            poem = poem.replace("\r\r\n", " <nl> ")

            words = tokenizer.tokenize(poem)

            # # Filter out poems that don't have newlines
            # if words.count('<nl>') > 2:
            #     more_than_2_newlines += 1
            # else:
            #     continue

            lens += len(words)
            if len(words) > max_len:
                max_len = len(words)
            
            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length

            if isinstance(df.iloc[i]['Tags'], str):
                topic = df.iloc[i]['Tags']
                data[id]['category'] = self._find_category(topic)
            elif isinstance(df.iloc[i]['Title'], str):
                topic = df.iloc[i]['Title']
                data[id]['category'] = [0] * self.categories_len
            else:
                topic = df.iloc[i]["Poem"].replace("\r\r\n", " ")
                data[id]['category'] = [0] * self.categories_len
            data[id]['topic'] = infersent.encode([topic.strip()], tokenize=True)[0].tolist()

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocabulary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        df = pandas.read_csv(self.raw_data_path)

        for i in range(len(df)):
            poem = df.iloc[i]["Poem"]
            poem = poem.replace("\r\r\n", " <nl> ")

            words = tokenizer.tokenize(poem)

            # Filter out poems that don't have newlines
            if words.count('<nl>') <= 2:
                continue

            w2c.update(words)

        for w, c in w2c.items():
            if c >= self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
