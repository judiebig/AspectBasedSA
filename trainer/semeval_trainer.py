import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.data_filter import *
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


def my_collate(batch_data):
    texts, aspects, texts2num, aspects2num, labels = zip(*batch_data)
    tensor_texts2num = []
    tensor_aspects2num = []
    tensor_label = []
    for text2num, aspect2num, label in zip(texts2num, aspects2num, labels):
        tensor_texts2num.append(torch.Tensor(text2num))
        tensor_aspects2num.append(torch.Tensor(aspect2num))
        tensor_label.append(label+1)
    tensor_texts2num = pad_sequence(tensor_texts2num).permute(1, 0).long()
    tensor_aspects2num = pad_sequence(tensor_aspects2num).permute(1, 0).long()
    tensor_label = torch.Tensor(tensor_label).long()
    return tensor_texts2num, tensor_aspects2num, tensor_label


class DatasetSemeval(Dataset):
    def __init__(self, dataset, indicator, word_dict):
        super(DatasetSemeval, self).__init__()
        self.texts = None
        self.aspects = None
        self.label = None
        self.word_dict = word_dict

        if dataset == 'laptop':
            if indicator == 'train':
                laptop_df = pd.read_csv('output/laptop_data_train.csv')
                # laptop_df = pd.read_csv('output/laptop_data.csv')
                self.texts = laptop_df['text']
                self.aspects = laptop_df['term']
                self.label = laptop_df['polarity']
            elif indicator == 'test':
                laptop_df = pd.read_csv('output/laptop_data_test.csv')
                self.texts = laptop_df['text']
                self.aspects = laptop_df['term']
                self.label = laptop_df['polarity']
        elif dataset == 'restaurant':
            if indicator == 'train':
                laptop_df = pd.read_csv('output/res_data_train.csv')
                # laptop_df = pd.read_csv('output/laptop_data.csv')
                self.texts = laptop_df['text']
                self.aspects = laptop_df['term']
                self.label = laptop_df['polarity']
            elif indicator == 'test':
                laptop_df = pd.read_csv('output/res_data_test.csv')
                self.texts = laptop_df['text']
                self.aspects = laptop_df['term']
                self.label = laptop_df['polarity']
        else:
            raise AssertionError('Not in the scope of hypothesis')

        self.texts2num = word2num(self.texts, self.word_dict)
        self.aspects2num = word2num(self.aspects, self.word_dict)
        self.length = len(self.label)

        print(f"The total number of {indicator} data is {self.length}")

    def __getitem__(self, index: int):
        return self.texts[index], self.aspects[index], self.texts2num[index], self.aspects2num[index], self.label[index]

    def __len__(self):
        return self.length


class SemEvalTrainer(object):
    def __init__(self, model, opt):
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} \t {self.__class__.__name__} initialized.')
        self.opt = opt
        self.model = model
        self.word_dict = None
        self.train_dataloader = None
        self.test_dataloader = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=float(self.opt.lr))

        self.best_score = 0

    def encoder(self, dataset):
        texts = None
        if dataset == 'laptop':
            laptop_df = pd.read_csv('output/laptop_data.csv')
            texts = laptop_df['text']
        if dataset == 'restaurant':
            res_df = pd.read_csv('output/res_data.csv')
            texts = res_df['text']
        filtered_texts = texts.apply(preprocessing)
        word_dict = dict()
        word_dict['NAN'] = 0
        couter = 1
        word_list = []
        for text in filtered_texts:
            for word in text:
                word_list.append(word)
        result = Counter(word_list)
        result = sorted(result.items(), key=lambda d: d[1], reverse=True)
        for key, _ in result:
            word_dict[key] = couter
            couter += 1
        self.word_dict = word_dict
        # print(len(word_dict))

    def load_data(self, dataset):
        self.encoder(dataset)
        train_dataset = DatasetSemeval(dataset, "train", self.word_dict)
        test_dataset = DatasetSemeval(dataset, "test", self.word_dict)
        self.train_dataloader = DataLoader(
            shuffle=True,
            dataset=train_dataset,  # 生成单条数据
            batch_size=int(self.opt.batch_size),
            collate_fn=my_collate,
            drop_last=True
        )
        self.test_dataloader = DataLoader(
            shuffle=False,
            dataset=test_dataset,  # 生成单条数据
            batch_size=int(self.opt.batch_size),
            collate_fn=my_collate,
            drop_last=True
        )

    def train(self):
        for epoch in range(self.opt.epochs):
            print("*"*50)
            self.model.train()
            loss_list = []
            num_correct = []
            for data in self.train_dataloader:
                texts, aspects, labels = data
                preds = self.model(texts)

                self.optimizer.zero_grad()
                loss = self.criterion(preds, labels)
                loss.backward()
                loss_list.append(loss.item())
                self.optimizer.step()
                pred = preds.argmax(dim=1)
                num_correct.append(torch.eq(pred, labels).sum().float().item() * 1.0 / len(labels))
            # print(f"train loss: {np.mean(loss_list)}\t, "
            #       f"train accuracy: {np.mean(num_correct)}")
            print(f"train loss: {np.mean(loss_list)}")

            # evaluate
            self.evaluate()
        print(f'The best accuracy is {self.best_score}')

    def evaluate(self):
        self.model.eval()
        loss_list = []
        num_correct = []
        for data in self.test_dataloader:
            texts, aspects, labels = data
            preds = self.model(texts)
            loss = self.criterion(preds, labels)
            loss_list.append(loss.item())
            pred = preds.argmax(dim=1)
            num_correct.append(torch.eq(pred, labels).sum().float().item() * 1.0 / len(labels))
        print(f"test loss: {np.mean(loss_list)}\t, "
              f"test accuracy: {np.mean(num_correct)}")
        if self.best_score < np.mean(num_correct):
            self.best_score = np.mean(num_correct)





