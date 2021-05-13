from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class TextCNN(nn.Module):
    def __init__(self, opt):
        super(TextCNN, self).__init__()
        self.opt = opt
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} \t {self.__class__.__name__} initialized.')
        self.embedding = nn.Embedding(opt.vocab_size, opt.emb_dim)
        #
        self.conv = nn.Sequential(
            nn.Conv1d(opt.emb_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=1),
            nn.Dropout(0.25)
        )
        self.linear = nn.Linear(32, 3, bias=True)

    def _forward_unimplemented(self, *input: Any) -> None:
        raise NotImplementedError

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv(x)
        x = F.max_pool1d(x, x.size(-1)).squeeze()
        x = self.linear(x)
        return x
