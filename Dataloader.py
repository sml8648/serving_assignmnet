import transformers
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)

class ReturnData:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

    def tokenizing(self, sentence1, sentence2):

        data = []
        text = '[SEP]'.join([sentence1, sentence2])
        outputs = self.tokenizer(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)
        data.append(outputs['input_ids'])
        return data