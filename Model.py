import transformers
import torch
from tqdm.auto import tqdm

from transformers import RobertaForSequenceClassification, AutoConfig

class Model(RobertaForSequenceClassification):

    def __init__(self):
        config = AutoConfig.from_pretrained('klue/roberta-large')
        super(Model, self).__init__(config)

        # 사용할 모델을 호출합니다.
        config.num_labels = 1
        self.plm = RobertaForSequenceClassification.from_pretrained('klue/roberta-large', config=config).to('cuda')

    def forward(self, x):

        x = torch.tensor(x).to('cuda')
        x = self.plm(x)['logits']
        return x