import torch
import string

from torch.nn import LSTM
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

class QuestionEmbeddingModule(torch.nn.Module):

    def __init__(self, name='bert-base-uncased', mask_punctuation=True, output_dim=128):
        super().__init__()
        self.model = BertModel.from_pretrained(name)
        self.model.base = name
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.base)
        self.output_dim = output_dim
        self.lstm = LSTM(self.model.config.hidden_size, self.output_dim, 2, batch_first=True)

        if mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    @property
    def device(self):
        return self.model.device

    def set_device(self, device):
        self.model.to(device)
        self.lstm.to(device)
    
    @property
    def bert(self):
        return self.model

    def forward(self, **kwargs):
        Q = self.query(**kwargs)

        return Q

    def query(self, input_ids, attention_mask, **kwargs):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.lstm(Q)[0]

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        print(Q.shape, mask.shape)
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
