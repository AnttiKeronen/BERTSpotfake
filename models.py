import torch
import torch.nn as nn
from transformers import BertModel

class TextOnlyModel(nn.Module):
    def __init__(self, text_fc1_out=2742, text_fc2_out=32, dropout_p=0.4, fine_tune_text=False):
        super(TextOnlyModel, self).__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            return_dict=True
        )

        self.fc1 = nn.Linear(768, text_fc1_out)
        self.fc2 = nn.Linear(text_fc1_out, text_fc2_out)
        self.out = nn.Linear(text_fc2_out, 1)

        self.dropout = nn.Dropout(dropout_p)

        # freeze if needed
        for p in self.bert.parameters():
            p.requires_grad = fine_tune_text

    def forward(self, text_ids, text_mask):
        bert_out = self.bert(input_ids=text_ids, attention_mask=text_mask)
        x = bert_out["pooler_output"]

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.out(x)).squeeze(-1)
        return x.float()
