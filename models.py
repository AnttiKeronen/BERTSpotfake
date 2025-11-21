import torch
import torch.nn as nn
from transformers import BertModel

class TextOnlyModel(nn.Module):
    def __init__(self, dropout_p=0.3, fine_tune_text=True):
        super(TextOnlyModel, self).__init__()

        # Load full BERT (trainable)
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            return_dict=True
        )

        # Classification head (stronger than before)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        )

        # Set fine-tuning
        for p in self.bert.parameters():
            p.requires_grad = fine_tune_text  # True = full fine-tuning

    def forward(self, text_ids, text_mask):
        bert_out = self.bert(
            input_ids=text_ids,
            attention_mask=text_mask
        )

        # CLS token embedding
        x = bert_out.last_hidden_state[:, 0, :]  # better than pooler_output

        # Classification
        x = self.classifier(x)
        x = torch.sigmoid(x).squeeze(-1)

        return x.float()
