import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }


class CustomDataLoader:
    def __init__(
        self,
        data,
        bert_model_ver,
        tokenizer,
        test_size,
        batch_size,
        max_len,
        random_state,
        data_config,
    ) -> None:
        self.data = data
        self.bert_model_ver = bert_model_ver
        self.tokenizer = tokenizer
        self.test_size = test_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.random_state = random_state
        self.data_config = data_config

    def prepare_dataloaders(self):
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.data[self.data_config["text_col"]],
            self.data[self.data_config["target_col"]],
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_texts.index = np.arange(len(train_texts))
        test_texts.index = np.arange(len(test_texts))
        train_labels.index = np.arange(len(train_labels))
        test_labels.index = np.arange(len(test_labels))

        tokenizer = BertTokenizer.from_pretrained(self.bert_model_ver)

        train_dataset = CustomDataset(
            train_texts, train_labels, tokenizer, self.max_len
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, self.max_len)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return (train_loader, test_loader)
