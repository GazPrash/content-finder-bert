import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataloader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        tokenizer,
        device,
        report_path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.report_path = report_path

    def train(self):
        self.model.train()
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            labels = labels.to(torch.int64)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                labels = labels.to(torch.int64)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        return accuracy_score(actual_labels, predictions), classification_report(
            actual_labels, predictions
        )

    def write_results(self, accuracy, report):
        report_json = json.dumps(report, indent=4)
        output_text = f"Accuracy: {accuracy}\n\nClassification Report:\n{report_json}"
        with open(self.report_path, "w") as file:
            file.write(output_text)
        print(f"Results stored in {self.report_path}")

    def initiate_training(self, total_epochs):
        for epoch in range(total_epochs):
            print(f"Epoch {epoch + 1}/{4}")
            self.train()
            accuracy, report = self.evaluate()
            self.write_results(accuracy, report)
            # print(f"Validation Accuracy: {accuracy:.4f}")
            # print(report)

    def predict_sample(self, text, max_length):
        self.model.eval()
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            return preds.item()
            # return "spam" if preds.item() == 1 else "ham"
        # result = predict_sentiment(test_text, model, tokenizer, device)
        # print(test_text)
        # print(f"Predicted sentiment: {result}")
