from sklearn import cluster
from config.model_config import ModelConfig
from website_finder.model.bertclf import BERTClassifier
from website_finder.training.trainer import Trainer
from website_finder.dataloader.dataset import CustomDataLoader
from website_finder.model.clustering import ClusterClassifier
from utility.preprocessing import prepare_data
from transformers import BertTokenizer, AdamW
import pandas as pd
import torch


clf_config = ModelConfig()
data = pd.read_csv(clf_config.datapath)
data, data_category_map = prepare_data(
    data,
    clf_config.data_key_text_col,
    clf_config.data_key_target_col,
    clf_config.data_key_initial_target_col,
)
model = BERTClassifier("bert-base-uncased", clf_config.num_classes)
model.to(clf_config.device)
tokenizer = BertTokenizer.from_pretrained(clf_config.bert_model_ver)
optimizer = torch.optim.Adam(model.parameters(), lr=clf_config.learning_rate)

data_config = {
    "text_col": clf_config.data_key_text_col,
    "target_col": clf_config.data_key_target_col,
}
custom_dataloader = CustomDataLoader(
    data,
    clf_config.bert_model_ver,
    tokenizer,
    clf_config.test_size,
    clf_config.batch_size,
    clf_config.max_len,
    clf_config.random_state,
)

train_loader, test_loader = custom_dataloader.prepare_dataloaders()
trainer = Trainer(
    model,
    train_loader,
    test_loader,
    tokenizer,
    optimizer,
    clf_config.device,
    clf_config.report_tracking_filepath,
)

trainer.initiate_training(clf_config.total_epochs)


def predict_sentiment(text, model, tokenizer, device, max_length=clf_config.max_len):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return preds.item()
        # return "spam" if preds.item() == 1 else "ham"


test_text = "Insert sample text for classification"
category_int = trainer.predict_sample(test_text, clf_config.max_len)

cluster_model = ClusterClassifier(
    data,
    "all-MiniLM-L6-v2",
    clf_config.data_key_text_col,
    clf_config.data_key_target_col,
)

cluster_model.prepare_text_cluster(predicted_category=category_int)
predicted_sites = cluster_model.predict_top_n(test_text, n=10)
print(predicted_sites)
