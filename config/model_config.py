from dataclasses import dataclass


@dataclass
class ModelConfig:
    datapath: str = "file.csv"
    data_key_text_col: str = "website_url"
    data_key_target_col: str = "target"
    data_key_initial_target_col: str = "Category"

    device: str = "cuda"
    learning_rate: float = 2e-05
    bert_model_ver: str = "bert-base-uncased"
    total_epochs: int = 4
    num_classes: int = 2
    test_size: float = 0.2
    random_state: int = 42
    max_len: int = 128
    batch_size: int = 16

    report_tracking_filepath = "reportdata.txt"
