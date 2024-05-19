# ContentFinder: Intelligent Content Search BERT Transformer and Sentence Clustering

## Overview
ContentFinder is a reverse search application that utilizes BERT (Bidirectional Encoder Representations from Transformers) text classifier model and sentence clustering to facilitate content retrieval based on natural language queries. This project was designed to develop an extensive transformer-based search algorithm for information retrieval. The process is particularly useful for extracting information from large databases when the exact query is unclear, similar to how ChatGPT can respond to conversational queries without needing precisely worded questions. The goal of this project was to create an information retrieval tool to act as an intermediary between users and large-scale documents. For instance, lawyers dealing with extensive documentation could use this model to find relevant laws. While most of the times exact searches may yield no results, this tool could help by showcasing similar laws based on their search queries.

## How it works?
The search process involves two key components:

Fine Tuning BERT: Initially, based on the input database, BERT pre-trained transformer model is fine-tuned on the given database or text documents relevant to the domain of interest. For instance, if the application is meant to find books, the model can be trained on a dataset comprising book descriptions, etc.

Sentence Clustering: Once the BERT model is fine-tuned to your speficiations, it can be used to process your queries, after classifying your query respect to a particular class or category, we can then use the process of sentence clustering using the `all-MiniLM-L6-v2` Sentence Embedding model to now cluster all those text that lie in the same category. Now, we will apply the cosine similarity metric to find the top `n` most closest or related text in the database as compared to your query to provide the best search result.

### Alternative Use Case
If your dataset is unlabelled or contains numerous categories (e.g., book titles and descriptions) and you want to apply this search algorithm, then this process can be reversed on its head. Initially, sentence clustering and cosine similarity can be used to group the texts into `n` zones or categories based on their similarity or relatedness, and then we can label them as such and then use that labels to fine tune the BERT model and then proceed as usual. (Will provide seperate version in future)

## Usage

- 1. Installing the dependencies:
```bash
pip install -r requirements.txt
```
- 2. Modifying the code to your needs and specifications
First, access the `model_config.py` to adjust the model parameters according to your dataset.
```python
@dataclass
class ModelConfig:
    datapath: str = "data/website_classification.csv"
    data_key_text_col: str = "cleaned_website_text"
    data_key_target_col: str = "target"
    data_key_initial_target_col: str = "Category"

    device: str = "cuda"
    learning_rate: float = 2e-05
    bert_model_ver: str = "bert-base-uncased"
    total_epochs: int = 4
    num_classes: int = 16
    test_size: float = 0.2
    random_state: int = 42
    max_len: int = 128
    batch_size: int = 16

    report_tracking_filepath = "reportdata.txt"
```

Once you're happy with the configuration, now you can finder.py to fine-tune the model and then finally modify the following code to answer your search queries!
```python
query_text = "Tell me any website for watching football matches"
category_int = trainer.predict_sample(query_text, clf_config.max_len)
cluster_model = ClusterClassifier(
    data,
    "all-MiniLM-L6-v2",
    clf_config.data_key_text_col,
    clf_config.data_key_target_col,
)
cluster_model.prepare_text_cluster(predicted_category=category_int)
predicted_sites = cluster_model.predict_top_n(query_text, n=10) # set n according to how many search results you wish to process
print(predicted_sites)
```
## Contributing
Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Credits
ContentFinder is developed and maintained by Me!(pShr).