
# Sentiment Analysis on Hotel Reviews using BERT and XLNet

The goal of this project was to design and implement a Sentiment Analysis model for Hotel Reviews using state-of-the-art Transformer Architectures. Sentiment Analysis refers to the use of Natural Language Processing (NLP) techniques to classify data under class labels like negative, neurtral or positive. In this project, BERT and XLNet have been fine-tuned for Sentiment Analysis on a [Hotel Reviews dataset](https://www.kaggle.com/code/skappal7/hotel-review-sentiment-topic-modelling/input).


## Setup Instructions

This project is basd on Python. Download and install [Python](https://www.python.org/downloads/). If you plan to train the model locally on your NVIDIA GPU, it is recommended to install [CUDA](https://developer.nvidia.com/cuda-downloads) for hardware acceleration.  


- Clone this repository
```
  git clone <repo url>
```
- Navigate to the repo
```
  cd sentiment-analysis-using-transformers
```

- Create a virtual environment
```
  python -m venv <env name>
```

- Activate the environment
```
  \<env name>\Scripts\activate
```

- Install all the required packages listed in `requirements.txt`
```
  pip install -r requirements.txt
```


## Usage Guide

The notebooks for preprocessing the dataset and training the model are in the `sentiment-analysis` directory. They can be 
executed in a Jupyter Notebook environment.

Run `test_model.py` for inference in terminal or run `app.py` for a browser based interface powered by [Gradio](https://www.gradio.app/).

The trained models and checkpoints are saved in the `models` directory.

`data` contains the Hotel Reviews dataset.


## Results

| Model  | Evaluation Loss | Accuracy | Precision | Recall | F1-Score | Epoch |
|--------|-----------------|----------|-----------|--------|----------|-------|
| BERT   | 0.5827          | 75.42%   | 75.32%    | 75.42% | 75.29%   | 5.0   |
| XLNet  | 0.5242          | 77.40%   | 77.15%    | 77.40% | 77.22%   | 5.0   |

While both models are effective for Sentiment Analysis, XLNet appears to have a slight advantage over BERT. 

