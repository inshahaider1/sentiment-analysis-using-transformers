# for inference

from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizer

# load the locally trained model
model = BertForSequenceClassification.from_pretrained("models/local-bert")
tokenizer = BertTokenizer.from_pretrained("models/local-bert")

output = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

while True:
    print(output(input("Enter review: ")))