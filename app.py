# deploy on Hugging Face spaces

import gradio as gr
from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizer

# load the locally trained model
model = BertForSequenceClassification.from_pretrained("models/local-bert")
tokenizer = BertTokenizer.from_pretrained("models/local-bert")

output = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# define prediction function
def sentiment(text):
    out = output(text)[0]
    return f"Label: {out['label']}\nScore: {out['score']}"

# define gradio interface
demo = gr.Interface(
    fn=sentiment,
    inputs=gr.Textbox(lines=5, label="Enter review:"),
    outputs=gr.Textbox(label="Prediction:"),
    title="Sentiment Analysis using BERT",
    description="Fine-tuned BERT base model for sentiment analysis on hotel reviews",
    allow_flagging="never"
)

# run
demo.launch()
