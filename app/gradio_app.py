import torch
import torch.nn.functional as F
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification

# ==========================
# Load BERT (bert-base-cased)
# ==========================
bert_model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
bert_model.load_state_dict(torch.load("bert_best_model.pt", map_location="cpu"))
bert_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# ==========================
# Load XLNet (xlnet-base-cased)
# ==========================
xlnet_model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
xlnet_model.load_state_dict(torch.load("xlnet_best_model.pt", map_location="cpu"))
xlnet_model.eval()
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# ==========================
# Prediction function
# ==========================
def predict(text, model_name):
    if model_name == "BERT":
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = bert_model(**inputs)
    else:  # XLNet
        inputs = xlnet_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = xlnet_model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    label = "Fake" if probs[1] > probs[0] else "Real"
    return {"Fake": float(probs[1]), "Real": float(probs[0])}, label

# ==========================
# Gradio UI
# ==========================
def inference(text, model_choice):
    probs, label = predict(text, model_choice)
    return f"Prediction: {label}", probs

demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter a news article or headline..."),
        gr.Radio(["BERT", "XLNet"], value="BERT"),
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Label(label="Probabilities"),
    ],
    title="Fake News Detection App",
    description="Select a model (BERT or XLNet) to classify whether the news is Fake or Real.",
    examples=[
        ["The latest polls show Trump's approval rating increasing steadily over the past month.", "BERT"],
        ["The latest polls show Trump's approval rating increasing steadily over the past month.", "XLNet"],
    ]
)

if __name__ == "__main__":
    demo.launch()
