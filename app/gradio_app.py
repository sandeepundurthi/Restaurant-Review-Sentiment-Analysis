import gradio as gr
import pickle
import re
import numpy as np

# Load model and vectorizer
with open("models/lr_model_3class.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = text.split()
    stopwords = {'the','a','an','and','or','in','on','at','of','this','is','to','with','for','it','was','we','had'}
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

# Predict function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    proba = model.predict_proba(vect)[0]
    confidence = max(proba)
    label = np.argmax(proba)

    sentiments = ["❌ Negative", "😐 Neutral", "✅ Positive"]
    colors = ["#DC2626", "#6B7280", "#16A34A"]

    return f"""
    <div style='font-size:22px; font-weight:bold; color:{colors[label]};'>
        {sentiments[label]} Review
    </div>
    <div style='font-size:16px; color:#555;'>Confidence Score: {confidence:.2f}</div>
    """

# Build layout with Blocks
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""
    <h1 style='text-align:center; font-size:2.2em;'>🍽️ Restaurant Review Sentiment Analyzer</h1>
    <p style='text-align:center; font-size:1.1em; color:gray;'>Classify a restaurant review as Positive, Neutral, or Negative with confidence score </p>
    """)

    with gr.Row():
        input_box = gr.Textbox(lines=4, label="Enter your review", placeholder="e.g. The food was bland and overpriced...")
        output_box = gr.HTML(label="Prediction Result")

    with gr.Row():
        submit_btn = gr.Button("🚀 Analyze", size="lg")
        clear_btn = gr.Button("🧹 Clear")

    submit_btn.click(fn=predict_sentiment, inputs=input_box, outputs=output_box)
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[input_box, output_box])

# Launch app
if __name__ == "__main__":
    demo.launch()
