import os
import gradio as gr
from src.engine import VisionLanguageEngine

engine = VisionLanguageEngine()
DEFAULT_CLASSES = ["airplane", "motorcycle", "car", "face", "watch", "background clutter"]

def predict(image, classes_text):
    if image is None:
        return "Please upload an image.", {}
    
    class_list = [c.strip() for c in classes_text.split(",") if c.strip()]
    if not class_list:
        class_list = DEFAULT_CLASSES
        
    top_pred, scores = engine.zero_shot_classify(image, class_list)
    return top_pred, scores

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(value=", ".join(DEFAULT_CLASSES), label="Classes (Comma Separated)")
    ],
    outputs=[
        gr.Textbox(label="Top Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="Multimodal Vision-Language Fusion Engine",
    description="Zero-shot classification & feature evaluation using OpenCLIP."
)


if __name__ == "__main__":
    # Fetch PORT set by Render environment, default to 10000 if running locally
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)