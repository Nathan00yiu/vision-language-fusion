import torch
import open_clip
from PIL import Image

# Enable MPS acceleration on Apple Silicon (M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"

class VisionLanguageEngine:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        tensor_img = self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = self.model.encode_image(tensor_img)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def zero_shot_classify(self, image: Image.Image, class_names: list[str]) -> tuple[str, dict]:
        prompts = [f"a photo of a {c}" for c in class_names]
        text_tokens = self.tokenizer(prompts).to(device)

        image_features = self.extract_image_features(image)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity converted to softmax probabilities
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        scores = {c: float(prob) for c, prob in zip(class_names, similarity.tolist())}
        top_prediction = class_names[similarity.argmax().item()]
        return top_prediction, scores