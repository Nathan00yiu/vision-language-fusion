import numpy as np
import torch
from datasets import load_dataset
from src.engine import VisionLanguageEngine
from src.analytics import run_lda_analysis

def evaluate_caltech101():
    print("Loading Caltech-101 subset...")
    dataset = load_dataset("flwrlabs/caltech101", split="train[:500]")
    
    engine = VisionLanguageEngine()
    class_names = dataset.features["label"].names
    
    all_features = []
    all_labels = []
    correct = 0

    for idx, item in enumerate(dataset):
        image = item["image"].convert("RGB")
        label = item["label"]
        
        # Extract features for LDA
        feats = engine.extract_image_features(image).cpu().numpy()
        all_features.append(feats[0])
        all_labels.append(label)
        
        # Zero-shot classification
        top_pred, _ = engine.zero_shot_classify(image, class_names)
        if top_pred == class_names[label]:
            correct += 1

    accuracy = correct / len(dataset)
    print(f"Zero-Shot Top-1 Accuracy: {accuracy * 100:.2f}%")

    # Run LDA analysis
    lda_results = run_lda_analysis(np.array(all_features), np.array(all_labels))
    print(f"LDA Explained Variance Ratio: {lda_results['explained_variance_ratio'][:3]}")

if __name__ == "__main__":
    evaluate_caltech101()