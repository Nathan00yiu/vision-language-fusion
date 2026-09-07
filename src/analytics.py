import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

def run_lda_analysis(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    features: (N, 512) normalized visual embeddings
    labels: (N,) target class indices
    """
    max_components = min(len(np.unique(labels)) - 1, 10)
    lda = LinearDiscriminantAnalysis(n_components=max_components)
    
    transformed_features = lda.fit_transform(features, labels)
    predictions = lda.predict(features)
    
    return {
        "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
        "report": classification_report(labels, predictions, output_dict=True),
        "transformed_features": transformed_features
    }