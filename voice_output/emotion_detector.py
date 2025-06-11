from transformers import pipeline

class EmotionDetector:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

    def detect_emotion(self, text: str) -> str:
        result = self.classifier(text, top_k=1)[0]
        return result["label"]  # e.g., 'fear', 'joy', 'anger'
