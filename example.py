# Initialize classifier
from medimageinsightmodel import MedImageInsight
import base64


classifier = MedImageInsight(
    model_dir="2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

# Load model
classifier.load_model()

import urllib.request

image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
image_path = "CXR145_IM-0290-1001.png"

urllib.request.urlretrieve(image_url, image_path)
print(f"Image downloaded to {image_path}")


image = base64.encodebytes(read_image(image_path)).decode("utf-8")

# Example inference
images = [image]
labels = ["normal", "Pneumonia", "unclear"]

#Zero-shot classification
results = classifier.predict(images, labels)
print(results)

#Image embeddings
results = classifier.encode(images = images)
print(results)

#Text embeddings
results = classifier.encode(texts = labels)
print(results)
