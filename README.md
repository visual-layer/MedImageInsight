---
license: mit
tags:
- medical
---
# MedImageInsight: Open-Source Medical Image Embedding Model

This repository provides a simplified implementation for using the MedImageInsight model, an open-source medical imaging embedding model presented in the paper [MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging](https://arxiv.org/abs/2410.06542) by Noel C. F. Codella et al. The official guide to access the model from Microsoft is quite complicated, and it is arguable whether the model is truly open-source. This repository aims to make it easier to use the MedImageInsight model for various tasks, such as zero-shot classification, image embedding, and text embedding.

What we have done:

- Downloaded the models from Azure
- Got rid of all the unnecessary files
- Got rid of unnecessary MLflow code to make a standalone implementation
- Moved to uv for dependency management
- Added multi-label classification
- Created an example with the FastAPI service

## Usage

1. Clone the repository and navigate to the project directory.

Make sure you have git-lfs installed (https://git-lfs.com)
```bash
git lfs install
```

```bash
git clone https://huggingface.co/lion-ai/MedImageInsights
```
3. Install the required dependencies
We are using [uv](https://github.com/astral-sh/uv) package manager to simplify the installation.

To create a virtual env, simply run:
```bash
uv sync
```
Or to run a single script, just run:
```bash
uv run example.py
```

That's it!

## Examples
See to the `example.py` file.
### Zero-shot image classification

Here's an example of how to use the `MedImageInsight` class for zero-shot classification:

```python
# Initialize classifier
classifier = MedImageInsight(
    model_dir="2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)

# Load model
classifier.load_model()

# Read image
image = base64.encodebytes(read_image("image.png")).decode("utf-8")

# Zero-shot classification
images = [image]
labels = ["normal", "Pneumonia", "unclear"]
results = classifier.predict(images, labels)
print(results)
```
###  Multi-label zero-shot image classification
Run multi-label image classification (without softmax at the end)
```python
# Multilabel classification example
images = [image]
labels = ["normal", "Pneumonia", "Fracture", "Tumor"]
results = classifier.predict(images, labels, multilabel=True)
print(results)

```
### Image embeddings

```python
results = classifier.encode(images=images)
print(results["image_embeddings"])
```

### Text embeddings
```python
results = classifier.encode(texts=labels)
print(results["text_embeddings"])
```
### FastAPI server
```bash
uv run fastapi_app.py
```
Go to localhost:8000/docs to see the swagger.

The application provides endpoints for classification and image embeddings. Images have to be base64 encoded.


## Roadmap
- [x] Basic implementation
- [x] Multilabel classification
- [x] FastAPI service
- [ ] HF compatible API (from_pretrained())
- [ ] Explainability

## Acknowledgments

This repository is based on the work presented in the paper "MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging" by Noel C. F. Codella et al. ([arXiv:2410.06542](https://arxiv.org/abs/2410.06542)).