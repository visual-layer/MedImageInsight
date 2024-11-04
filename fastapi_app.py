from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from medimageinsightmodel import MedImageInsight
import base64

# Initialize FastAPI app
app = FastAPI(title="Medical Image Analysis API")

# Initialize model
classifier = MedImageInsight(
    model_dir="2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)
classifier.load_model()


class ClassificationRequest(BaseModel):
    images: List[str]  # Base64 encoded images
    labels: List[str]
    multilabel : bool = False

class EmbeddingRequest(BaseModel):
    images: List[str] = None  # Base64 encoded images
    texts: List[str] = None

@app.post("/predict")
async def predict(request: ClassificationRequest):
    try:
        results = classifier.predict(
            images=request.images,
            labels=request.labels,
            multilabel = request.multilabel
        )
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encode")
async def encode(request: EmbeddingRequest):
    try:
        results = classifier.encode(images=request.images, texts= request.texts)
        results["image_embeddings"] = results["image_embeddings"].tolist() if results["image_embeddings"] is not None else None
        results["text_embeddings"] = results["text_embeddings"].tolist() if results["text_embeddings"] is not None else None

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)