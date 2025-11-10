from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import io
import cv2
import numpy as np
from datetime import datetime

from watermark_embedder import WatermarkEmbedder

app = FastAPI(title="AI Image Generation & Watermarking API")


# Simple in-memory "auth"
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    user_id: str
    token: str


class GenerateRequest(BaseModel):
    user_id: str
    model_name: str
    prompt: str


def _render_placeholder_image(prompt: str, model_name: str, size: tuple = (512, 512)) -> np.ndarray:
    """
    Placeholder image generator. Replace with real model output integration.
    """
    height, width = size
    image = np.full((height, width, 3), 245, dtype=np.uint8)

    # Draw simple grid background
    for i in range(0, width, 32):
        cv2.line(image, (i, 0), (i, height), (220, 220, 220), 1)
    for j in range(0, height, 32):
        cv2.line(image, (0, j), (width, j), (220, 220, 220), 1)

    # Render text: model and prompt
    cv2.putText(image, f"Model: {model_name}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
    wrapped = []
    words = prompt.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if len(test) > 32:
            wrapped.append(line)
            line = w
        else:
            line = test
    if line:
        wrapped.append(line)
    y = 80
    for l in wrapped[:6]:
        cv2.putText(image, l, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        y += 34

    return image


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    # NOTE: Replace with real authentication and user management
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Missing credentials")
    # Simplified user id and token
    user_id = f"user_{abs(hash(req.username)) % 1_000_000}"
    token = f"token_{abs(hash(req.username + req.password)) % 1_000_000}"
    return LoginResponse(user_id=user_id, token=token)


@app.post("/generate-and-watermark")
def generate_and_watermark(req: GenerateRequest):
    """
    1) Generate an image using the selected model (placeholder here)
    2) Embed watermark with user_id, model_name, and current date
    3) Return the watermarked image as PNG
    """
    if not req.user_id or not req.model_name or not req.prompt:
        raise HTTPException(status_code=400, detail="user_id, model_name and prompt are required")

    # Generate image (placeholder)
    generated = _render_placeholder_image(req.prompt, req.model_name)

    # Prepare metadata
    metadata = {
        "user": req.user_id,
        "model": req.model_name,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d")
    }

    # Embed watermark
    embedder = WatermarkEmbedder()
    watermarked = embedder.embed_from_array(generated, metadata)

    # Encode as PNG for response
    success, buf = cv2.imencode(".png", watermarked)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/png",
        headers={
            "X-Watermark-User": metadata["user"],
            "X-Watermark-Model": metadata["model"],
            "X-Watermark-Date": metadata["timestamp"],
        },
    )


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


