from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Upload any image with visible objects to get their dimensions in pixels."
    }

def classify_shape(approx):
    sides = len(approx)
    if sides <= 5:
        return "hook_shape"
    elif 6 <= sides <= 14:
        return "multi-slot block"
    else:
        return "rod"

def calculate_dimensions(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w, h  # in pixels

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    image = cv2.imread(temp_filename)
    os.remove(temp_filename)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        shape = classify_shape(approx)
        width_px, height_px = calculate_dimensions(contour)
        detected.append({
            "shape": shape,
            "width_px": width_px,
            "height_px": height_px
        })

    return {"detected_objects": detected or []}
