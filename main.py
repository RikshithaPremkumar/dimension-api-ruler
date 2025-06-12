from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import os
from collections import Counter

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
        "message": "Upload an image with visible objects. Dimensions are returned in centimeters (cm) using estimated scaling."
    }
PIXELS_PER_CM = 37.8  

def classify_shape(approx):
    sides = len(approx)
    if sides <= 5:
        return "hook_shape"
    elif 6 <= sides <= 14:
        return "multi-slot block"
    else:
        return "rod"

def calculate_dimensions_in_cm(contour):
    x, y, w, h = cv2.boundingRect(contour)
    width_cm = round(w / PIXELS_PER_CM, 2)
    height_cm = round(h / PIXELS_PER_CM, 2)
    return width_cm, height_cm

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
    edged = cv2.Canny(blurred, 20, 100)
    edged = cv2.dilate(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    shape_list = []

    for contour in contours:
        if cv2.contourArea(contour) < 200:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        shape = classify_shape(approx)
        width_cm, height_cm = calculate_dimensions_in_cm(contour)

        detected.append({
            "shape": shape,
            "width_cm": width_cm,
            "height_cm": height_cm
        })
        shape_list.append(shape)

    shape_count = dict(Counter(shape_list))
    return {
        "total_object_count": len(detected),
        "object_type_count": shape_count,
        "detected_objects": detected
    }
