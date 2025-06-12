from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import uuid
import os
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Upload an image with a visible 10 cm ruler to get object dimensions in cm."}

def calibrate_pixels_per_cm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 9.5 < w / h < 10.5 or 9.5 < h / w < 10.5: 
            return w / 10.0 if w > h else h / 10.0

    return None

def classify_shape(approx):
    sides = len(approx)
    if sides <= 5:
        return "hook_shape"
    elif sides > 6 and sides < 15:
        return "multi-slot block"
    else:
        return "rod"

def calculate_dimensions(contour, pixels_per_cm):
    x, y, w, h = cv2.boundingRect(contour)
    width_cm = round(w / pixels_per_cm, 2)
    height_cm = round(h / pixels_per_cm, 2)
    return width_cm, height_cm

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4()}.jpg"
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())

    image = cv2.imread(filename)
    os.remove(filename)

    pixels_per_cm = calibrate_pixels_per_cm(image)
    if pixels_per_cm is None:
        return JSONResponse(status_code=400, content={"error": "Ruler not detected. Ensure a 10cm ruler is visible."})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dimensions = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        shape = classify_shape(approx)
        w_cm, h_cm = calculate_dimensions(contour, pixels_per_cm)
        dimensions.append({
            "shape": shape,
            "width_cm": w_cm,
            "height_cm": h_cm
        })

    return {"detected_objects": dimensions}
