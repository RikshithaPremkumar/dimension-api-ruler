from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import shutil
import uuid
import os
from fastapi.responses import JSONResponse

app = FastAPI()

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
    length = max(w, h) / pixels_per_cm
    width = min(w, h) / pixels_per_cm
    breadth = 0.4 
    return round(length, 2), round(width, 2), breadth

def detect_ruler(contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) != 0 else 0
        area = cv2.contourArea(cnt)
        if 9 < aspect_ratio < 13 and area > 5000:
            return max(w, h)
    return None

@app.post("/")
async def analyze_image(file: UploadFile = File(...)):
    img_id = str(uuid.uuid4())
    img_path = f"{img_id}.jpg"

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ruler_length_px = detect_ruler(contours)
    if not ruler_length_px:
        return JSONResponse({"error": "10 cm ruler not detected. Please include a reference object."}, status_code=400)

    pixels_per_cm = ruler_length_px / 10.0

    results = []
    type_count = {}

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        if cnt is not None and detect_ruler([cnt]) == ruler_length_px:
            continue 

        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        shape = classify_shape(approx)
        length, width, breadth = calculate_dimensions(cnt, pixels_per_cm)

        results.append({
            "type": shape,
            "length": length,
            "width": width,
            "breadth": breadth
        })

        if shape in type_count:
            type_count[shape] += 1
        else:
            type_count[shape] = 1

    os.remove(img_path)

    return JSONResponse({
        "total_objects": len(results),
        "object_types": type_count,
        "dimensions_cm": results
    })
