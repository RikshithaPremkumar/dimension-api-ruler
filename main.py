# main.py
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import shutil
import uvicorn
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

def calculate_dimensions(contour, scale_factor):
    x, y, w, h = cv2.boundingRect(contour)
    length = max(w, h) * scale_factor
    width = min(w, h) * scale_factor
    breadth = 0.4 
    return round(length, 2), round(width, 2), breadth

@app.post("/analyze/")
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
    scale_factor = 1 / 100.0

    results = []
    type_count = {}

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        shape = classify_shape(approx)
        length, width, breadth = calculate_dimensions(cnt, scale_factor)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
