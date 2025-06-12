from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import uuid
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PIXELS_PER_CM = 37.8

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
    width_cm = round(w / PIXELS_PER_CM, 2)
    height_cm = round(h / PIXELS_PER_CM, 2)
    return width_cm, height_cm, x, y, w, h

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head><title>Object Detector</title></head>
        <body>
            <h2>Upload an image to detect all objects and get measurements in cm</h2>
            <form action="/analyze/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    temp_file = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(temp_file)
    os.remove(temp_file)

    if image is None:
        return HTMLResponse("<h3>Error: Invalid image file.</h3>", status_code=400)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 5, 90)
    edged = cv2.dilate(edged, None, iterations=3)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    shape_list = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        width_cm, height_cm, x, y, w, h = calculate_dimensions(contour)

        if width_cm < 0.2 or height_cm < 0.2:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        shape = classify_shape(approx)

        detected.append({
            "shape": shape,
            "width_cm": width_cm,
            "height_cm": height_cm
        })
        shape_list.append(shape)

    shape_counts = dict(Counter(shape_list))

    html = f"<h2>Total Objects Detected: {len(detected)}</h2>"
    html += "<h3>Type-wise Count:</h3><ul>"
    for k, v in shape_counts.items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul><h3>Object Details:</h3><ol>"
    for obj in detected:
        html += f"<li>Shape: {obj['shape']}, Width: {obj['width_cm']} cm, Height: {obj['height_cm']} cm</li>"
    html += "</ol><a href='/'>← Upload another image</a>"

    return HTMLResponse(html)
