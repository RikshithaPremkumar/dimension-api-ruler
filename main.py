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

# Estimated pixels per cm (based on 96 DPI screens)
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
        <head><title>Universal Object Detector</title></head>
        <body>
            <h2>Upload an image (with 1 or more objects)</h2>
            <form action="/analyze/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4()}.jpg"
    with open(filename, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(filename)
    os.remove(filename)

    if image is None:
        return HTMLResponse("<h3>Error: Invalid image uploaded.</h3>", status_code=400)

    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # enhance contrast
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Robust edge detection
    edged = cv2.Canny(blurred, 5, 70)
    edged = cv2.dilate(edged, None, iterations=2)

    # Get contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    shape_counts = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 60:  # Accept very small objects too
            continue

        width_cm, height_cm, x, y, w, h = calculate_dimensions(contour)
        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio > 200 or aspect_ratio < 0.01:
            continue  # ignore super-thin lines

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        shape = classify_shape(approx)

        detected.append({
            "shape": shape,
            "width_cm": width_cm,
            "height_cm": height_cm
        })
        shape_counts.append(shape)

    type_count = dict(Counter(shape_counts))

    # Generate HTML output
    html = f"<h2>Total Objects Detected: {len(detected)}</h2>"
    html += "<h3>Type-wise Count:</h3><ul>"
    for shape, count in type_count.items():
        html += f"<li>{shape}: {count}</li>"
    html += "</ul><h3>Object Details:</h3><ol>"
    for obj in detected:
        html += f"<li>Shape: {obj['shape']}, Width: {obj['width_cm']} cm, Height: {obj['height_cm']} cm</li>"
    html += "</ol><a href='/'>← Upload another image</a>"

    return HTMLResponse(content=html)
