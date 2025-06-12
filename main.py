from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import os
from collections import Counter
from typing import List

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

def calculate_dimensions_in_cm(contour):
    x, y, w, h = cv2.boundingRect(contour)
    width_cm = round(w / PIXELS_PER_CM, 2)
    height_cm = round(h / PIXELS_PER_CM, 2)
    return width_cm, height_cm

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <head>
            <title>Object Dimension Detector</title>
        </head>
        <body>
            <h2>Upload an Image to Detect Object Dimensions (in cm)</h2>
            <form action="/analyze/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload and Analyze">
            </form>
        </body>
    </html>
    """

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze_image(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    image = cv2.imread(temp_filename)
    os.remove(temp_filename)

    if image is None:
        return HTMLResponse(content="<h3>Error: Invalid image file.</h3>", status_code=400)

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
    total_objects = len(detected)

    result_html = f"<h2>Total Objects Detected: {total_objects}</h2>"
    result_html += "<h3>Type-wise Count:</h3><ul>"
    for shape, count in shape_count.items():
        result_html += f"<li>{shape}: {count}</li>"
    result_html += "</ul>"

    result_html += "<h3>Object Details:</h3><ol>"
    for obj in detected:
        result_html += (
            f"<li>Shape: {obj['shape']}, Width: {obj['width_cm']} cm, Height: {obj['height_cm']} cm</li>"
        )
    result_html += "</ol>"

    result_html += '<a href="/">&#8592; Upload another image</a>'

    return HTMLResponse(content=result_html)
