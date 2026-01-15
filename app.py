from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import urllib.request
from PIL import Image
import json

# ----------------- Download Face Detection Model -----------------
MODEL_PATH = "detector.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

if not os.path.exists(MODEL_PATH):
    print("Downloading face detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# ----------------- Load Skin Type Model -----------------
skin_model = load_model("skin_type_model.h5")
skin_labels = ["dry", "normal", "oily"]

# ----------------- Flask App -----------------
app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)

# ----------------- Image Quality Assessment -----------------
def assess_image_quality(image_path):
    """
    Assess image quality (blur, lighting, framing)
    Returns: (is_good_quality, issues_list, quality_score)
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, ["Cannot read image"], 0.0
    
    issues = []
    scores = []
    
    # 1. Blur Detection using Laplacian variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(laplacian_var / 100.0, 1.0)  # Normalize
    scores.append(blur_score)
    
    if laplacian_var < 100:
        issues.append("Image appears blurry. Please use a clearer photo.")
    
    # 2. Lighting Assessment using brightness
    brightness = np.mean(gray)
    if brightness < 50:
        issues.append("Image is too dark. Please use better lighting.")
        lighting_score = brightness / 50.0
    elif brightness > 200:
        issues.append("Image is overexposed. Please reduce lighting.")
        lighting_score = (255 - brightness) / 55.0
    else:
        lighting_score = 1.0
    scores.append(lighting_score)
    
    # 3. Resolution Check
    height, width = img.shape[:2]
    if width < 640 or height < 480:
        issues.append("Image resolution is too low. Please use higher quality camera.")
        resolution_score = min((width * height) / (640 * 480), 1.0)
    else:
        resolution_score = 1.0
    scores.append(resolution_score)
    
    # Calculate overall quality score
    quality_score = np.mean(scores)
    is_good_quality = quality_score >= 0.6 and len(issues) <= 1
    
    return is_good_quality, issues, quality_score

# ----------------- Face Detection -----------------
def detect_face_from_path(image_path):
    """
    Detect and crop face from image using MediaPipe Face Detection
    Returns cropped face image or None if no face detected
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Configure face detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    
    # Detect faces
    with vision.FaceDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)
        
        if not detection_result.detections:
            return None
        
        # Choose the largest face
        best = max(
            detection_result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height
        )
        
        bbox = best.bounding_box
        h, w, _ = img.shape
        
        x = bbox.origin_x
        y = bbox.origin_y
        bw = bbox.width
        bh = bbox.height
        
        # Add padding around face
        pad_w = int(0.15 * bw)
        pad_h = int(0.25 * bh)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)
        
        face = img[y1:y2, x1:x2]
        return face

# ----------------- Skin Type Prediction with Confidence -----------------
def detect_skin_type(face_img):
    """
    Predict skin type from cropped face image
    Returns (skin_type, confidence_score, all_predictions)
    """
    resized = cv2.resize(face_img, (128, 128))
    arr = img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = skin_model.predict(arr, verbose=0)[0]
    predicted_idx = np.argmax(preds)
    skin_type = skin_labels[predicted_idx]
    confidence = float(preds[predicted_idx] * 100)
    
    # Get all predictions for transparency
    all_preds = {
        skin_labels[i]: float(preds[i] * 100) 
        for i in range(len(skin_labels))
    }
    
    return skin_type.capitalize(), confidence, all_preds

# ----------------- Enhanced Product Recommendations -----------------
def get_product_suggestions(skin_type):
    """
    Get comprehensive product recommendations based on skin type
    Organized by category with descriptions
    """
    recommendations = {
        "Oily": {
            "description": "Your skin produces excess sebum, which can lead to shine and enlarged pores. Focus on oil-control and lightweight hydration.",
            "morning_routine": [
                {"product": "Gentle foaming cleanser", "reason": "Removes excess oil without stripping skin"},
                {"product": "Oil-free toner with niacinamide", "reason": "Minimizes pores and controls sebum"},
                {"product": "Lightweight gel moisturizer", "reason": "Hydrates without adding shine"},
                {"product": "Mattifying sunscreen SPF 50+", "reason": "Protects skin with shine control"}
            ],
            "evening_routine": [
                {"product": "Salicylic acid cleanser", "reason": "Deep cleans pores"},
                {"product": "BHA exfoliating toner", "reason": "Prevents breakouts"},
                {"product": "Niacinamide serum", "reason": "Regulates oil production"},
                {"product": "Oil-free night gel", "reason": "Repairs without clogging pores"}
            ],
            "weekly_treatments": [
                {"product": "Clay mask", "reason": "Draws out impurities and excess oil"},
                {"product": "Chemical exfoliant (AHA/BHA)", "reason": "Unclogs pores deeply"}
            ],
            "avoid": ["Heavy creams", "Oil-based products", "Comedogenic ingredients"]
        },
        "Dry": {
            "description": "Your skin lacks moisture and may feel tight or flaky. Focus on intense hydration and barrier repair.",
            "morning_routine": [
                {"product": "Creamy hydrating cleanser", "reason": "Cleanses without stripping moisture"},
                {"product": "Hydrating toner with hyaluronic acid", "reason": "Boosts moisture levels"},
                {"product": "Rich moisturizer with ceramides", "reason": "Locks in hydration and repairs barrier"},
                {"product": "Moisturizing sunscreen SPF 50+", "reason": "Protects while hydrating"}
            ],
            "evening_routine": [
                {"product": "Oil-based cleanser", "reason": "Gently removes makeup"},
                {"product": "Essence or hydrating toner", "reason": "Prepares skin for treatment"},
                {"product": "Hyaluronic acid serum", "reason": "Deep hydration"},
                {"product": "Overnight sleeping mask", "reason": "Intensive moisture repair"}
            ],
            "weekly_treatments": [
                {"product": "Hydrating sheet mask", "reason": "Instant moisture boost"},
                {"product": "Gentle enzyme exfoliant", "reason": "Removes dead skin without irritation"}
            ],
            "avoid": ["Alcohol-based products", "Harsh scrubs", "Foaming cleansers"]
        },
        "Normal": {
            "description": "Your skin has a balanced level of moisture and oil. Maintain this balance with gentle, nourishing products.",
            "morning_routine": [
                {"product": "Gentle gel cleanser", "reason": "Cleanses without disrupting balance"},
                {"product": "Balancing toner", "reason": "Maintains pH balance"},
                {"product": "Lightweight moisturizer", "reason": "Keeps skin comfortable"},
                {"product": "Broad-spectrum sunscreen SPF 50+", "reason": "Daily protection"}
            ],
            "evening_routine": [
                {"product": "Micellar water or gentle cleanser", "reason": "Removes impurities gently"},
                {"product": "Antioxidant toner", "reason": "Protects from environmental stress"},
                {"product": "Vitamin C or retinol serum", "reason": "Anti-aging and brightening"},
                {"product": "Balancing night cream", "reason": "Overnight nourishment"}
            ],
            "weekly_treatments": [
                {"product": "Hydrating face mask", "reason": "Maintains optimal moisture"},
                {"product": "Gentle exfoliator", "reason": "Promotes cell turnover"}
            ],
            "avoid": ["Overly harsh products", "Unnecessary actives"]
        }
    }
    
    return recommendations.get(skin_type, recommendations["Normal"])

# ----------------- Skincare Tips -----------------
def get_skincare_tips(skin_type):
    """
    Generate personalized skincare tips
    """
    tips = {
        "Oily": [
            "Wash your face twice daily, but avoid over-cleansing",
            "Use blotting papers throughout the day to control shine",
            "Don't skip moisturizer - even oily skin needs hydration",
            "Choose non-comedogenic makeup and skincare products",
            "Stay hydrated - drinking water helps regulate oil production"
        ],
        "Dry": [
            "Avoid hot water - use lukewarm water for cleansing",
            "Apply moisturizer immediately after washing while skin is damp",
            "Use a humidifier in dry environments",
            "Protect skin from harsh weather with barrier creams",
            "Drink plenty of water and eat omega-3 rich foods"
        ],
        "Normal": [
            "Maintain your routine consistently",
            "Protect your skin from sun damage daily",
            "Adjust products seasonally as needed",
            "Focus on anti-aging and prevention",
            "Stay consistent with healthy lifestyle habits"
        ]
    }
    
    return tips.get(skin_type, tips["Normal"])

# ----------------- Routes -----------------
@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Process uploaded image and analyze skin type
    """
    # Check if image was uploaded
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Save uploaded image
    upload_path = "static/uploads/uploaded.jpg"
    file.save(upload_path)

    # Step 1: Image Quality Assessment
    is_good_quality, quality_issues, quality_score = assess_image_quality(upload_path)
    
    if not is_good_quality:
        return render_template(
            "result.html",
            error="Image Quality Issues Detected",
            quality_issues=quality_issues,
            quality_score=round(quality_score * 100, 1),
            show_retake=True
        )

    # Step 2: Detect face
    try:
        face = detect_face_from_path(upload_path)
        if face is None:
            return render_template(
                "result.html",
                error="No face detected. Please upload a clear face image with your face centered.",
                show_retake=True
            )
    except Exception as e:
        return render_template(
            "result.html",
            error=f"Error processing image: {str(e)}",
            show_retake=True
        )

    # Save cropped face
    face_path = "static/uploads/face.jpg"
    cv2.imwrite(face_path, face)

    # Step 3: Predict skin type with confidence
    skin_type, confidence, all_predictions = detect_skin_type(face)

    # Step 4: Get comprehensive recommendations
    recommendations = get_product_suggestions(skin_type)
    
    # Step 5: Get skincare tips
    tips = get_skincare_tips(skin_type)

    return render_template(
        "result.html",
        skin_type=skin_type,
        confidence=round(confidence, 1),
        all_predictions=all_predictions,
        recommendations=recommendations,
        tips=tips,
        quality_score=round(quality_score * 100, 1),
        face_url=url_for("static", filename="uploads/face.jpg")
    )

@app.route("/about")
def about():
    """About page with methodology info"""
    return render_template("about.html")

# ----------------- Error Handlers -----------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html"), 500

# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)