from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ------------------------------------------------------------
# OralSight Demo App
# ------------------------------------------------------------
# Pipeline:
#   1. Upload image
#   2. Oral-validity classifier
#   3. Image quality gate
#   4. Caries-like region detector
#   5. Display safe, non-diagnostic guidance
# ------------------------------------------------------------


APP_DIR = Path(__file__).parent

ORAL_VALIDITY_MODEL_PATH = APP_DIR / "models" / "oralsight_validity_yolov8n_cls_best.pt"
CARIES_MODEL_PATH = APP_DIR / "models" / "caries_yolov8s_aug30_best.pt"


# Decision thresholds
ORAL_ACCEPT_THRESHOLD = 0.80
NON_ORAL_REJECT_THRESHOLD = 0.85

CARIES_CONFIDENCE_THRESHOLD = 0.25


# Quality gate thresholds from Experiment 005 tuned version
MIN_WIDTH_UNUSABLE = 320
MIN_HEIGHT_UNUSABLE = 320

MIN_WIDTH_BORDERLINE = 500
MIN_HEIGHT_BORDERLINE = 500

BLUR_UNUSABLE = 25
BLUR_BORDERLINE = 80

BRIGHTNESS_DARK_UNUSABLE = 40
BRIGHTNESS_DARK_BORDERLINE = 65

BRIGHTNESS_BRIGHT_BORDERLINE = 215
BRIGHTNESS_BRIGHT_UNUSABLE = 240

CONTRAST_UNUSABLE = 18
CONTRAST_BORDERLINE = 32

GLARE_BORDERLINE_PERCENT = 5.0
GLARE_UNUSABLE_PERCENT = 12.0


@st.cache_resource
def load_oral_validity_model():
    return YOLO(str(ORAL_VALIDITY_MODEL_PATH))


@st.cache_resource
def load_caries_model():
    return YOLO(str(CARIES_MODEL_PATH))


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def pil_to_cv2(pil_image):
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


def calculate_quality_metrics(pil_image):
    image = pil_to_cv2(pil_image)

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    b, g, r = cv2.split(image)
    glare_mask = (r > 245) & (g > 245) & (b > 245)
    glare_percent = float(np.sum(glare_mask) / glare_mask.size * 100)

    return {
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "glare_percent": glare_percent,
    }


def classify_quality(metrics):
    reasons = []
    severe_issues = 0
    borderline_issues = 0

    width = metrics["width"]
    height = metrics["height"]
    blur_score = metrics["blur_score"]
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    glare_percent = metrics["glare_percent"]

    # Resolution
    if width < MIN_WIDTH_UNUSABLE or height < MIN_HEIGHT_UNUSABLE:
        severe_issues += 1
        reasons.append("very low resolution")
    elif width < MIN_WIDTH_BORDERLINE or height < MIN_HEIGHT_BORDERLINE:
        borderline_issues += 1
        reasons.append("low resolution")

    # Blur
    if blur_score < BLUR_UNUSABLE:
        severe_issues += 1
        reasons.append("too blurry")
    elif blur_score < BLUR_BORDERLINE:
        borderline_issues += 1
        reasons.append("slightly blurry")

    # Too dark
    if brightness < BRIGHTNESS_DARK_UNUSABLE:
        severe_issues += 1
        reasons.append("too dark")
    elif brightness < BRIGHTNESS_DARK_BORDERLINE:
        borderline_issues += 1
        reasons.append("dark")

    # Too bright
    if brightness > BRIGHTNESS_BRIGHT_UNUSABLE:
        severe_issues += 1
        reasons.append("too bright")
    elif brightness > BRIGHTNESS_BRIGHT_BORDERLINE:
        borderline_issues += 1
        reasons.append("bright")

    # Contrast
    if contrast < CONTRAST_UNUSABLE:
        severe_issues += 1
        reasons.append("low contrast")
    elif contrast < CONTRAST_BORDERLINE:
        borderline_issues += 1
        reasons.append("borderline contrast")

    # Glare
    if glare_percent > GLARE_UNUSABLE_PERCENT:
        severe_issues += 1
        reasons.append("heavy glare")
    elif glare_percent > GLARE_BORDERLINE_PERCENT:
        borderline_issues += 1
        reasons.append("some glare")

    if severe_issues >= 2:
        quality_label = "unusable"
    elif severe_issues == 1 and borderline_issues >= 2:
        quality_label = "unusable"
    elif severe_issues == 1:
        quality_label = "borderline"
    elif borderline_issues >= 2:
        quality_label = "borderline"
    else:
        quality_label = "usable"

    return quality_label, reasons


def run_oral_validity_classifier(model, image_path):
    result = model.predict(str(image_path), verbose=False)[0]

    probs = result.probs
    pred_class_id = int(probs.top1)
    pred_label = result.names[pred_class_id]
    confidence = float(probs.top1conf)

    if pred_label == "oral" and confidence >= ORAL_ACCEPT_THRESHOLD:
        decision = "ACCEPT"
    elif pred_label == "non_oral" and confidence >= NON_ORAL_REJECT_THRESHOLD:
        decision = "REJECT"
    else:
        decision = "UNCERTAIN"

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "decision": decision,
    }


def run_caries_detector(model, image_path):
    results = model.predict(
        str(image_path),
        conf=CARIES_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "confidence": confidence,
                "box": xyxy,
            })

    return annotated_rgb, detections


def show_quality_result(quality_label, reasons, metrics):
    if quality_label == "usable":
        st.success("Image quality: usable")
    elif quality_label == "borderline":
        st.warning("Image quality: borderline")
    else:
        st.error("Image quality: unusable")

    if reasons:
        st.write("Reasons:")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("No major image quality issues detected.")

    with st.expander("View quality metrics"):
        st.write({
            "width": metrics["width"],
            "height": metrics["height"],
            "blur_score": round(metrics["blur_score"], 2),
            "brightness": round(metrics["brightness"], 2),
            "contrast": round(metrics["contrast"], 2),
            "glare_percent": round(metrics["glare_percent"], 2),
        })


def retake_guidance(reasons):
    guidance = []

    if "too blurry" in reasons or "slightly blurry" in reasons:
        guidance.append("Hold the phone steady and keep the teeth in focus.")
    if "too dark" in reasons or "dark" in reasons:
        guidance.append("Use brighter lighting, but avoid harsh direct flash.")
    if "too bright" in reasons or "bright" in reasons or "some glare" in reasons or "heavy glare" in reasons:
        guidance.append("Avoid strong reflections or direct flash on the teeth.")
    if "very low resolution" in reasons or "low resolution" in reasons:
        guidance.append("Upload a clearer, higher-resolution image.")
    if "low contrast" in reasons or "borderline contrast" in reasons:
        guidance.append("Try taking the image in more even lighting.")

    if not guidance:
        guidance.append("Retake with the teeth centred and filling most of the frame.")

    return guidance


def main():
    st.set_page_config(
        page_title="OralSight Demo",
        layout="wide",
    )

    st.title("OralSight Demo")
    st.caption("Experimental AI workflow for oral image quality checking and caries-like region highlighting.")

    st.warning(
        "This is an experimental prototype. It does not provide a dental diagnosis. "
        "Any highlighted regions are AI-generated suspicious caries-like areas and should be reviewed by a dental professional."
    )

    if not ORAL_VALIDITY_MODEL_PATH.exists():
        st.error(f"Missing oral-validity model: {ORAL_VALIDITY_MODEL_PATH}")
        st.stop()

    if not CARIES_MODEL_PATH.exists():
        st.error(f"Missing caries detector model: {CARIES_MODEL_PATH}")
        st.stop()

    oral_validity_model = load_oral_validity_model()
    caries_model = load_caries_model()

    uploaded_file = st.file_uploader(
        "Upload a close-up oral/teeth image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image_path = save_uploaded_file(uploaded_file)
    pil_image = Image.open(image_path).convert("RGB")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Uploaded image")
        st.image(pil_image, use_container_width=True)

    with right_col:
        st.subheader("Step 1 — Oral image validity")

        validity = run_oral_validity_classifier(oral_validity_model, image_path)

        pred_label = validity["pred_label"]
        confidence = validity["confidence"]
        decision = validity["decision"]

        st.write(f"Prediction: **{pred_label}**")
        st.write(f"Confidence: **{confidence:.4f}**")

        if decision == "ACCEPT":
            st.success("Decision: ACCEPT — image appears suitable for oral analysis.")
        elif decision == "UNCERTAIN":
            st.warning(
                "Decision: UNCERTAIN — the image may not show teeth clearly enough. "
                "You can retake the image, or continue in demo mode."
            )
        else:
            st.error(
                "Decision: REJECT — this does not appear to be a suitable close-up oral/teeth image."
            )
            st.write("Please retake with the teeth centred and filling most of the frame.")

    if decision == "REJECT":
        st.stop()

    st.divider()

    st.subheader("Step 2 — Image quality gate")

    quality_metrics = calculate_quality_metrics(pil_image)
    quality_label, quality_reasons = classify_quality(quality_metrics)

    show_quality_result(quality_label, quality_reasons, quality_metrics)

    if quality_label == "unusable":
        st.error("This image quality is too poor for reliable detection.")
        st.write("Retake guidance:")
        for item in retake_guidance(quality_reasons):
            st.write(f"- {item}")

        run_anyway = st.checkbox("Run detector anyway for research/demo purposes")

        if not run_anyway:
            st.stop()

    elif quality_label == "borderline":
        st.warning(
            "Detection can continue, but reliability may be reduced because image quality is borderline."
        )

    st.divider()

    st.subheader("Step 3 — Caries-like region detector")

    annotated_rgb, detections = run_caries_detector(caries_model, image_path)

    result_col_1, result_col_2 = st.columns(2)

    with result_col_1:
        st.image(annotated_rgb, caption="AI-highlighted suspicious regions", use_container_width=True)

    with result_col_2:
        st.write(f"Detected suspicious caries-like regions: **{len(detections)}**")

        if detections:
            for i, detection in enumerate(detections, start=1):
                st.write(f"Region {i}: confidence **{detection['confidence']:.3f}**")
        else:
            st.write("No suspicious caries-like regions were detected.")

        st.info(
            "These results are not a diagnosis. They only show regions that the model considers similar "
            "to labelled caries examples in the training dataset."
        )


if __name__ == "__main__":
    main()