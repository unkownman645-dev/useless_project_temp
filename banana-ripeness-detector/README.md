# 🍌 Banana Maturity Analyzer

> **"Advanced Computer Vision for an Extremely Important Problem."**

A beginner-friendly Computer Vision project built with **Python**, **OpenCV**, **NumPy**, and **Streamlit** that determines banana ripeness from photos and provides culinary recommendations (including emergency banana bread deployment).

---

## 📋 Table of Contents
1. [Project Objective](#-project-objective)
2. [Technologies Used](#-technologies-used)
3. [How the System Works](#-how-the-system-works)
   - [1. Banana Segmentation & Contour Detection](#1-banana-segmentation--contour-detection)
   - [2. RGB to HSV Color Conversion](#2-rgb-to-hsv-color-conversion)
   - [3. Color Quantification](#3-color-quantification)
   - [4. Ripeness Scoring & Classification](#4-ripeness-scoring--classification)
4. [Installation & Setup](#-installation--setup)
5. [How to Run](#-how-to-run)
6. [Ripeness Stages & Culinary Recommendations](#-ripeness-stages--culinary-recommendations)
7. [Limitations](#-limitations)
8. [🚀 Future Upgrade Roadmap](#-future-upgrade-roadmap)

---

## 🎯 Project Objective

Bananas are one of the world's most popular fruits, yet millions are discarded each year due to unexpected overripening. Determining their stage of ripeness is a classic problem in **agricultural computer vision**.

The goal of this project is to provide an accessible, hands-on demonstration of foundational computer vision principles:
- Isolating a target object from backgrounds using **morphological operations** and **contour detection**.
- Using the **HSV (Hue-Saturation-Value)** color space instead of RGB for lighting-invariant color detection.
- Quantifying pixel ratios to compute a continuous **Ripeness Score (0–100%)**.
- Presenting real-time diagnostics via an interactive **Streamlit** web dashboard.

---

## 💻 Technologies Used

| Technology | Role |
| :--- | :--- |
| **Python 3.9+** | Core programming language |
| **OpenCV (`opencv-python-headless`)** | Image transformations, HSV color space masking, contour analysis, morphological filtering |
| **NumPy** | High-performance array operations, pixel masking, and histogram summation |
| **Streamlit** | Interactive modern web dashboard and reactive UI |
| **Altair / Pandas** | Color distribution bar chart and tabular metrics |
| **Pillow (PIL)** | Safe image ingestion and format decoding |

---

## 🧠 How the System Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Upload Photo   │ ──> │ Banana Segmentation  │ ──> │ RGB → HSV Conversion │
│ (JPG, JPEG, PNG)│     │  (Contour Detection) │     │ (Lighting Isolation) │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                │
                                                                ▼
┌────────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│  Streamlit Dashboard   │ <── │ 0–100% Score &     │ <── │ Green/Yellow/Brown │
│  & Emergency Banner    │     │ 5-Stage Category   │     │ Pixel Extraction   │
└────────────────────────┘     └────────────────────┘     └────────────────────┘
```

### 1. Banana Segmentation & Contour Detection
In real-world photos, bananas sit on cutting boards, kitchen counters, or white plates. If we analyze the entire image, the background color would corrupt our measurements.
- We first build a candidate mask covering all potential banana hues (green, yellow, and brown).
- We apply **morphological closing** (`cv2.morphologyEx` with `cv2.MORPH_CLOSE`) to fill internal holes and **opening** (`cv2.MORPH_OPEN`) to discard isolated noise.
- We call `cv2.findContours` to locate continuous boundaries. The largest valid contour represents the banana.
- We draw a filled polygon mask of this contour. All subsequent color analyses are strictly constrained inside this mask.

### 2. RGB to HSV Color Conversion
In the standard **RGB** color model, changing the room's lighting alters all three values (Red, Green, Blue) simultaneously. 
In **HSV**:
- **Hue ($H \in [0, 179]$)** represents the pure chromatic color.
- **Saturation ($S \in [0, 255]$)** represents color intensity/vibrance.
- **Value ($V \in [0, 255]$)** represents brightness.

By filtering predominantly on **Hue**, the system remains stable under shadows and varying ambient brightness.

### 3. Color Quantification
Inside the segmented banana boundary, pixels are filtered into three ranges:
- **🟢 Green Pixels (Unripe flesh & stem):** $H \in [35, 85]$, $S \ge 40$, $V \ge 40$
- **🟡 Yellow Pixels (Ripe skin):** $H \in [17, 34]$, $S \ge 45$, $V \ge 60$
- **🟤 Brown / Dark Spots (Sugar spots & necrosis):** $H \in [6, 22]$, $V \le 140$ or dark spots with $V \le 60$.

Each pixel is assigned exclusively to avoid double-counting. Percentages are normalized against total classified banana pixels:
$$\text{Percentage}_{\text{Color}} = \left( \frac{\text{Pixel Count}_{\text{Color}}}{\text{Total Banana Pixels}} \right) \times 100$$

### 4. Ripeness Scoring & Classification
A weighted continuous index evaluates overall maturity:
$$\text{Ripeness Score} = \min\left(100.0, \, (\% \text{Yellow} \times 0.70) + (\% \text{Brown} \times 1.00)\right)$$

- **$0\%$**: Completely green / raw.
- **$\approx 70\%$**: Golden yellow / prime eating condition.
- **$100\%$**: Highly spotted or fully brown.

---

## 📦 Installation & Setup

### 1. Prerequisites
Make sure you have **Python 3.9** or newer installed. Check your installation by opening a terminal:
```bash
python --version
```

### 2. Navigate to Project Directory
```bash
cd banana-ripeness-detector
```

### 3. Create a Virtual Environment (Recommended)
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The app will open automatically in your default web browser at:
```
http://localhost:8501
```

> **💡 Quick Testing Tip:** Don't have a banana photo on hand? Select **"🧪 Test with Sample Bananas"** in the sidebar to test any of the 5 ripeness stages with procedurally generated banana images!

---

## 🍌 Ripeness Stages & Recommendations

| Stage | Visual Indicator | Ripeness Score / Rule | Recommended Usage |
| :--- | :--- | :--- | :--- |
| **🟢 Unripe** | Green skin, firm | $\text{Green} \ge 40\%$ or $\text{Score} < 30\%$ | Starchy & firm. Best for savory cooking (banana chips, curry) or leave at room temperature for 3–5 days. |
| **🟡 Becoming Ripe** | Yellow with green tips | $\text{Green} \ge 15\%$ or $\text{Score} < 55\%$ | Mild sweetness, higher resistant starch. Great for slicing over oatmeal or breakfast bowls. |
| **🍌 Perfectly Ripe** | Vibrant golden yellow | $55\% - 74\%$ (Low Green) | Peak snacking window! Optimal balance of natural sugars and potassium. |
| **🟤 Very Ripe** | Yellow with brown spots | $75\% - 87\%$ or $\text{Brown} \ge 18\%$ | Sweet and aromatic with sugar spots. Ideal for smoothies, pancakes, and baking. |
| **🚨 Extremely Ripe** | Brown / dark skin | $\ge 88\%$ or $\text{Brown} \ge 38\%$ | **BANANA EMERGENCY!** Sugar peak. Preheat oven to 350°F (175°C) and bake Banana Bread immediately! |

---

## ⚠️ Limitations

1. **Color-Based Ambiguity:**
   - Bananas placed on yellow or green tablecloths/plates may blend into the background if their hues overlap closely.
2. **Extreme Lighting:**
   - Harsh glare (specular reflections) turns pixels pure white ($S \approx 0, V \approx 255$), while heavy shadows can turn yellow areas black.
3. **Bruising vs. Ripeness:**
   - Mechanical damage or impact bruises appear brown even on unripe green bananas.
4. **Single-Perspective Visibility:**
   - A single 2D image only inspects the visible side; the underside of the banana might differ in ripeness.

---

## 🔮 Future Upgrade Roadmap

This beginner project provides an intuitive baseline. Here is how it can be upgraded to production-grade computer vision:

### 1. 🎯 YOLO Object Detection (YOLOv8 / YOLOv11)
- Train or fine-tune an Ultralytics YOLO model on fruits to detect and generate precise bounding boxes around bananas, ignoring distracting background clutter completely.

### 2. 🤖 Deep Learning Classification Model
- Replace heuristic color thresholds with a Convolutional Neural Network (CNN) like **ResNet-50**, **MobileNetV3**, or **EfficientNet**.
- Train on thousands of labeled banana images across standard ripeness scales (CIE L\*a\*b\* or RGB).

### 3. 📹 Live Camera & WebRTC Video Stream
- Integrate `cv2.VideoCapture` or `streamlit-webrtc` to enable live camera analysis, allowing users to hold a banana up to their laptop webcam or smartphone camera for real-time maturity grading.

### 4. ✂️ Automatic Instance Segmentation (SAM / Mask R-CNN)
- Integrate Meta's **Segment Anything Model (SAM)** or **Mask R-CNN** for pixel-perfect mask generation that cleanly isolates the banana silhouette regardless of background complexity.

### 5. 📊 Custom Banana Ripeness Dataset
- Leverage public datasets (e.g., from Kaggle or Roboflow) encompassing diverse varieties (Cavendish, Plantains, Red Bananas, Apple Bananas) across all 7 standard commercial ripening stages.

---

## 📜 License
This project is open-source under the MIT License. Feel free to modify, extend, and share!
