"""
=============================================================================
🍌 BANANA MATURITY ANALYZER
"Advanced Computer Vision for an Extremely Important Problem."
=============================================================================
A beginner-friendly Computer Vision project built with Python, OpenCV,
NumPy, and Streamlit to accurately detect and classify banana ripeness.

Pipeline:
1. Image Ingestion & Validation (Upload or Synthetic Samples)
2. Banana Segmentation & Contour Detection (Background filtering)
3. RGB to HSV Color Space Conversion
4. Green, Yellow, and Brown Pixel Quantification
5. Ripeness Score Calculation (0 - 100%)
6. Ripeness Stage Classification & Culinary Recommendations
7. Banana Emergency Response System
=============================================================================
"""

import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Banana Maturity Analyzer",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional engineering dashboard aesthetic with a playful touch
st.markdown("""
<style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Emergency Alert Animation */
    @keyframes emergency-pulse {
        0% { border-color: #ff1744; box-shadow: 0 0 10px rgba(255, 23, 68, 0.5); }
        50% { border-color: #ff9100; box-shadow: 0 0 25px rgba(255, 145, 0, 0.8); }
        100% { border-color: #ff1744; box-shadow: 0 0 10px rgba(255, 23, 68, 0.5); }
    }
    
    .emergency-banner {
        background: linear-gradient(135deg, #b71c1c 0%, #d50000 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff5252;
        animation: emergency-pulse 1.5s infinite;
        margin-bottom: 20px;
    }
    
    .emergency-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    
    .card-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SYNTHETIC SAMPLE GENERATOR (FOR INSTANT TESTING WITHOUT LOCAL FILES)
# -----------------------------------------------------------------------------
def generate_sample_banana_image(sample_type: str) -> np.ndarray:
    """
    Generates a realistic synthetic banana image using OpenCV drawing functions.
    Allows beginners to test the entire CV pipeline immediately even without
    having their own photos ready.
    """
    # Create white canvas (representing a clean table/counter background)
    h, w = 400, 500
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    # Base banana banana curve parameters (approximated using an ellipse & polylines)
    center = (250, 200)
    axes = (160, 65)
    angle = -18

    # Determine base BGR color & spots according to sample type
    if sample_type == "🟢 Unripe Banana":
        base_color = (45, 180, 50)      # Vivid Green
        tip_color = (35, 130, 30)       # Dark Green stem
        spots_count = 0
    elif sample_type == "🟡 Becoming Ripe":
        base_color = (30, 215, 235)     # Yellow body
        tip_color = (45, 175, 50)       # Green stem
        spots_count = 3
    elif sample_type == "🍌 Perfectly Ripe":
        base_color = (25, 215, 248)     # Golden Yellow
        tip_color = (30, 80, 120)       # Brownish stem
        spots_count = 8
    elif sample_type == "🟤 Very Ripe":
        base_color = (35, 185, 215)     # Deep Yellow with spotting
        tip_color = (25, 45, 75)        # Dark brown stem
        spots_count = 40
    else:  # 🚨 Extremely Ripe
        base_color = (30, 115, 155)     # Dark Amber/Brown
        tip_color = (15, 25, 45)        # Very dark stem
        spots_count = 120

    # Draw main banana body (curved banana shape via overlapping filled ellipses)
    cv2.ellipse(img, center, axes, angle, 0, 360, base_color, -1, cv2.LINE_AA)

    # For "Becoming Ripe", overlay green coloration on the ends/tips
    if sample_type == "🟡 Becoming Ripe":
        cv2.ellipse(img, (135, 225), (45, 28), angle, 0, 360, (45, 180, 50), -1, cv2.LINE_AA)
        cv2.ellipse(img, (375, 150), (45, 26), angle, 0, 360, (45, 180, 50), -1, cv2.LINE_AA)

    # For "Extremely Ripe", overlay large brown necrotic soft patches
    if sample_type == "🚨 Extremely Ripe":
        cv2.ellipse(img, (220, 205), (50, 26), angle + 10, 0, 360, (25, 55, 95), -1, cv2.LINE_AA)
        cv2.ellipse(img, (315, 180), (45, 24), angle - 10, 0, 360, (25, 50, 85), -1, cv2.LINE_AA)

    # Cut top ellipse to give banana crescent curve
    cv2.ellipse(img, (250, 165), (150, 55), angle, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)

    # Add banana stem / tips
    cv2.circle(img, (110, 235), 14, tip_color, -1, cv2.LINE_AA)
    cv2.circle(img, (395, 140), 12, tip_color, -1, cv2.LINE_AA)

    # Add random brown sugar spots / necrotic patches
    if spots_count > 0:
        np.random.seed(42)
        # Create banana mask to constrain spots strictly inside the fruit
        banana_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(banana_mask, center, axes, angle, 0, 360, 255, -1)
        cv2.ellipse(banana_mask, (250, 165), (150, 55), angle, 0, 360, 0, -1)

        y_indices, x_indices = np.where(banana_mask > 0)
        if len(x_indices) > 0:
            random_idx = np.random.choice(len(x_indices), size=min(spots_count, len(x_indices)), replace=False)
            for idx in random_idx:
                spot_x, spot_y = x_indices[idx], y_indices[idx]
                radius = np.random.randint(2, 6)
                spot_color = (np.random.randint(20, 50), np.random.randint(35, 75), np.random.randint(55, 105))
                cv2.circle(img, (spot_x, spot_y), radius, spot_color, -1, cv2.LINE_AA)

    # Convert from OpenCV BGR to RGB for standard display & downstream processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


# -----------------------------------------------------------------------------
# COMPUTER VISION PIPELINE: SEGMENTATION & CONTOUR DETECTION
# -----------------------------------------------------------------------------
def segment_banana(image_rgb: np.ndarray):
    """
    Isolates the banana from the background using HSV color thresholding
    and contour detection.

    Why this matters:
    Without segmentation, a white plate, wooden cutting board, or colorful
    tablecloth could be counted as banana pixels, ruining the accuracy!

    Returns:
        banana_mask (np.ndarray): Binary mask where 255 = Banana, 0 = Background
        contour_annotated (np.ndarray): Original image with banana contours drawn
        status_msg (str): Informative message on segmentation outcome
    """
    # Step 1: Convert RGB image to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Step 2: Define candidate color threshold encompassing banana hues
    # Bananas range from green (Hue 35-85), yellow (Hue 15-35), to warm brown/amber (Hue 5-25).
    # Backgrounds like white/grey surfaces (low Saturation) or black surfaces (low Value) are excluded.
    lower_candidate = np.array([5, 25, 25], dtype=np.uint8)
    upper_candidate = np.array([88, 255, 255], dtype=np.uint8)
    candidate_mask = cv2.inRange(hsv, lower_candidate, upper_candidate)

    # Step 3: Clean up noise using Morphological Operations
    # Closing (Dilation then Erosion) bridges small gaps inside the banana
    # Opening (Erosion then Dilation) removes tiny background noise flecks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    cleaned_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Step 4: Find Contours in the binary mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image_rgb.shape[:2]
    total_pixels = h * w
    banana_mask = np.zeros((h, w), dtype=np.uint8)
    contour_annotated = image_rgb.copy()

    if contours:
        # Filter contours with reasonable area (> 0.8% of image area)
        min_area = total_pixels * 0.008
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        if valid_contours:
            # Sort by area descending and pick the largest contour as the banana
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Fill the detected contour to create a solid banana mask
            cv2.drawContours(banana_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Draw an attractive boundary around the banana for visual feedback
            cv2.drawContours(contour_annotated, [largest_contour], -1, (0, 255, 230), thickness=3)
            
            # Draw a bounding rectangle
            x, y, bw, bh = cv2.boundingRect(largest_contour)
            cv2.rectangle(contour_annotated, (x, y), (x + bw, y + bh), (255, 215, 0), 2)
            cv2.putText(contour_annotated, "BANANA DETECTED", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2, cv2.LINE_AA)

            area_pct = (cv2.contourArea(largest_contour) / total_pixels) * 100
            status_msg = f"✅ Banana successfully segmented via largest contour ({area_pct:.1f}% of frame)."
            return banana_mask, contour_annotated, status_msg

    # Fallback if no clean single contour was found
    if np.count_nonzero(cleaned_mask) > (total_pixels * 0.02):
        banana_mask = cleaned_mask
        status_msg = "⚠️ Fallback: Color-thresholded mask used (no single dominant contour boundary)."
    else:
        # Ultimate fallback: analyze entire image if background is uniform or colors differ
        banana_mask = np.full((h, w), 255, dtype=np.uint8)
        status_msg = "ℹ️ Standard fallback: Whole image analyzed (low contrast with background)."

    return banana_mask, contour_annotated, status_msg


# -----------------------------------------------------------------------------
# COMPUTER VISION PIPELINE: HSV COLOR QUANTIFICATION
# -----------------------------------------------------------------------------
def analyze_ripeness_colors(image_rgb: np.ndarray, banana_mask: np.ndarray):
    """
    Converts RGB image to HSV and detects Green, Yellow, and Brown pixels
    strictly within the segmented banana region.

    HSV Ranges in OpenCV:
        Hue: 0 to 179 (representing 0 to 360 degrees)
        Saturation: 0 to 255 (color purity/intensity)
        Value: 0 to 255 (brightness/luminance)
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # 1. GREEN RANGE (Unripe banana flesh & stem)
    # Hue: 35 - 85 covers yellowish-green to deep forest green
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    raw_green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2. YELLOW RANGE (Ripe, golden banana skin)
    # Hue: 18 - 35 covers warm golden yellow to bright lemon yellow
    lower_yellow = np.array([17, 45, 60], dtype=np.uint8)
    upper_yellow = np.array([34, 255, 255], dtype=np.uint8)
    raw_yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 3. BROWN & SUGAR SPOTS RANGE (Overripe necrosis & dark spotting)
    # Hue: 5 - 20 with lower brightness, or any very dark spots inside the banana
    lower_brown = np.array([6, 35, 20], dtype=np.uint8)
    upper_brown = np.array([22, 255, 140], dtype=np.uint8)
    mask_brown_hue = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spots (bruises, sugar spots, senescent spotting)
    lower_dark_spots = np.array([0, 0, 0], dtype=np.uint8)
    upper_dark_spots = np.array([180, 255, 60], dtype=np.uint8)
    mask_dark_spots = cv2.inRange(hsv, lower_dark_spots, upper_dark_spots)

    raw_brown_mask = cv2.bitwise_or(mask_brown_hue, mask_dark_spots)

    # Constrain all color masks strictly within the segmented banana area
    mask_brown = cv2.bitwise_and(raw_brown_mask, raw_brown_mask, mask=banana_mask)
    
    # Priority handling: Brown spots take priority over yellow
    raw_yellow_mask = cv2.bitwise_and(raw_yellow_mask, cv2.bitwise_not(raw_brown_mask))
    mask_yellow = cv2.bitwise_and(raw_yellow_mask, raw_yellow_mask, mask=banana_mask)
    
    # Green mask constrained
    raw_green_mask = cv2.bitwise_and(raw_green_mask, cv2.bitwise_not(raw_brown_mask))
    mask_green = cv2.bitwise_and(raw_green_mask, raw_green_mask, mask=banana_mask)

    # Count pixels
    green_pixels = int(np.count_nonzero(mask_green))
    yellow_pixels = int(np.count_nonzero(mask_yellow))
    brown_pixels = int(np.count_nonzero(mask_brown))
    total_classified = green_pixels + yellow_pixels + brown_pixels

    if total_classified == 0:
        # Fallback if banana colors weren't registered (e.g. non-banana image)
        return {
            "green_pct": 0.0,
            "yellow_pct": 0.0,
            "brown_pct": 0.0,
            "green_mask": mask_green,
            "yellow_mask": mask_yellow,
            "brown_mask": mask_brown,
            "total_pixels": 0,
            "valid": False
        }

    # Relative percentages normalized to 100%
    pct_green = round((green_pixels / total_classified) * 100.0, 1)
    pct_yellow = round((yellow_pixels / total_classified) * 100.0, 1)
    pct_brown = round((brown_pixels / total_classified) * 100.0, 1)

    return {
        "green_pct": pct_green,
        "yellow_pct": pct_yellow,
        "brown_pct": pct_brown,
        "green_mask": mask_green,
        "yellow_mask": mask_yellow,
        "brown_mask": mask_brown,
        "total_pixels": total_classified,
        "valid": True
    }


# -----------------------------------------------------------------------------
# RIPENESS SCORE & CLASSIFICATION LOGIC
# -----------------------------------------------------------------------------
def calculate_ripeness(pct_green: float, pct_yellow: float, pct_brown: float):
    """
    Computes a continuous Ripeness Score (0 - 100%) and categorizes the banana
    into one of 5 standard ripeness stages.

    Scoring Logic:
    - Green (Unripe): 0% ripeness contribution
    - Yellow (Sweet): 70% ripeness baseline contribution
    - Brown (Spotted / Overripe): 100% ripeness maximum contribution
    """
    # Calculate weighted maturity index
    raw_score = (pct_yellow * 0.70) + (pct_brown * 1.00)
    score = min(100.0, max(0.0, raw_score))

    # Stage Classification (Evaluated hierarchically)
    if pct_brown >= 38.0 or score >= 88.0:
        stage = "🚨 Extremely Ripe"
        badge_bg = "#d32f2f"
        is_emergency = True
        desc = "Structural integrity compromised. Massive sugar concentration."
        recommendation = (
            "🚨 BANANA EMERGENCY: Peel immediately! Do NOT discard. This is prime "
            "material for the ultimate Banana Bread, caramelized pancakes, or freeze "
            "in chunks for sweet baking later."
        )
    elif pct_brown >= 18.0 or score >= 75.0:
        stage = "🟤 Very Ripe"
        badge_bg = "#795548"
        is_emergency = False
        desc = "Prominent sugar spotting, intense banana fragrance, soft texture."
        recommendation = (
            "Rich and aromatic! Perfect for protein shakes, banana muffins, smoothies, "
            "or homemade banana nice-cream."
        )
    elif pct_green >= 40.0 or (pct_green > pct_yellow and score < 30.0):
        stage = "🟢 Unripe"
        badge_bg = "#2e7d32"
        is_emergency = False
        desc = "Firm, rich in dietary fiber and resistant starch, low sugar."
        recommendation = (
            "Too starchy and astringent for raw snacking. Great for savory cooking "
            "(green banana curry, savory chips), or leave on the kitchen counter for 3–4 days."
        )
    elif pct_green >= 15.0 or score < 55.0:
        stage = "🟡 Becoming Ripe"
        badge_bg = "#c0ca33"
        is_emergency = False
        desc = "Transition phase with green tips and emerging yellow body."
        recommendation = (
            "Firm with moderate sweetness and higher resistant starch. Ideal for "
            "slicing over hot oatmeal or wait 24–48 hours for full sweetness."
        )
    else:
        stage = "🍌 Perfectly Ripe"
        badge_bg = "#ffa000"
        is_emergency = False
        desc = "Peak golden color, firm yet creamy texture, optimal sweetness."
        recommendation = (
            "Peak snacking window! High in accessible potassium and balanced natural sugars. "
            "Enjoy fresh out of the peel or sliced into morning cereal."
        )

    return {
        "score": round(score, 1),
        "stage": stage,
        "badge_bg": badge_bg,
        "is_emergency": is_emergency,
        "desc": desc,
        "recommendation": recommendation
    }


# -----------------------------------------------------------------------------
# CHART GENERATOR
# -----------------------------------------------------------------------------
def render_color_distribution_chart(pct_green: float, pct_yellow: float, pct_brown: float):
    """
    Renders an Altair bar chart with customized banana color palette.
    """
    df = pd.DataFrame({
        "Color Component": ["🟢 Green (Unripe)", "🟡 Yellow (Ripe)", "🟤 Brown (Sugar Spots)"],
        "Percentage": [pct_green, pct_yellow, pct_brown],
        "ColorHex": ["#4CAF50", "#FFD700", "#795548"]
    })

    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Color Component:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Percentage:Q", title="Surface Area (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("ColorHex:N", scale=None),
        tooltip=["Color Component", alt.Tooltip("Percentage:Q", format=".1f")]
    ).properties(height=260)

    st.altair_chart(chart, use_container_width=True)


# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
def main():
    # Header Section
    st.title("🍌 BANANA MATURITY ANALYZER")
    st.caption("“Advanced Computer Vision for an Extremely Important Problem.”")
    st.markdown("---")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Analysis Controls")
        
        input_mode = st.radio(
            "Select Input Source:",
            ["📤 Upload My Own Image", "🧪 Test with Sample Bananas"],
            help="Choose to upload a banana photo or pick an instant generated sample."
        )
        
        uploaded_file = None
        selected_sample = None

        if input_mode == "📤 Upload My Own Image":
            uploaded_file = st.file_uploader(
                "Upload a Banana Image",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG. For best results, place banana on a contrasting surface."
            )
        else:
            selected_sample = st.selectbox(
                "Choose a Synthetic Banana Sample:",
                [
                    "🍌 Perfectly Ripe",
                    "🟢 Unripe Banana",
                    "🟡 Becoming Ripe",
                    "🟤 Very Ripe",
                    "🚨 Extremely Ripe"
                ]
            )
            st.info("💡 Tip: Synthetic samples simulate real banana color distribution and test the CV engine instantly!")

        st.markdown("---")
        st.subheader("🔬 CV Diagnostics Toggle")
        show_masks = st.checkbox("Show Advanced Computer Vision Masks", value=True,
                                 help="Inspect segmented contours, HSV channels, and color isolation masks.")
        
        st.markdown("---")
        st.markdown("### 📚 About the Project")
        st.markdown(
            "Built with **Python**, **OpenCV**, and **Streamlit** to demonstrate beginner-friendly "
            "image processing: color space conversion (RGB → HSV), morphological filtering, "
            "contour segmentation, and feature extraction."
        )

    # Ingestion & Loading
    image_rgb = None

    if input_mode == "📤 Upload My Own Image":
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(file_bytes))
                image_rgb = np.array(pil_image.convert("RGB"))
            except Exception as e:
                st.error(f"❌ Error loading uploaded image: {str(e)}. Please try another valid image.")
                return
        else:
            # Friendly landing prompt when no image is uploaded
            st.info("👋 Welcome! Please upload a banana photo via the sidebar or select **'🧪 Test with Sample Bananas'** to begin analysis.")
            
            # Quick instructions card
            st.markdown("""
            <div class="card-box">
                <h4>🎯 How It Works:</h4>
                <ol>
                    <li><b>Segmentation:</b> OpenCV filters out table/background pixels using morphological operations and external contours.</li>
                    <li><b>HSV Conversion:</b> The image is converted to HSV to separate chromatic tones from light intensity.</li>
                    <li><b>Quantification:</b> Green, Yellow, and Brown pixel ratios are calculated.</li>
                    <li><b>Score & Recommendation:</b> A 0–100% ripeness score determines the optimal culinary purpose!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            return
    else:
        # Load synthetic sample
        image_rgb = generate_sample_banana_image(selected_sample)

    # -------------------------------------------------------------------------
    # COMPUTER VISION PROCESSING EXECUTION
    # -------------------------------------------------------------------------
    with st.spinner("🔍 Segmenting fruit and computing HSV color histograms..."):
        # 1. Segment banana from background
        banana_mask, contour_viz, status_msg = segment_banana(image_rgb)
        
        # 2. Analyze color distribution
        color_data = analyze_ripeness_colors(image_rgb, banana_mask)

    if not color_data["valid"]:
        st.warning(
            "⚠️ No banana-like colors detected in the image! "
            "Please ensure the image contains a clearly visible banana with adequate lighting."
        )
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
        return

    # 3. Calculate ripeness score and stage
    ripeness = calculate_ripeness(
        color_data["green_pct"],
        color_data["yellow_pct"],
        color_data["brown_pct"]
    )

    # -------------------------------------------------------------------------
    # EMERGENCY BANNER IF EXTREMELY RIPE
    # -------------------------------------------------------------------------
    if ripeness["is_emergency"]:
        st.markdown(f"""
        <div class="emergency-banner">
            <div class="emergency-title">🚨🚨 BANANA EMERGENCY DECLARED! 🚨🚨</div>
            <p style="font-size: 1.05rem; margin-bottom: 8px;">
                <b>CRITICAL ALERT:</b> Brown pigmentation has reached <b>{color_data['brown_pct']}%</b>!
                Structural failure is imminent. Immediate intervention required to prevent waste.
            </p>
            <div style="background: rgba(0, 0, 0, 0.25); padding: 10px 14px; border-radius: 8px;">
                <b>👨‍🍳 Emergency Protocol:</b> Preheat oven to 350°F (175°C) immediately. Procure flour, butter, 
                and brown sugar. Deploy into Banana Bread or freeze into chunks within 4 hours.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN RESULTS DASHBOARD
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns([1, 1.2], gap="large")

    # LEFT COLUMN: Visual Inspection
    with col_left:
        st.subheader("📷 Visual Input & Segmentation")
        
        tab_original, tab_contour = st.tabs(["Original Image", "Contour Detection"])
        with tab_original:
            st.image(image_rgb, caption="Input Frame", use_container_width=True)
        with tab_contour:
            st.image(contour_viz, caption="OpenCV Isolated Banana Boundary", use_container_width=True)
            st.caption(status_msg)

    # RIGHT COLUMN: Metrics & Diagnostics
    with col_right:
        st.subheader("📊 Maturity Assessment")
        
        # Stage & Score Header
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <span class="status-badge" style="background-color: {ripeness['badge_bg']}; color: white;">
                {ripeness['stage']}
            </span>
            <span style="font-weight: 700; font-size: 1.1rem;">
                Maturity Index: {ripeness['score']}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Ripeness Progress Bar
        st.progress(int(ripeness["score"]))
        st.caption(f"Stage Description: {ripeness['desc']}")

        # 3-Column Color Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Green (Firmness)", f"{color_data['green_pct']}%")
        m2.metric("🟡 Yellow (Sweetness)", f"{color_data['yellow_pct']}%")
        m3.metric("🟤 Brown (Sugar Spots)", f"{color_data['brown_pct']}%")

        # Color Distribution Chart
        st.markdown("##### 📈 Color Surface Area Distribution")
        render_color_distribution_chart(
            color_data["green_pct"],
            color_data["yellow_pct"],
            color_data["brown_pct"]
        )

    # -------------------------------------------------------------------------
    # RECOMMENDATION CARD
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💡 Culinary Recommendation")
    st.info(ripeness["recommendation"])

    # -------------------------------------------------------------------------
    # ADVANCED CV DIAGNOSTICS & MASKS (EDUCATIONAL SECTION)
    # -------------------------------------------------------------------------
    if show_masks:
        with st.expander("🔍 Advanced Computer Vision Pipeline Breakdown (Educational)", expanded=False):
            st.markdown("""
            Here is how OpenCV processed your banana behind the scenes:
            1. **Banana Mask**: Filtered out background pixels via morphological operations and external contour search.
            2. **Green Mask**: Pixels within Hue $[35, 85]$ (indicates unripeness & starch content).
            3. **Yellow Mask**: Pixels within Hue $[17, 34]$ (indicates high carotenoids and sweet sucrose).
            4. **Brown / Spots Mask**: Pixels with low Value/Luminance or Hue $[6, 22]$ (indicates sugar spotting).
            """)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.image(banana_mask, caption="1. Banana Isolation Mask", use_container_width=True)
            c2.image(color_data["green_mask"], caption="2. Green Pixels Detected", use_container_width=True)
            c3.image(color_data["yellow_mask"], caption="3. Yellow Pixels Detected", use_container_width=True)
            c4.image(color_data["brown_mask"], caption="4. Brown Pixels Detected", use_container_width=True)


if __name__ == "__main__":
    main()
