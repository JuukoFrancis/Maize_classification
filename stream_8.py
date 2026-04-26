"""
================================================================
  Maize Leaf Disease Classifier — Ensemble Streamlit App
  Models: ResNet50 + MobileNetV2 + EfficientNetB0
  Method: Softmax-Weighted Averaging (auto-calculated)
  Model Storage: Hugging Face Bucket (JuukoFrancis/Maize_models)
================================================================
  Run:
      pip install streamlit tensorflow pillow scikit-learn matplotlib seaborn requests
      streamlit run app.py
================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.backends.backend_pdf import PdfPages
import io
import requests

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Maize Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  HEADING HELPERS
# ══════════════════════════════════════════════════════════════
HC = "#111827"

def h1(text):
    st.markdown(f"<h1 style='font-family:Playfair Display,serif;color:{HC};margin-bottom:0.2rem;'>{text}</h1>", unsafe_allow_html=True)

def h2(text):
    st.markdown(f"<h2 style='font-family:Playfair Display,serif;color:{HC};margin-top:1.2rem;margin-bottom:0.2rem;'>{text}</h2>", unsafe_allow_html=True)

def h3(text):
    st.markdown(f"<h3 style='font-family:Playfair Display,serif;color:{HC};margin-top:0.8rem;margin-bottom:0.2rem;'>{text}</h3>", unsafe_allow_html=True)

def h4(text):
    st.markdown(f"<h4 style='font-family:DM Sans,sans-serif;font-weight:700;color:{HC};margin-top:0.8rem;margin-bottom:0.1rem;'>{text}</h4>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; background-color: #faf8f4 !important; }
h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 { font-family: 'Playfair Display', serif !important; color: #111827 !important; }
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
section[data-testid="stSidebar"] > div > div > div { background-color: #111827 !important; }
section[data-testid="stSidebar"] * { color: #f9fafb !important; }
section[data-testid="stSidebar"] .stButton > button {
    background-color: #16a34a !important; color: #ffffff !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1rem !important;
    font-weight: 600 !important; width: 100% !important; transition: background 0.2s !important; }
section[data-testid="stSidebar"] .stButton > button:hover { background-color: #15803d !important; }
section[data-testid="stSidebar"] hr { border-color: #374151 !important; }
.stApp { background-color: #faf8f4 !important; }
[data-testid="stMetric"] { background: #ffffff !important; border-radius: 12px !important; padding: 1rem !important; box-shadow: 0 1px 8px rgba(0,0,0,0.06) !important; }
[data-testid="stMetricLabel"] p,[data-testid="stMetricValue"],[data-testid="stMetricDelta"] { color: #111827 !important; }
[data-testid="stProgress"] > div > div { background-color: #16a34a !important; border-radius: 99px !important; }
[data-testid="stFileUploader"] { border: 2px dashed #86efac !important; border-radius: 16px !important; background: #f0fdf4 !important; }
[data-testid="stExpander"] { border: 1px solid #d1fae5 !important; border-radius: 12px !important; background: #f9fafb !important; }
[data-testid="stExpander"] summary p { color: #111827 !important; font-weight: 600 !important; }
[data-testid="stCaptionContainer"] p { color: #6b7280 !important; }
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #16a34a !important; color: #ffffff !important; border: none !important;
    border-radius: 12px !important; padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important; font-size: 1rem !important; width: 100% !important;
    margin-top: 0.75rem !important; transition: background 0.2s !important; }
div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #15803d !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
VAL_ACCURACIES = {
    "ResNet50"      : 0.90,
    "MobileNetV2"   : 0.88,
    "EfficientNetB0": 0.92,
}

CLASS_NAMES = ["Healthy", "MLB (Maize Leaf Blight)", "MSV (Maize Streak Virus)"]
CLASS_SHORT = ["Healthy", "MLB", "MSV"]
IMG_SIZE    = (224, 224)
TEST_DIR    = "sample"

# ══════════════════════════════════════════════════════════════
#  HUGGING FACE BUCKET URLS
#  Bucket: JuukoFrancis/Maize_models
#  These are direct CDN download links — no quota limits, no confirmation pages
# ══════════════════════════════════════════════════════════════
HF_BUCKET_BASE = "https://huggingface.co/datasets/JuukoFrancis/Maize_models/resolve/main"

MODEL_FILES = {
    "ResNet50"      : "resnet50_best_phase2.keras",
    "MobileNetV2"   : "mobilenetv2_maize_model.h5",
    "EfficientNetB0": "best_phase2.keras",
}

MODEL_PATHS = {name: f"models/{fname}" for name, fname in MODEL_FILES.items()}

PREP_NOTES = {
    "ResNet50"      : "ResNet50 mean-subtraction (`keras.applications.resnet50.preprocess_input`)",
    "MobileNetV2"   : "MobileNetV2 [-1, 1] scaling (`keras.applications.mobilenet_v2.preprocess_input`)",
    "EfficientNetB0": "No rescale — EfficientNetB0 handles preprocessing internally",
}

# ══════════════════════════════════════════════════════════════
#  MODEL DOWNLOADER — Hugging Face Bucket (direct HTTP, no gdown needed)
# ══════════════════════════════════════════════════════════════
def get_hf_headers():
    """Return auth header if HF_TOKEN is set (required for private buckets)."""
    token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_model(name: str, path: str) -> bool:
    url = f"{HF_BUCKET_BASE}/{MODEL_FILES[name]}"
    try:
        with requests.get(url, headers=get_hf_headers(), stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        # Reject tiny files (HTML error pages served instead of model)
        if os.path.getsize(path) < 1_000_000:
            os.remove(path)
            return False
        return True
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        st.error(f"❌ **{name}** download error: {e}")
        return False


def ensure_models_downloaded():
    """Download missing models from HF Bucket. Cached files are reused."""
    os.makedirs("models", exist_ok=True)
    status = {}

    for name, path in MODEL_PATHS.items():
        # Already cached and valid?
        if os.path.exists(path) and os.path.getsize(path) >= 1_000_000:
            status[name] = True
            continue

        # Stale / corrupt file — delete and re-download
        if os.path.exists(path):
            os.remove(path)

        banner = st.info(f"⬇️ Downloading **{name}** from Hugging Face Bucket…", icon="☁️")
        ok = download_model(name, path)
        if ok:
            mb = os.path.getsize(path) / 1_000_000
            banner.success(f"✅ **{name}** ready — {mb:.1f} MB")
        else:
            banner.error(
                f"❌ **{name}** failed. Make sure your bucket is **public** "
                f"or add `HF_TOKEN` to your Streamlit secrets.")
        status[name] = ok

    return status


download_status = ensure_models_downloaded()

# ══════════════════════════════════════════════════════════════
#  SOFTMAX WEIGHTS
# ══════════════════════════════════════════════════════════════
def softmax_weights(acc_dict, temperature=5.0):
    names = list(acc_dict.keys())
    accs  = np.array([acc_dict[n] for n in names], dtype=float)
    scaled = accs * temperature
    exp    = np.exp(scaled - scaled.max())
    return {n: float(p) for n, p in zip(names, exp / exp.sum())}

AUTO_WEIGHTS = softmax_weights(VAL_ACCURACIES)

# ══════════════════════════════════════════════════════════════
#  FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════
def build_diagnostic_figures(results_map):
    figs, BG, HC_ = [], "#faf8f4", "#111827"

    for model_name, results in results_map.items():
        if results is None:
            continue

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor(BG); ax_cm.set_facecolor(BG)
        sns.heatmap(results["cm"], annot=True, fmt="d", cmap="Greens",
                    xticklabels=results["labels"], yticklabels=results["labels"],
                    linewidths=0.5, ax=ax_cm)
        ax_cm.set_title(f"{model_name} — Confusion Matrix", fontsize=11, fontweight="bold", color=HC_)
        ax_cm.set_xlabel("Predicted", color=HC_); ax_cm.set_ylabel("True", color=HC_)
        ax_cm.tick_params(colors=HC_); plt.tight_layout()
        figs.append((f"{model_name} — Confusion Matrix", fig_cm))

        per_class_acc = results["cm"].diagonal() / results["cm"].sum(axis=1)
        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
        fig_bar.patch.set_facecolor(BG); ax_bar.set_facecolor(BG)
        bars = ax_bar.bar(results["labels"], per_class_acc * 100,
                          color=["#16a34a", "#4ade80", "#86efac"], edgecolor="white", linewidth=0.8)
        ax_bar.set_ylim(0, 115); ax_bar.set_ylabel("Accuracy (%)", color=HC_)
        ax_bar.set_title(f"{model_name} — Per-Class Accuracy", fontsize=11, fontweight="bold", color=HC_)
        ax_bar.tick_params(colors=HC_)
        for bar, a in zip(bars, per_class_acc):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{a*100:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=HC_)
        ax_bar.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        figs.append((f"{model_name} — Per-Class Accuracy", fig_bar))

    comp_labels = [n for n, r in results_map.items() if r is not None]
    comp_values = [r["acc"] * 100 for r in results_map.values() if r is not None]
    if comp_values:
        fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
        fig_cmp.patch.set_facecolor(BG); ax_cmp.set_facecolor(BG)
        bars3 = ax_cmp.bar(comp_labels, comp_values,
                           color=["#16a34a", "#4ade80", "#86efac"], edgecolor="white", linewidth=0.8)
        ax_cmp.set_ylim(0, 115); ax_cmp.set_ylabel("Test Accuracy (%)", color=HC_)
        ax_cmp.set_title("Model Test Accuracy Comparison", fontsize=12, fontweight="bold", color=HC_)
        ax_cmp.tick_params(colors=HC_)
        for bar, val in zip(bars3, comp_values):
            ax_cmp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=HC_)
        ax_cmp.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        figs.append(("Model Comparison — Test Accuracy", fig_cmp))

    return figs


def export_figures_to_buffer(figs, fmt):
    buf = io.BytesIO()
    if fmt == "pdf":
        with PdfPages(buf) as pdf:
            for _, fig in figs:
                pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
        mime, ext = "application/pdf", "pdf"
    else:
        n = len(figs)
        combined, axes = plt.subplots(n, 1, figsize=(8, 5 * n))
        if n == 1: axes = [axes]
        combined.patch.set_facecolor("#faf8f4")
        for ax_slot, (title, src_fig) in zip(axes, figs):
            tmp = io.BytesIO()
            src_fig.savefig(tmp, format="png", bbox_inches="tight", facecolor=src_fig.get_facecolor())
            tmp.seek(0)
            from PIL import Image as PILImage
            ax_slot.imshow(PILImage.open(tmp)); ax_slot.axis("off")
            ax_slot.set_title(title, fontsize=10, fontweight="bold", color="#111827", pad=6)
        plt.tight_layout(pad=1.5)
        mime = "image/jpeg" if fmt == "jpeg" else "image/png"
        ext  = "jpg" if fmt == "jpeg" else "png"
        combined.savefig(buf, format="jpeg" if fmt=="jpeg" else "png",
                         bbox_inches="tight", facecolor="#faf8f4", dpi=150)
        plt.close(combined)
    buf.seek(0)
    return buf.read(), mime, ext

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models…")
def load_all_models():
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                loaded[name] = tf.keras.models.load_model(path)
            except Exception as e:
                st.warning(f"⚠️ Could not load **{name}**: {e}")
                loaded[name] = None
        else:
            loaded[name] = None
    return loaded

models_dict = load_all_models()

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
def get_datagen(model_name):
    if model_name == "ResNet50":    return ImageDataGenerator(preprocessing_function=resnet_preprocess)
    if model_name == "MobileNetV2": return ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
    return ImageDataGenerator()

def preprocess_for_model(pil_img, model_name):
    arr = img_to_array(pil_img.convert("RGB").resize(IMG_SIZE))
    if model_name == "ResNet50":    arr = resnet_preprocess(arr)
    if model_name == "MobileNetV2": arr = mobilenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)

def predict_single(pil_img, model_name, model):
    return model.predict(preprocess_for_model(pil_img, model_name), verbose=0)[0]

# ══════════════════════════════════════════════════════════════
#  ENSEMBLE PREDICTION
# ══════════════════════════════════════════════════════════════
def ensemble_predict(pil_img, weights):
    final_probs, per_model_probs = np.zeros(len(CLASS_NAMES)), {}
    total_w = sum(weights[n] for n in weights if models_dict.get(n) is not None)
    for name, model in models_dict.items():
        if model is None: continue
        probs = predict_single(pil_img, name, model)
        per_model_probs[name] = probs
        final_probs += (weights[name] / total_w) * probs
    return final_probs, per_model_probs

# ══════════════════════════════════════════════════════════════
#  TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Evaluating on test set…")
def evaluate_model_on_test(model_name):
    model = models_dict.get(model_name)
    if model is None or not os.path.exists(TEST_DIR): return None
    gen = get_datagen(model_name).flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=32, class_mode="categorical", shuffle=False)
    gen.reset()
    preds  = model.predict(gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = gen.classes[:len(y_pred)]
    labels = list(gen.class_indices.keys())
    return {"y_true": y_true, "y_pred": y_pred, "labels": labels,
            "cm": confusion_matrix(y_true, y_pred), "acc": np.mean(y_pred == y_true),
            "report": classification_report(y_true, y_pred, target_names=labels,
                                            output_dict=True, zero_division=0)}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 style='color:#f9fafb;font-family:Playfair Display,serif;'>🌽 Maize Classifier</h2>",
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#374151;'>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#f9fafb;font-family:Playfair Display,serif;'>☁️ Model Files</h3>",
                unsafe_allow_html=True)
    st.markdown("<small style='color:#9ca3af;'>HF Bucket · JuukoFrancis/Maize_models</small>",
                unsafe_allow_html=True)

    for name, path in MODEL_PATHS.items():
        loaded = models_dict.get(name) is not None
        cached = os.path.exists(path)
        if loaded:
            icon, label, color = "🟢", "Loaded", "#4ade80"
        elif cached:
            icon, label, color = "🟡", "Cached (load failed)", "#fbbf24"
        else:
            icon, label, color = "🔴", "Not downloaded", "#f87171"
        size_str = f" · {os.path.getsize(path)/1_000_000:.0f} MB" if cached else ""
        st.markdown(
            f"<div style='margin-bottom:8px;'>"
            f"<span style='color:{color};font-weight:600;'>{icon} {name}</span><br>"
            f"<span style='color:#9ca3af;font-size:0.78rem;'>{label}{size_str}</span>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#374151;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#f9fafb;font-family:Playfair Display,serif;'>⚖️ Ensemble Weights</h3>",
                unsafe_allow_html=True)
    st.caption("Auto-calculated via softmax over validation accuracies.")

    for name, w in AUTO_WEIGHTS.items():
        acc = VAL_ACCURACIES[name]
        st.markdown(f"""
        <div style="background:#1f2937;border-radius:10px;padding:10px 14px;margin-bottom:8px;">
          <div style="font-weight:700;font-size:0.95rem;color:#f9fafb;">{name}</div>
          <div style="display:flex;justify-content:space-between;margin-top:4px;">
            <span style="font-size:0.8rem;color:#9ca3af;">Val Acc: {acc*100:.0f}%</span>
            <span style="font-size:0.9rem;font-weight:700;color:#4ade80;">Weight: {w:.3f}</span>
          </div>
          <div style="background:#374151;border-radius:99px;height:6px;margin-top:6px;">
            <div style="background:#16a34a;border-radius:99px;height:6px;width:{w*100:.1f}%;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#374151;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#f9fafb;font-family:Playfair Display,serif;'>Model Diagnostics</h3>",
                unsafe_allow_html=True)
    show_diagnostics = st.button("📊 Run Diagnostics")

# ══════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════
h1("🌽 Maize Leaf Disease Classifier")
st.markdown(
    f"<p style='color:#6b7280;font-size:1.05rem;margin-top:-0.3rem;'>"
    f"Ensemble model · <strong style='color:{HC};'>ResNet50 + MobileNetV2 + EfficientNetB0</strong>"
    f" · Softmax-Weighted Averaging</p>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  DIAGNOSTICS PANEL
# ══════════════════════════════════════════════════════════════
if show_diagnostics:
    h2("📊 Individual Model Diagnostics")
    st.info(f"Running evaluation on test set: `{TEST_DIR}/`")
    if not os.path.exists(TEST_DIR):
        st.error(f"Test directory `{TEST_DIR}` not found.")
    else:
        results_map = {n: evaluate_model_on_test(n) for n in MODEL_PATHS}

        for model_name, results in results_map.items():
            st.markdown(
                f"<h3 style='font-family:Playfair Display,serif;color:{HC};"
                f"border-left:4px solid #16a34a;padding-left:0.65rem;margin-top:1.6rem;'>"
                f"{model_name}</h3>", unsafe_allow_html=True)
            if results is None:
                st.warning(f"Could not evaluate {model_name} — model not loaded."); continue
            st.caption(f"🔧 {PREP_NOTES[model_name]}")
            r = results["report"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{results['acc']*100:.2f}%")
            c2.metric("Precision", f"{r['weighted avg']['precision']*100:.2f}%")
            c3.metric("Recall",    f"{r['weighted avg']['recall']*100:.2f}%")
            c4.metric("F1-Score",  f"{r['weighted avg']['f1-score']*100:.2f}%")
            ch1, ch2 = st.columns(2)
            figs_m = build_diagnostic_figures({model_name: results})
            with ch1: st.pyplot(figs_m[0][1])
            with ch2: st.pyplot(figs_m[1][1])
            for _, f in figs_m: plt.close(f)
            st.markdown("<hr style='border-color:#e5e7eb;'>", unsafe_allow_html=True)

        h3("📈 Model Comparison — Test Accuracy")
        all_figs = build_diagnostic_figures(results_map)
        if all_figs:
            st.pyplot(all_figs[-1][1])
            for _, f in all_figs: plt.close(f)

        st.markdown("<hr style='border-color:#e5e7eb;'>", unsafe_allow_html=True)
        h3("⬇️ Download All Charts")
        fc, bc = st.columns([1, 2], gap="medium")
        with fc:
            chosen_fmt = st.selectbox("fmt", ["pdf", "png", "jpeg"],
                format_func=lambda x: {"pdf":"📄 PDF","png":"🖼️ PNG","jpeg":"📷 JPEG"}[x],
                label_visibility="collapsed")
        with bc:
            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
            if st.button("📦 Prepare download", type="primary", use_container_width=True):
                with st.spinner(f"Rendering as {chosen_fmt.upper()}…"):
                    ef = build_diagnostic_figures(results_map)
                    fb, mt, fe = export_figures_to_buffer(ef, chosen_fmt)
                    for _, f in ef: plt.close(f)
                st.download_button(f"⬇️ Download maize_diagnostics.{fe}",
                                   data=fb, file_name=f"maize_diagnostics.{fe}",
                                   mime=mt, use_container_width=True)
    st.markdown("<hr style='border-color:#e5e7eb;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  IMAGE UPLOAD & PREDICTION
# ══════════════════════════════════════════════════════════════
h2("🔍 Classify a Leaf Image")
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    uploaded = st.file_uploader("Upload a maize leaf image", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
        classify_clicked = st.button("Classify Leaf 🌿", type="primary", use_container_width=True)
    else:
        classify_clicked = False

with col_result:
    if uploaded and classify_clicked:
        with st.spinner("Running ensemble prediction…"):
            final_probs, per_model_probs = ensemble_predict(pil_img, AUTO_WEIGHTS)

        pred_idx, pred_label = np.argmax(final_probs), CLASS_NAMES[np.argmax(final_probs)]
        confidence = final_probs[pred_idx] * 100
        badge_bg, badge_fg = ("#dcfce7","#166534") if confidence>=80 else \
                             (("#fef9c3","#713f12") if confidence>=60 else ("#fee2e2","#7f1d1d"))
        icon = "🟢" if pred_idx == 0 else "🔴"

        st.markdown(f"""
        <div style="background:white;border-radius:18px;padding:2rem;
                    box-shadow:0 2px 24px rgba(0,0,0,0.07);text-align:center;margin-top:0.5rem;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">{icon}</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.9rem;color:{HC};margin-bottom:0.4rem;">{pred_label}</div>
            <span style="display:inline-block;background:{badge_bg};color:{badge_fg};
                         padding:0.35rem 1.4rem;border-radius:99px;font-size:1rem;font-weight:600;">
                {confidence:.1f}% confidence</span>
        </div>""", unsafe_allow_html=True)

        h4("Ensemble Probabilities")
        for cls, prob in zip(CLASS_SHORT, final_probs):
            st.markdown(f"<span style='color:{HC};font-weight:600;'>{cls}</span>", unsafe_allow_html=True)
            st.progress(float(prob), text=f"{prob*100:.1f}%")

        h4("⚖️ Weights Used in This Prediction")
        for col, (name, w) in zip(st.columns(len(AUTO_WEIGHTS)), AUTO_WEIGHTS.items()):
            with col:
                st.markdown(f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:0.9rem;text-align:center;">
                    <div style="font-weight:700;font-size:0.82rem;color:{HC};margin-bottom:4px;">{name}</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#16a34a;">{w:.3f}</div>
                    <div style="font-size:0.75rem;color:#6b7280;margin-top:2px;">Val acc: {VAL_ACCURACIES[name]*100:.0f}%</div>
                    <div style="background:#dcfce7;border-radius:99px;height:5px;margin-top:8px;">
                        <div style="background:#16a34a;border-radius:99px;height:5px;width:{int(w*100)}%;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with st.expander("🔬 Per-model breakdown"):
            for col, (name, probs) in zip(st.columns(len(per_model_probs)), per_model_probs.items()):
                with col:
                    st.markdown(f"<span style='font-weight:700;color:{HC};'>{name}</span>", unsafe_allow_html=True)
                    st.markdown(f"→ **{CLASS_SHORT[np.argmax(probs)]}** ({np.max(probs)*100:.1f}%)")
                    for cls, p in zip(CLASS_SHORT, probs): st.caption(f"{cls}: {p*100:.1f}%")

        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        fig_pie.patch.set_facecolor("#faf8f4")
        wedges, texts, autotexts = ax_pie.pie(
            final_probs, labels=CLASS_SHORT, autopct="%1.1f%%",
            colors=["#16a34a","#dc2626","#f59e0b"], startangle=140, textprops={"fontsize":10})
        for t in texts + autotexts: t.set_color(HC)
        ax_pie.set_title("Class Probability Distribution", fontsize=11, fontweight="bold", color=HC)
        plt.tight_layout(); st.pyplot(fig_pie); plt.close(fig_pie)

    elif uploaded and not classify_clicked:
        st.markdown("""
        <div style="background:white;border:2px dashed #86efac;border-radius:18px;
                    padding:5rem 2rem;text-align:center;margin-top:1rem;">
            <div style="font-size:3rem;">👆</div>
            <div style="font-size:1.1rem;margin-top:1rem;font-weight:500;color:#374151;">
                Image ready! Press <strong>Classify Leaf 🌿</strong> to run the ensemble.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:white;border:2px dashed #86efac;border-radius:18px;
                    padding:5rem 2rem;text-align:center;color:#9ca3af;margin-top:1rem;">
            <div style="font-size:3rem;">🌿</div>
            <div style="font-size:1.1rem;margin-top:1rem;font-weight:500;">
                Upload a maize leaf image to get started
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.85rem;'>"
    "Maize Leaf Disease Classifier · Ensemble: ResNet50 + MobileNetV2 + EfficientNetB0 · "
    "Models hosted on Hugging Face · Weights auto-calculated via softmax over validation accuracies"
    "</div>", unsafe_allow_html=True)