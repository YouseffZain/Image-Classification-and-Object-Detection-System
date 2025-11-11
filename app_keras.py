# app_keras.py - Streamlit demo for Chest X-ray Classification (Keras/TensorFlow)
import io, time, tempfile, numpy as np, zipfile, os, shutil
from typing import Tuple
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

st.set_page_config(page_title="Chest X-ray Classifier (Keras)", layout="centered")
st.title("Chest X-ray Classification - Keras Demo App")
st.markdown("Upload a Keras model (.keras / .h5) or a zipped SavedModel, then upload an image.\n\n"
            "**Disclaimer:** Educational demo only; not for medical use.")

with st.sidebar:
    st.header("Settings")
    default_labels = "Cardiomegaly,Pneumonia,Sick,healthy,tuberculosis"
    label_text = st.text_input("Class labels (comma-separated)", value=default_labels)
    class_names = [c.strip() for c in label_text.split(",") if c.strip()]
    num_classes_sidebar = len(class_names) if class_names else 2
    img_size = st.number_input("Input image size", min_value=64, max_value=1024, value=224, step=32)
    normalize_choice = st.selectbox("Normalization", ["Keras DenseNet preprocess_input", "Zero-Mean/Unit-Std (per-image)"])
    saliency_on = st.checkbox("Show saliency map (simple)", value=True)
    use_tta = st.checkbox("Use TTA (horizontal flip)", value=False)
    st.divider()
    # Accept ANY file type to avoid browser filtering issues
    model_file = st.file_uploader("Upload model (.keras / .h5 / zipped SavedModel)", type=None, accept_multiple_files=False)
    weights_url = st.text_input("Model URL (optional; direct link to .keras/.h5 or zipped SavedModel)")
    topk = st.slider("Top-k to display", 1, 10, min(5, num_classes_sidebar) if num_classes_sidebar>=5 else num_classes_sidebar)

def pil_from_dicom(file) -> Image.Image:
    ds = pydicom.dcmread(file)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    arr = (arr * 255.0).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

@st.cache_resource(show_spinner=False)
def demo_model(num_classes: int, input_shape: Tuple[int,int,int]=(224,224,3)) -> keras.Model:
    base = DenseNet121(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
    out = layers.Dense(num_classes, activation="softmax")(base.output)
    return keras.Model(base.input, out)

def preprocess_pil(pil_img: Image.Image, size: int, mode: str) -> np.ndarray:
    img = ensure_rgb(pil_img).resize((size, size))
    x = np.array(img).astype("float32")
    if mode.startswith("Keras DenseNet"):
        x = preprocess_input(x)
    else:
        x = x / 255.0
        mean = x.mean(axis=(0,1), keepdims=True)
        std = x.std(axis=(0,1), keepdims=True) + 1e-6
        x = (x - mean) / std
    return np.expand_dims(x, axis=0)

def predict(model: keras.Model, pil_img: Image.Image, img_size: int, normalize_choice: str, tta: bool=False):
    x = preprocess_pil(pil_img, img_size, normalize_choice)
    start = time.perf_counter()
    if tta:
        x_flip = x[:, :, ::-1, :]
        preds = model.predict(x, verbose=0)
        preds_flip = model.predict(x_flip, verbose=0)
        probs = (preds + preds_flip) / 2.0
    else:
        probs = model.predict(x, verbose=0)
    elapsed = time.perf_counter() - start
    return probs[0], elapsed, x

def _bytes_signature(b: bytes) -> str:
    if b.startswith(b"\x89HDF\r\n\x1a\n"):  # HDF5
        return "hdf5"
    if b.startswith(b"PK\x03\x04"):  # ZIP
        return "zip"
    return "unknown"

def _extract_zip_to_temp(upload_bytes: bytes) -> str:
    tmpdir = tempfile.mkdtemp(prefix="savedmodel_")
    zpath = os.path.join(tmpdir, "model.zip")
    with open(zpath, "wb") as f:
        f.write(upload_bytes)
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(tmpdir)
    # Heuristic: if a single top-level folder exists, use it
    entries = [os.path.join(tmpdir, e) for e in os.listdir(tmpdir) if e != "model.zip"]
    dirs = [e for e in entries if os.path.isdir(e)]
    if len(dirs) == 1:
        return dirs[0]
    return tmpdir

def load_model_from_upload(upload_bytes: bytes):
    sig = _bytes_signature(upload_bytes[:8])
    try:
        if sig == "hdf5":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(upload_bytes); tmp.flush()
                path = tmp.name
            model = keras.models.load_model(path, compile=False)
            return model, "Loaded .h5 model."
        elif sig == "zip":
            folder = _extract_zip_to_temp(upload_bytes)
            model = keras.models.load_model(folder, compile=False)
            return model, "Loaded zipped SavedModel."
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(upload_bytes); tmp.flush()
                path = tmp.name
            model = keras.models.load_model(path, compile=False)
            return model, "Loaded .keras model."
    except Exception as e:
        return demo_model(num_classes_sidebar, input_shape=(img_size, img_size, 3)), f"Failed to load: {e}. Using demo model."

def load_model_from_url(url: str):
    try:
        import requests
        r = requests.get(url, timeout=120); r.raise_for_status()
        return load_model_from_upload(r.content)
    except Exception as e:
        return demo_model(num_classes_sidebar, input_shape=(img_size, img_size, 3)), f"Failed to fetch URL: {e}. Using demo model."

uploaded = st.file_uploader("Upload Chest X-ray image", type=["png","jpg","jpeg","dcm"])
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".dcm"):
            if not _HAS_PYDICOM:
                st.error("pydicom not installed. Upload PNG/JPG instead."); st.stop()
            pil_img = pil_from_dicom(uploaded)
        else:
            pil_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read image: {e}"); st.stop()

    st.image(pil_img, caption="Input image", use_container_width=True)

    model = demo_model(num_classes_sidebar, input_shape=(img_size, img_size, 3))
    load_msg = "Demo model (random weights)."
    if model_file is not None:
        model, load_msg = load_model_from_upload(model_file.read())
    elif weights_url:
        model, load_msg = load_model_from_url(weights_url)

    with st.expander("Model load status", expanded=False):
        st.write(load_msg)
        st.write(f"Output classes detected: {model.output_shape[-1]}")

    probs, elapsed, x_tensor = predict(model, pil_img, img_size, normalize_choice, tta=use_tta)
    k = model.output_shape[-1]
    names = class_names if len(class_names) == k else [f"class_{i}" for i in range(k)]

    st.subheader("Prediction")
    top_indices = np.argsort(probs)[::-1][:min(topk, k)]
    for rank, idx in enumerate(top_indices, start=1):
        p = float(probs[idx])
        st.write(f"Top {rank}: {names[idx]} - {p:.3f}")
        st.progress(min(max(p, 0.0), 1.0))

    st.caption(f"Inference time: {elapsed*1000:.1f} ms (TensorFlow). " + ("Loaded user model." if (model_file is not None or weights_url) else "Demo model."))

    if saliency_on:
        st.subheader("Saliency (simple)")
        try:
            with tf.GradientTape() as tape:
                x = tf.convert_to_tensor(x_tensor)
                tape.watch(x)
                preds = model(x, training=False)
                top_idx = tf.argmax(preds[0])
                top_score = preds[0, top_idx]
            grads = tape.gradient(top_score, x)
            grads = tf.math.abs(grads); grads = grads / (tf.reduce_max(grads) + 1e-12)
            grads = grads[0].numpy()
            vis = x[0].numpy()
            vmin, vmax = vis.min(), vis.max()
            if vmax > vmin:
                vis = (vis - vmin) / (vmax - vmin)
            gray = vis.mean(axis=2, keepdims=True)
            heat = grads.mean(axis=2, keepdims=True)
            alpha = 0.6
            blended = (1 - alpha) * gray + alpha * heat
            blended = (np.clip(blended, 0, 1) * 255).astype("uint8")
            blended = np.repeat(blended, 3, axis=2)
            st.image(Image.fromarray(blended.squeeze()), caption="Simple saliency (not a medical heatmap)", use_container_width=True)
        except Exception:
            st.info("Saliency unavailable for this configuration.")
else:
    st.info("Upload a PNG/JPG (or DICOM if pydicom is installed) to begin.")
