from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "baseline_efficientnet_b0.pt"
CLASSES = ["sehat", "sakit"]
LABELS = {"sehat": "Healthy", "sakit": "Diseased"}
IMG_SIZE = (224, 448)  # height x width, matching the ~1:2 ratio of the training images
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

st.set_page_config(page_title="Oil Palm Leaf Disease Detection", page_icon="🌴", layout="centered")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, device


tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

st.title("🌴 Oil Palm Leaf Disease Detection")
st.caption(
    "Research prototype: EfficientNet-B0 (transfer learning) classifies a close-up oil palm leaf image "
    "as healthy or diseased, and Grad-CAM shows which regions influenced the decision."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model, device = load_model()

file = st.file_uploader("Upload a close-up photo of an oil palm leaf", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())

    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
    rgb = np.asarray(img.resize((IMG_SIZE[1], IMG_SIZE[0])), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded image", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM (red = most influential)", use_container_width=True)

    st.subheader(f"Prediction: {LABELS[CLASSES[pred]]}")
    st.write(f"Confidence: **{probs[pred] * 100:.1f}%**")
    for i, c in enumerate(CLASSES):
        st.progress(float(probs[i]), text=f"{LABELS[c]}: {probs[i] * 100:.1f}%")

with st.expander("About this model and its limitations"):
    st.markdown(
        """
- **Data:** 1,583 close-up, cropped leaf images (899 diseased, 684 healthy) from oil palm plantations in
  Tanah Laut, labeled by domain experts (Mendeley Data, DOI 10.17632/cp65skff89.1).
- **Validation:** accuracy of about 98.9% when whole cartons (photo batches) are held out for testing.
  A model that sees only 6 color statistics per image already reaches about 89.6%, so color differences
  are partly informative.
- **Not yet validated** on photos from other sources, cameras, or plantations. Diseased and healthy images
  may have been captured in different sessions, so the model may partly learn photography style.
- **Binary only:** it does not identify the type of disease, and a confident answer is not a diagnosis.
- Photos that are not close-up oil palm leaves (or are very different from the training images) can still
  produce a confident but meaningless answer.
- This is a research prototype and not a replacement for an agronomist's assessment.
        """
    )