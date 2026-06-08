import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

st.sidebar.title("🐾 Cat vs Dog Classifier")

st.warning(
    "This model only recognizes Cats and Dogs. Other animals may produce incorrect predictions."
)

st.sidebar.markdown("""
### Model Information

- Model: ResNet18
- Accuracy: 98.06%
- Framework: PyTorch
- Deployment: Streamlit

### Developer

Vansh Garg
""")

model = resnet18(weights=None)

model.fc = nn.Linear(512, 2)

model.load_state_dict(
    torch.load(
        "cats_dogs_resnet18.pth",
        map_location="cpu"
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🐱🐶 Cat vs Dog Classifier")

st.markdown(
    "Upload an image and click **Predict** to identify whether it is a Cat or a Dog."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        if st.button("🔍 Predict", use_container_width=True):

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():

                output = model(image_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )[0]

            cat_prob = probabilities[0].item() * 100
            dog_prob = probabilities[1].item() * 100

            prediction = "Cat 🐱" if cat_prob > dog_prob else "Dog 🐶"
            confidence = max(cat_prob, dog_prob)

            st.success(
                f"Prediction: {prediction}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write("Cat Probability")
            st.progress(int(cat_prob))

            st.write("Dog Probability")
            st.progress(int(dog_prob))

st.markdown("---")
st.caption("Built with PyTorch • ResNet18 • Streamlit")