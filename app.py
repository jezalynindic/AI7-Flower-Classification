
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Flower Classification AI",
    page_icon="🌻",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "EfficientNetB0.keras"
RESULTS_PATH = BASE_DIR / "comparison_results.csv"

CLASS_NAMES = ["Roses", "Sunflowers"]

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #fffaf0;
}

.main-title {
    text-align: center;
    color: #365b32;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 16px;
}

.result-card {
    background-color: #edf7e8;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #c5dfb9;
}

.result-title {
    color: #315c2c;
    font-size: 28px;
    font-weight: bold;
}

.section-title {
    color: #365b32;
    font-size: 27px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ==========================================
# LOAD COMPARISON RESULTS
# ==========================================

@st.cache_data
def load_results():

    if not RESULTS_PATH.exists():
        return None

    return pd.read_csv(RESULTS_PATH)


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🌻 Flower AI")

st.sidebar.write(
    "AI 7.0: Transfer Learning Performance "
    "Comparison in Image Classification"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Flower Classifier",
        "Model Comparison",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Classification Categories")
st.sidebar.write("🌹 Roses")
st.sidebar.write("🌻 Sunflowers")

st.sidebar.divider()

st.sidebar.markdown("### Selected Model")
st.sidebar.write("EfficientNetB0")

st.sidebar.markdown("### Test Accuracy")
st.sidebar.write("99.51%")

st.sidebar.divider()

st.sidebar.caption("Artificial Intelligence 7.0")


# ==========================================
# PAGE 1: FLOWER CLASSIFIER
# ==========================================

if page == "Flower Classifier":

    st.markdown(
        '<div class="main-title">'
        '🌻 Flower Classification AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Roses vs. Sunflowers using EfficientNetB0'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.info(
        "Upload a flower image to identify whether "
        "it is a Rose or a Sunflower."
    )

    uploaded_file = st.file_uploader(
        "Choose a flower image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Your uploaded flower image",
            width=500
        )

        if st.button(
            "Classify Flower",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing flower image..."
                ):

                    model = load_model()

                    processed_image = preprocess_image(
                        image
                    )

                    prediction = model.predict(
                        processed_image,
                        verbose=0
                    )

                    probability = float(
                        prediction[0][0]
                    )

                    if probability >= 0.5:

                        predicted_class = "Sunflowers"
                        confidence = probability * 100
                        flower_emoji = "🌻"

                    else:

                        predicted_class = "Roses"
                        confidence = (1 - probability) * 100
                        flower_emoji = "🌹"

                st.success("Classification completed!")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="font-size:60px;">
                            {flower_emoji}
                        </div>
                        <div class="result-title">
                            {predicted_class}
                        </div>
                        <p>Predicted Flower Category</p>
                        <h1>{confidence:.2f}%</h1>
                        <p>Model Prediction Score</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as error:

                st.error(f"Prediction error: {error}")


# ==========================================
# PAGE 2: MODEL COMPARISON DASHBOARD
# ==========================================

elif page == "Model Comparison":

    st.markdown(
        '<div class="main-title">'
        '📊 Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Comparing MobileNetV2, ResNet50, and EfficientNetB0'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    results = load_results()

    if results is None:

        st.error(
            "comparison_results.csv was not found. "
            "Please train the models first."
        )

    else:

        required_columns = [
            "Model",
            "Training Accuracy",
            "Validation Accuracy",
            "Test Accuracy",
            "Training Time (seconds)"
        ]

        if not all(
            column in results.columns
            for column in required_columns
        ):

            st.error(
                "The comparison CSV does not contain "
                "all required columns."
            )

        else:

            st.subheader("Overall Performance")

            # Best model based on validation accuracy,
            # then test accuracy, then training time.
            best_model = (
                results.sort_values(
                    by=[
                        "Validation Accuracy",
                        "Test Accuracy",
                        "Training Time (seconds)"
                    ],
                    ascending=[False, False, True]
                )
                .iloc[0]
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Best Model",
                best_model["Model"]
            )

            col2.metric(
                "Best Validation Accuracy",
                f'{best_model["Validation Accuracy"]:.2f}%'
            )

            col3.metric(
                "Best Test Accuracy",
                f'{best_model["Test Accuracy"]:.2f}%'
            )

            st.divider()

            # ----------------------------------
            # COMPARISON TABLE
            # ----------------------------------

            st.subheader("Model Comparison Table")

            display_results = results.copy()

            for column in [
                "Training Accuracy",
                "Validation Accuracy",
                "Test Accuracy"
            ]:

                display_results[column] = (
                    display_results[column].map(
                        lambda value: f"{value:.2f}%"
                    )
                )

            display_results[
                "Training Time (seconds)"
            ] = display_results[
                "Training Time (seconds)"
            ].map(
                lambda value: f"{value:.2f}"
            )

            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ----------------------------------
            # ACCURACY COMPARISON
            # ----------------------------------

            st.subheader("Accuracy Comparison")

            accuracy_data = results.set_index("Model")[
                [
                    "Training Accuracy",
                    "Validation Accuracy",
                    "Test Accuracy"
                ]
            ]

            st.bar_chart(
                accuracy_data,
                y=[
                    "Training Accuracy",
                    "Validation Accuracy",
                    "Test Accuracy"
                ],
                y_label="Accuracy (%)",
                height=400
            )

            st.caption(
                "Comparison of training, validation, "
                "and testing accuracy for each model."
            )

            st.divider()

            # ----------------------------------
            # TRAINING TIME
            # ----------------------------------

            st.subheader("Training Time Comparison")

            time_data = results.set_index("Model")[
                ["Training Time (seconds)"]
            ]

            st.bar_chart(
                time_data,
                y="Training Time (seconds)",
                y_label="Seconds",
                height=350
            )

            fastest_model = results.loc[
                results["Training Time (seconds)"].idxmin()
            ]

            st.info(
                f'The fastest model was '
                f'{fastest_model["Model"]}, '
                f'with a training time of '
                f'{fastest_model["Training Time (seconds)"]:.2f} seconds.'
            )

            st.divider()

            # ----------------------------------
            # PERFORMANCE ANALYSIS
            # ----------------------------------

            st.subheader("Performance Analysis")

            st.markdown(
                f"""
                **Best overall choice: {best_model["Model"]}**

                This model achieved:

                - Validation accuracy of
                  {best_model["Validation Accuracy"]:.2f}%
                - Test accuracy of
                  {best_model["Test Accuracy"]:.2f}%
                - Training time of
                  {best_model["Training Time (seconds)"]:.2f} seconds

                Based on these experimental results,
                {best_model["Model"]} is our selected model
                for the flower classification application.
                """
            )

            st.download_button(
                "Download Comparison Results",
                data=results.to_csv(index=False),
                file_name="model_comparison_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==========================================
# PAGE 3: ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">'
        '🌼 About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader("Project Title")

    st.write(
        "Comparing Transfer Learning Performance "
        "in Image Classification (Binary Classification)"
    )

    st.subheader("Project Description")

    st.write(
        """
        This project compares three pre-trained deep learning
        architectures for binary flower image classification.

        The models used are MobileNetV2, ResNet50,
        and EfficientNetB0.

        The goal is to identify which architecture provides
        the best balance between classification accuracy
        and training efficiency.
        """
    )

    st.subheader("Dataset")

    st.write("Flower categories:")

    st.write("🌹 Roses")
    st.write("🌻 Sunflowers")

    st.subheader("Transfer Learning Models")

    st.write("1. MobileNetV2")
    st.write("2. ResNet50")
    st.write("3. EfficientNetB0")

    st.subheader("Selected Model")

    st.success(
        "EfficientNetB0 was selected based on its "
        "validation accuracy, test accuracy, and "
        "training time in our experiment."
    )

    st.subheader("Technology Used")

    st.write("- Python")
    st.write("- TensorFlow / Keras")
    st.write("- Streamlit")
    st.write("- Pandas")
    st.write("- Matplotlib")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI 7.0 | Binary Flower Classification | "
    "Transfer Learning Performance Comparison"
)
