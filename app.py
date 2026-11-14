
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Content Moderation Assistant",
    page_icon="🛡️",
    layout="centered"
)

# Application title
st.title("🛡️ AI-Powered Content Moderation Assistant")
st.write(
    "Analyze online comments and identify potentially "
    "harmful content using Artificial Intelligence."
)

st.info(
    "This tool assists human moderators. "
    "It does not make final moderation decisions."
)

# Load AI model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None
    )

# User input
user_text = st.text_area(
    "Enter a comment to analyze:",
    placeholder="Type a social media comment here..."
)

# Analyze button
if st.button("Analyze Content", type="primary"):

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        with st.spinner("AI is analyzing the content..."):

            try:
                classifier = load_model()
                results = classifier(user_text)[0]

                # Find the highest scoring category
                scores = {
                    item["label"]: item["score"]
                    for item in results
                }

                highest_label = max(
                    scores,
                    key=scores.get
                )

                highest_score = scores[highest_label]

                # Moderation recommendation
                if highest_score >= 0.70:
                    decision = "REVIEW"
                    st.warning(
                        "Potentially harmful content detected. "
                        "Human review recommended."
                    )

                elif highest_score >= 0.40:
                    decision = "CHECK"
                    st.info(
                        "Uncertain content. "
                        "Additional review recommended."
                    )

                else:
                    decision = "LIKELY SAFE"
                    st.success(
                        "No strong harmful-content signal detected."
                    )

                # Display results
                st.subheader("Analysis Results")

                st.write("**Recommended action:**", decision)
                st.write("**Highest detected category:**", highest_label)
                st.write(
                    "**Model confidence:**",
                    f"{highest_score * 100:.2f}%"
                )

                st.subheader("All Category Scores")

                for label, score in scores.items():
                    st.write(
                        f"{label}: {score * 100:.2f}%"
                    )
                    st.progress(float(score))

                st.caption(
                    "Model scores are not proof of harmful intent. "
                    "Context and human judgment are important."
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.write(
                    "Check your internet connection and model setup."
                )

# Footer
st.divider()
st.caption("AI Content Moderation Assistant | Portfolio Project")
