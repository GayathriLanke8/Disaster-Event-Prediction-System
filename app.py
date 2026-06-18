import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("disaster_model_.pkl")


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Disaster Event Prediction",
    page_icon="🌍",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🌍 Disaster Event Prediction System")

st.write(
    "Enter disaster details to predict whether it is a major disaster or not."
)


# -----------------------------
# User Inputs
# -----------------------------

disaster_type = st.selectbox(
    "Disaster Type",
    [
        "Wildfire",
        "Flood",
        "Earthquake",
        "Hurricane",
        "Drought",
        "Landslide",
        "Volcanic Eruption"
    ]
)


location = st.selectbox(
    "Location",
    [
        "Chile",
        "India",
        "Italy",
        "USA",
        "Japan",
        "Indonesia",
        "Turkey",
        "Philippines"
    ]
)


severity_level = st.slider(
    "Severity Level",
    min_value=1,
    max_value=10,
    value=5
)


affected_population = st.number_input(
    "Affected Population",
    min_value=0,
    value=10000
)


economic_loss = st.number_input(
    "Estimated Economic Loss (USD)",
    min_value=0.0,
    value=1000000.0
)


response_time = st.number_input(
    "Response Time (Hours)",
    min_value=0.0,
    value=24.0
)


aid_provided = st.selectbox(
    "Aid Provided",
    [
        "Yes",
        "No"
    ]
)


infra_damage = st.slider(
    "Infrastructure Damage Index",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)



# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict"):


    input_data = pd.DataFrame({

        "disaster_type": [
            disaster_type
        ],

        "location": [
            location
        ],

        "severity_level": [
            severity_level
        ],

        "affected_population": [
            affected_population
        ],

        "estimated_economic_loss_usd": [
            economic_loss
        ],

        "response_time_hours": [
            response_time
        ],

        "aid_provided": [
            aid_provided
        ],

        "infrastructure_damage_index": [
            infra_damage
        ]

    })


    prediction = model.predict(input_data)[0]


    # Probability

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0]

        major_probability = probability[1]

        normal_probability = probability[0]



    # Output

    if prediction == 1:

        st.error(
            "⚠️ Major Disaster Detected"
        )

        st.write(
            f"Major Disaster Probability: {major_probability:.2%}"
        )

    else:

        st.success(
            "✅ Not a Major Disaster"
        )

        st.write(
            f"Normal Probability: {normal_probability:.2%}"
        )


    st.subheader("Input Details")

    st.dataframe(
        input_data
    )