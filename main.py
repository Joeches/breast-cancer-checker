import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pyttsx3
import threading

# Load CSS styles
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS styles
load_css("static/styles.css")

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Calculate model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Configure Streamlit UI
st.title("Breast Cancer Diagnosis")
st.sidebar.header("Patient Input")

# Collect patient data using sliders
patient_data = {}
for feature in data.feature_names:
    try:
        min_val, max_val = np.min(X[:, data.feature_names.tolist().index(feature)]), np.max(X[:, data.feature_names.tolist().index(feature)])
        default_val = (min_val + max_val) / 2  # Use the average as the default value
        patient_data[feature] = st.sidebar.slider(f"{feature}:", float(min_val), float(max_val), float(default_val))
    except Exception as e:
        st.sidebar.write(f"Error: {str(e)}")

# Button to view patient input
if st.sidebar.button("View Patient Input"):
    st.sidebar.subheader("Patient Input")
    st.sidebar.write(patient_data)

try:
    # Make prediction based on patient data
    prediction = clf.predict(scaler.transform(np.array(list(patient_data.values())).reshape(1, -1)))[0]
    prediction_text = data.target_names[prediction]

    # Calculate prediction probabilities
    probabilities = clf.predict_proba(scaler.transform(np.array(list(patient_data.values())).reshape(1, -1)))[0]
    prob_df = pd.DataFrame({"Class": data.target_names, "Probability": probabilities})

    # Visualize prediction probabilities
    st.subheader("Prediction Probabilities")

    # Define the color palette
    colors = ["#e91e63", "#009688"]

    # Set the color palette and plot the barplot
    sns.set_palette(colors)
    fig, ax = plt.subplots()
    sns.barplot(x="Class", y="Probability", data=prob_df, ax=ax)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    ax.set_xticklabels(data.target_names)
    st.pyplot(fig)

    # Display the result
    st.subheader("Diagnosis Result")
    result_text = f"The diagnosis result is {prediction_text}"
    st.write(result_text)

    # Initialize the Text-to-Speech engine
    engine = pyttsx3.init()

    # Function to run pyttsx3 in a separate thread
    def run_pyttsx3():
        engine.say(result_text)
        engine.runAndWait()

    # Start pyttsx3 in a separate thread
    pyttsx3_thread = threading.Thread(target=run_pyttsx3)
    pyttsx3_thread.start()

    # Display if the patient has breast cancer or not
    st.subheader("Breast Cancer Diagnosis")
    if prediction == 0:
        st.write("The patient is diagnosed with <span class='diagnosis-result-benign'>**Benign**</span> (non-cancerous) tumor.", unsafe_allow_html=True)
    else:
        st.write("The patient is diagnosed with <span class='diagnosis-result-malignant'>**Malignant**</span> (cancerous) tumor.", unsafe_allow_html=True)

    # Display model accuracy
    st.subheader("Model Accuracy")
    st.write(f"The model accuracy is {accuracy:.2%}")

    # Button to view other datasets
    if st.button("View Other Datasets"):
        st.subheader("Other Datasets")
        st.write("You can display additional datasets here.")

except Exception as e:
    st.subheader("Error")
    st.write(str(e))

# Show dataset information
if st.checkbox("Show Dataset Information"):
    st.subheader("Breast Cancer Dataset")
    st.write(f"Number of instances: {X.shape[0]}")
    st.write(f"Number of features: {X.shape[1]}")
    st.write(f"Number of classes: {len(data.target_names)}")
    st.write(f"Class names: {', '.join(data.target_names)}")

# Streamlit test conclusion
st.sidebar.markdown("---")
st.sidebar.subheader("Test Conclusion")
st.sidebar.write("Thank you for using the Breast Cancer Diagnosis app!")
st.sidebar.write("If you have any feedback or questions, please contact us at:")
st.sidebar.write("Developer Company: JoeTech Digitals")
st.sidebar.write("Email: techbreedafrican@gmail.com")
st.sidebar.write("Mobile: +2348163399026")
st.sidebar.write("Stay healthy!")

# Hide Streamlit menu and footer
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)
