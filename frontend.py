import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/predict"


def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Prepare input data
    data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    # Send request to Flask API
    response = requests.post(API_URL, json={"features": data})
    result = response.json()

    # Interpret response
    if "prediction" in result:
        return "Diabetic" if result["prediction"] == 1 else "Not Diabetic"
    else:
        return "Error: " + result["error"]


# Gradio Interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"), gr.Number(label="Glucose"), gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"), gr.Number(label="Insulin"), gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"), gr.Number(label="Age")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Diabetes Prediction Website",
    description="Enter health parameters to predict diabetes."
)

if __name__ == '__main__':
    iface.launch(share=True)
