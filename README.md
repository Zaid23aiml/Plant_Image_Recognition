# 🌿 Plant Image Recognition System  

An AI-powered web application that identifies medicinal plants from images and provides their scientific names and benefits using Deep Learning.

🔗 **Live App:**  
https://plantimagerecognition-gvfbdmdryuxdeeocejp3hr.streamlit.app/

---

## 📌 Project Overview

The Plant Image Recognition System is a deep learning-based application that classifies plant images into 150 plant species.

It uses a MobileNetV2-based Convolutional Neural Network (CNN) trained using transfer learning to achieve high accuracy.

The system provides:

- 🌱 Plant Name  
- 🔬 Scientific Name  
- 💊 Medicinal Benefits  
- 📊 Top-3 Predictions with Confidence Scores  
- 📈 Confidence Bar Chart Visualization  

The application is deployed using **Streamlit Cloud** for real-time interaction.

---

## 🧠 Model Details

- Architecture: MobileNetV2 (Transfer Learning)
- Framework: TensorFlow / Keras
- Input Image Size: 224x224
- Total Classes: 150 Plant Species
- Image Normalization: Rescaled (0–1)
- Confidence Threshold Handling for Unknown Images
- Top-3 Predictions Enabled

---

## 📊 Model Performance

- Training Accuracy: ~97–99%
- Validation Accuracy: ~99–100%
- Test Accuracy: ~95–100%

> Note: Real-world accuracy may vary depending on lighting conditions, background complexity, and image clarity.

---

## 🖥️ Tech Stack

### Backend
- Python
- TensorFlow
- Keras
- NumPy
- Pandas

### Frontend
- Streamlit
- Matplotlib (Confidence Visualization)

### Deployment
- GitHub
- Streamlit Cloud

---

## 📂 Project Structure

Plant_Image_Recognition/
│
├── app.py
├── plant_model.h5
├── plants_data.xlsx
├── class_names.txt
├── requirements.txt
└── README.md


---

## 🚀 Installation (Run Locally)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Zaid23aiml/Plant_Image_Recognition.git
cd Plant_Image_Recognition


2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py


✨ Features

✔ Upload plant image (jpg, jpeg, png)
✔ Real-time Prediction
✔ Top-3 Predictions
✔ Confidence Score Display
✔ Confidence Bar Chart
✔ Medicinal Benefits Information
✔ Error Handling for Low Confidence
✔ Clean & Responsive UI

🔮 Future Improvements

Add more plant species

Improve real-world generalization

Add plant disease detection

Add weed detection

Add multilingual support

Convert into a mobile application

👨‍💻 Author

Mohd Zaid Ansari
AI & Machine Learning Enthusiast

GitHub: https://github.com/Zaid23aiml

🙏 Acknowledgements

TensorFlow & Keras

Streamlit

MobileNetV2 (Google Research)