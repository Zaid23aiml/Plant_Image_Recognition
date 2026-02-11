# 🌿 Plant Image Recognition System

An AI-powered web application that identifies medicinal plants from images and provides their **plant name, scientific name, benefits, and prediction confidence** using Deep Learning.

---

## 🚀 Features

- 🌱 Recognizes **150 different plant species**
- 🧠 Trained using **MobileNetV2 (Transfer Learning)**
- 📸 Works with images from:
  - Training dataset
  - Google images
  - Mobile camera photos
- 📊 Shows prediction **confidence** and **Top-3 results**
- 🛑 Prevents wrong predictions using confidence threshold
- 📱 Mobile-friendly web interface (Streamlit)

---

## 🧠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2**
- **NumPy, Pandas**
- **Streamlit**
- **Matplotlib**
- **Pillow**

---

## 📂 Project Structure

Plant_Image_Recognition/
│
├── app.py # Streamlit web app
├── plant_model.h5 # Trained deep learning model
├── plants_data.xlsx # Plant name, scientific name, benefits
├── class_names.txt # Class labels (150 plants)
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ How the Model Works

1. Plant images are organized into class folders.
2. Data augmentation is applied to generate variations.
3. Model is trained using **MobileNetV2** with fine-tuning.
4. Train / Validation / Test split is used.
5. On image upload:
   - Model predicts plant class
   - Confidence is calculated
   - Scientific name & benefits are fetched from Excel file

---

## ▶️ Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run the app
streamlit run app.py

🌐 Live Deployment

The application is deployed on Streamlit Cloud.

🔗 Live App: https://plantimagerecognition-gvfbdmdryuxdeeocejp3hr.streamlit.app/

Model Performance

Training Accuracy: ~97–99%

Validation Accuracy: ~99–100%

Test Accuracy: ~95–100%

Uses confidence threshold to avoid false predictions on unseen images

Note: Accuracy may vary for real-world images depending on lighting, angle, and background.

🔮 Future Improvements

Add more plant species

Improve real-world accuracy with more diverse images

Add disease detection

Deploy as a mobile app

Multilingual support


Author

Zaid Ansari
AI & Machine Learning Enthusiast
Plant Image Recognition Project


Acknowledgements

TensorFlow & Keras

Streamlit

MobileNetV2 (Google)
