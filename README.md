# 🧠 Brain Tumor MRI Classifier

This AI-powered tool analyzes brain MRI scans to detect and classify tumors using a Convolutional Neural Network (CNN). The web application is built with Streamlit and provides detailed predictions and visual explanations for uploaded MRI images.

---

## 🚀 Features

- **Brain Tumor Detection:** Classifies MRI scans into four categories: Glioma, Meningioma, Pituitary, or No Tumor.
- **Model Insights:** Displays model performance metrics and per-class precision, recall, and F1-score.
- **Grad-CAM Visualization:** Highlights regions of the MRI scan that influenced the model's prediction.
- **User-Friendly Interface:** Clean, interactive UI with sidebar metrics and expandable tumor information.
- **Educational Disclaimer:** For educational and research purposes only.

---

## 🏗️ Project Structure

```
brain-tumor-classifier/
├── src/
│   └── streamlit_app.py      # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── ...                       # Other files (models, assets, etc.)
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/brain-tumor-classifier.git
   cd brain-tumor-classifier
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖼️ Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run src/streamlit_app.py
   ```

2. **Open your browser:**  
   Go to the local URL provided by Streamlit (usually http://localhost:8501).

3. **Upload an MRI image:**  
   - Click the upload button and select a brain MRI image.
   - View the predicted tumor type, confidence, and Grad-CAM visualization.

---

## 📊 Model Information

- **Model Architecture:** Custom CNN
- **Training Data:** 1,695 MRI scans
- **Test Accuracy:** 76.0%
- **Balanced Accuracy:** 74.8%
- **Macro F1-Score:** 74.5%

**Performance by Tumor Type:**

| Tumor Type  | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Glioma      | 0.78      | 0.93   | 0.85     |
| Meningioma  | 0.65      | 0.51   | 0.57     |
| No Tumor    | 0.89      | 0.63   | 0.74     |
| Pituitary   | 0.75      | 0.93   | 0.83     |

---

## ⚠️ Disclaimer

> **This tool is for educational purposes only. It is not intended for medical diagnosis or treatment. Always consult a medical professional for health-related decisions.**

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- Open-source brain MRI datasets

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.



