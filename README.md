
# Crop Recommendation System 🌾

A **Python-based Streamlit web app** that recommends the best crops to grow based on soil, weather, and environmental parameters. This helps farmers make data-driven decisions for better yield and sustainability.

---

## 🛠️ Technologies Used

- **Python**  
- **Streamlit** (Web app interface)  
- **Pandas** (Data manipulation)  
- **Scikit-learn** (Machine learning for crop recommendation)  
- **GitHub** (Version control)

---

## 📁 Project Structure

```markdown

crop-recommendation-system/
│
├─ Code/                 # Streamlit app scripts
│   └─ app.py            # Main Streamlit application
├─ Dataset/
|   └─ crop_recommendation.csv #Dataset 
├─ Models/
|   └─ crop_model.pkl
|   └─ scaler.pkl
|   └─ yield_model.pkl
|   └─ yield_scaler.pkl
├─ requirements.txt      # Python dependencies
└─ README.md

````

---

## 🚀 How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/vimalraju04/crop-recommendation-system.git
cd crop-recommendation-system/Code
````

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r ../requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

5. Open the browser at the URL shown in the terminal (usually `http://localhost:8501`).

---

## 🌐 Online Deployment

Live URL:

[https://vimalraju04-crop-recommendation-system.streamlit.app](https://crop-recommendation-system-vaah.streamlit.app/)

---

## 📊 Features

* Suggests the most suitable crop based on soil nutrients, temperature, rainfall, and other factors.
* Easy-to-use web interface with interactive input fields.
* Helps farmers reduce risk and improve yield.

---

## 📄 License

This project is **open source** under the MIT License.

---

