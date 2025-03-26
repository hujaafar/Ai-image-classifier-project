# 🖼️ CIFAR-10 Image Classifier Project

## 🚀 Project Overview
This is a simple image classification project built with **TensorFlow** and deployed as a web app using **Streamlit**.

The model is trained on the **CIFAR-10** dataset and can classify images into one of the following categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## ✅ Features
- Pre-trained Convolutional Neural Network (CNN) on CIFAR-10
- Simple web app interface to upload an image and get a prediction
- Fully deployable on **Streamlit Cloud**

---

## 📂 Project Structure
```
|-- cifar10_classifier.py  # Model training script
|-- cifar10_model.h5      # Saved trained model
|-- app.py                # Streamlit app file
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation (this file)
```

---

## 💻 How to Run Locally
1. Clone this repository:
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the Streamlit app:
```
streamlit run app.py
```

---


## 🤖 Model Training
If you’d like to retrain the model:
```
python cifar10_classifier.py
```
This will train the model on CIFAR-10 and save it as `cifar10_model.h5`.

---

## 🛠 Tech Stack
- Python
- TensorFlow
- NumPy
- Matplotlib
- Streamlit

---

## ⭐ Credits
Built with ❤️ by [hujaafar] as a beginner-friendly AI project.

---

## 📧 Contact
If you have questions, feel free to reach out or fork and improve the project!

