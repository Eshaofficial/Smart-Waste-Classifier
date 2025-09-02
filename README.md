♻️ Smart Waste Classifier

An AI-powered application that classifies waste on the basis of Confidence Score into different categories using a Deep Learning Model and an interactive Streamlit interface.

🚀 Features

🧠 Deep Learning model for image classification

📊 Training history visualization

🌐 User-friendly Streamlit web app

🖼️ Upload waste images for real-time prediction

📂 Organized project structure

📱 Deployable on Streamlit Cloud

🛠️ Tech Stack

Python

TensorFlow / Keras

Streamlit

NumPy, Pandas, Matplotlib


📂 Project Structure
Smart-Waste-Classifier/
├── app.py                # Main Streamlit app
├── waste_classifier.h5   # Trained deep learning model
├── training_history.pkl  # Training history
├── requirements.txt      # Dependencies
├── images/               # UI assets (logo, header background, etc.)
└── README.md             # Documentation

📂 Dataset  
This project uses the [Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2).  
Download the dataset and place it inside the `dataset/` folder before running the app.  

⚙️ Installation

Clone this repository

git clone https://github.com/Eshaofficial/Smart-Waste-Classifier.git
cd Smart-Waste-Classifier


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

📌 Usage

Launch the app in your browser.

Upload an image of waste material.

The model classifies it into various classes.

📊 Results

Model trained with high accuracy on waste classification dataset.

Visualized training and validation accuracy/loss using Matplotlib.

☁️ Deployment

You can deploy the app easily using Streamlit Cloud:

streamlit run app.py


🔗 (https://smart-waste-classifier-rlwnnlc6zwbyr2m6kgtxh3.streamlit.app/)

🤝 Contributing

Contributions are welcome!

Fork the repo

Create a feature branch

Submit a pull request

📜 License

This project is licensed under the MIT License.

✨ Built with passion for a cleaner and smarter world!
