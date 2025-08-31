♻️ Smart Waste Classifier

An AI-powered web app built with Streamlit that classifies waste into categories using a deep learning model. This project promotes smart waste management by assisting users in identifying whether waste is biodegradable, recyclable, or hazardous.

🚀 Features

🧠 Deep Learning Model trained to classify waste images

🎨 Interactive UI built with Streamlit

📊 Training history visualization (loss & accuracy curves)

🖼️ Custom background and logo integration

📦 Model (waste_classifier.h5) and training history (training_history.pkl) included

🌍 Deployable on Streamlit Cloud

📂 Repository Structure
Smart-Waste-Classifier/
│
├─ app.py                       # Main Streamlit app
├─ waste_classifier.h5          # Trained Keras/TensorFlow model (Git LFS)
├─ training_history.pkl         # Training history for visualization
├─ header_bg.jpg                # Header background image (Git LFS)
├─ logo_icon.png                # App logo (Git LFS)
├─ requirements.txt             # Python dependencies
├─ README.md                    # Project documentation
└─ .gitattributes               # Git LFS config for large files

🛠️ Installation
# Clone the repo
git clone https://github.com/Eshaofficial/Smart-Waste-Classifier.git
cd Smart-Waste-Classifier

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

▶️ Usage
streamlit run app.py


Then open the URL shown in your terminal (default: http://localhost:8501/).

☁️ Deploy on Streamlit Cloud

Push your repo to GitHub (with large files tracked via Git LFS).

Go to Streamlit Cloud
.

Select:

Repo: Eshaofficial/Smart-Waste-Classifier

Branch: main

File path: app.py

Click Deploy.

Your app will be live within a few minutes!

🧰 Requirements

tensorflow / keras

numpy

pandas

matplotlib

streamlit

Pillow

pip install -r requirements.txt

📊 Example Workflow

Upload an image of waste.

Model predicts the category (e.g., Biodegradable, Recyclable, Hazardous).

Displays result with probability.

Visualize training history to see how the model learned.

🔮 Future Improvements

Support for more waste categories

Mobile app integration

Real-time camera detection

Dataset expansion for better accuracy

🤝 Contributing

Contributions are welcome! Please fork this repo and submit a pull request.

📜 License

This project is licensed under the MIT License.

✨ Acknowledgments

TensorFlow/Keras for deep learning framework

Streamlit for easy web deployment

Git LFS for handling large files
