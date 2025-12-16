# Animal Classifier - Streamlit App

A web application that uses a trained ResNet18 model to classify animal images into 15 different categories.

## Features

- 🖼️ Upload animal images (JPG, JPEG, PNG)
- 🔍 Real-time animal classification
- 📊 Top 3 predictions with confidence scores
- 📈 Visual charts showing all predictions
- 🎨 Modern and user-friendly interface

## Supported Animals

- Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the model file `animal_resnet.pth` in the same directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)
4. Upload an image of an animal and click "Predict Animal"

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- PIL/Pillow
- pandas

##🤝 Contributions & Feedback

I’m still learning and growing in Machine Learning and Deep Learning.
If you notice any mistakes, improvements, or better ways to structure or optimize this project, I’d really appreciate your help.

Feel free to open an issue for suggestions or feedback

You’re welcome to submit a pull request to improve the code, model, or UI

Any tips on best practices, performance, or design are always welcome

This project is part of my learning journey, and constructive feedback will help me improve 🚀