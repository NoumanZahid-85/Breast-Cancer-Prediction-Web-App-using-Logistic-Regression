# Streamlit Cancer Prediction App

## Overview
This project is a Streamlit application designed for cancer prediction based on cell nuclei details. It utilizes machine learning models to provide predictions and visualizations, helping users understand the data and the model's output.

## Features
- **User Input**: Users can input cell nuclei details through a sidebar interface.
- **Data Visualization**: The app includes radar charts to visualize the input data.
- **Model Prediction**: The application predicts cancer presence based on the input data using a pre-trained logistic regression model.

## Technologies Used
- **Python**: The primary programming language.
- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Plotly**: For creating interactive visualizations.
- **Scikit-learn**: For machine learning functionalities.

## Installation
To run this project, you need to install the required packages. Use the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Streamlit_Cancer_App
   ```
2. Run the application:
   ```bash
   streamlit run app/main.py
   ```
3. Open your web browser and navigate to the provided local URL.

## Data
The application uses a dataset located in the `Data` folder. Ensure that the `data.csv` file is present in this directory.

## Model
The logistic regression model is stored in the `Model` directory as `logistic_model.pkl`, and the scaler is stored as `scaler.pkl`.

## Images
Here are some images related to the project:
### Benign Prediction
![Image 1](images/image.png)
### Malignant Prediction
![Image 2](images/image%20copy.png)
### Technical Details
![Image 3](images/image%20copy%202.png)
___