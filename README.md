# House-Price-Prediction-Application
This project is a machine learning-based application that predicts house prices based on specific features such as Lot Area, Year Built, and Living Area. It uses a Random Forest Regressor model and provides an interactive interface built with Streamlit.

## Features
- Predict house prices based on user-provided inputs.
- Utilizes a pre-trained machine learning model.
- Simple and user-friendly web interface.

## Dataset
The model is trained using the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset from Kaggle. The dataset contains various features of houses and their sale prices.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
2. Install Dependencies
Ensure you have Python installed, then run:

bash
pip install -r requirements.txt
3. Train the Model (Optional)
If the house_price_model.pkl file is not included or you want to retrain the model:

bash
python train_model.py
4. Run the Application
Start the Streamlit application:

bash
streamlit run app.py
Usage
Open the application in your browser (Streamlit will provide a local URL).
Enter the house features in the input fields (e.g., Lot Area, Year Built, etc.).
Click Predict Price to get the estimated house price.

Dependencies
The project uses the following Python libraries:
pandas
numpy
scikit-learn
joblib
streamlit

Project Structure
bash
/house-price-prediction
├── train_model.py        # Script to train the model
├── app.py                # Streamlit application
├── house_price_model.pkl # Pre-trained model file (generated after training)
├── requirements.txt      # List of dependencies
├── README.md             # Documentation
Screenshots
Add screenshots of your application here to give users a visual guide.

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.

Acknowledgments
Special thanks to Kaggle for providing the dataset and inspiration for this project.
