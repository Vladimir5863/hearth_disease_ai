# Heart Disease Risk Prediction using MLP

A machine learning project that predicts heart disease risk using a Multi-Layer Perceptron (MLP) neural network. The project includes model training, evaluation, and a Streamlit web application for interactive predictions.

## 📊 Overview

This project uses the Heart Disease Health Indicators dataset to train an MLP classifier that predicts whether a person has heart disease based on various health and lifestyle factors. The model achieves good performance using a two-hidden-layer neural network architecture.

**Built with [uv](https://github.com/astral-sh/uv)** - an extremely fast Python package installer and resolver written in Rust, replacing the need for pip and traditional virtual environment management.

## 🚀 Features

- **MLP Neural Network**: Two-layer neural network (128, 64 neurons) with ReLU activation
- **Preprocessing Pipeline**: Automated feature scaling and encoding
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Model Persistence**: Saved model and metadata for easy deployment
- **Performance Metrics**: ROC-AUC evaluation for binary classification

## 📋 Requirements

- **uv** - Fast Python package installer and resolver
- Python 3.13+ (managed by uv)
- Dependencies listed in `pyproject.toml`:
  - pandas >= 2.3.2
  - scikit-learn >= 1.7.1
  - streamlit >= 1.49.1
  - seaborn >= 0.13.2

## 🛠️ Installation

This project uses **[uv](https://github.com/astral-sh/uv)** - an extremely fast Python package installer and resolver written in Rust.

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd ai-projekat
   ```

2. **Install uv**

   ```bash
   pip install uv
   ```

3. **Install project dependencies**

   ```bash
   uv sync
   ```

   This will automatically:

   - Create a virtual environment
   - Install Python 3.13 if needed
   - Install all project dependencies

4. **Ensure you have the dataset**
   - Place `heart_2020_cleaned.csv` in the project root directory

## 🔍 Exploring Additional Analysis

To access the comprehensive analysis and model comparison notebooks:

```bash
# View Jupyter notebooks with dataset analysis and model comparison
git checkout jupyter_notebook

# Return to main branch
git checkout main
```

## 📈 Dataset Features

The model uses the following features to predict heart disease risk:

### Numeric Features

- **BMI**: Body Mass Index
- **PhysicalHealth**: Physical health (days in past 30 days when physical health was not good)
- **MentalHealth**: Mental health (days in past 30 days when mental health was not good)
- **SleepTime**: Hours of sleep per day

### Categorical Features

- **Smoking**: Yes/No
- **AlcoholDrinking**: Yes/No
- **Stroke**: Yes/No
- **DiffWalking**: Difficulty walking - Yes/No
- **Sex**: Female/Male
- **AgeCategory**: 18-24, 25-29, ..., 80 or older
- **Race**: American Indian/Alaskan Native, Asian, Black, Hispanic, Other, White
- **Diabetic**: No, No (borderline diabetes), Yes, Yes (during pregnancy)
- **PhysicalActivity**: Yes/No
- **GenHealth**: Excellent, Very good, Good, Fair, Poor
- **Asthma**: Yes/No
- **KidneyDisease**: Yes/No
- **SkinCancer**: Yes/No

## 🔧 Usage

### Training the Model

Train the MLP model using uv:

`bash uv run train_mlp.py`

This will:

- Load and preprocess the data
- Split data into train/validation/test sets (70%/15%/15%)
- Train the MLP classifier with early stopping
- Evaluate model performance using ROC-AUC
- Save the trained model (`mlp_model.pkl`) and metadata (`mlp_meta.json`)

### Running the Web Application

Launch the Streamlit web interface:

```bash
uv run streamlit run app_mlp.py
```

The web app provides:

- Interactive sidebar inputs for all features
- Real-time heart disease risk prediction
- Probability scores and binary predictions

## 🏗️ Model Architecture

- **Algorithm**: Multi-Layer Perceptron (MLPClassifier)
- **Hidden Layers**: 2 layers (128, 64 neurons)
- **Activation Function**: ReLU
- **Solver**: Adam optimizer
- **Learning Rate**: 0.001
- **Batch Size**: 1024
- **Early Stopping**: Yes (patience: 3 iterations)
- **Max Iterations**: 20

## 📊 Preprocessing Pipeline

1. **Numeric Features**: MinMaxScaler normalization
2. **Categorical Features**: One-hot encoding with unknown category handling
3. **Target Encoding**: "Yes"/"No" → 1/0

## 🎯 Performance

The model's performance is evaluated using ROC-AUC score on the test set. The exact performance will be displayed when running the training script.

## 🔮 Making Predictions

The trained model can predict:

- **Probability**: Likelihood of heart disease (0.0 to 1.0)
- **Binary Prediction**: Yes/No classification (threshold: 0.5)

## 🚀 Deployment

The Streamlit app can be easily deployed to various platforms:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using Procfile and requirements.txt
- **Docker**: Container-based deployment
- **Local Server**: Run locally for development/testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 📧 Contact

For questions or suggestions, please open an issue in the GitHub repository.

---

_Built with scikit-learn, Streamlit, uv, and ❤️_
