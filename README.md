# Wine Quality Prediction - End-to-End MLOps Project

A complete machine learning operations (MLOps) pipeline for predicting wine quality using physicochemical properties. This project demonstrates production-ready ML practices including automated data ingestion, validation, transformation, model training, evaluation, and deployment with experiment tracking.

## 🎯 Project Overview

This project predicts wine quality scores (0-10) based on 11 physicochemical features using an ElasticNet regression model. The entire pipeline is modularized, configurable, and production-ready with MLflow integration for experiment tracking.

### Key Features

- ✅ **End-to-End MLOps Pipeline**: Automated workflow from data ingestion to model deployment
- ✅ **Modular Architecture**: Clean, maintainable code structure with separate components
- ✅ **Configuration Management**: YAML-based configuration for easy customization
- ✅ **Experiment Tracking**: MLflow integration for tracking metrics, parameters, and artifacts
- ✅ **Data Validation**: Schema validation to ensure data quality
- ✅ **Web Interface**: Flask-based UI for real-time predictions
- ✅ **Logging**: Comprehensive logging for debugging and monitoring
- ✅ **Dockerization Ready**: Containerization support for deployment

## 📊 Dataset

**Wine Quality Dataset** - Red wine variants of Portuguese "Vinho Verde"

### Input Features (11)
1. **Fixed Acidity**: Non-volatile acids (tartaric acid)
2. **Volatile Acidity**: Amount of acetic acid (high levels = vinegar taste)
3. **Citric Acid**: Adds freshness and flavor
4. **Residual Sugar**: Sugar remaining after fermentation
5. **Chlorides**: Amount of salt
6. **Free Sulfur Dioxide**: Prevents microbial growth and oxidation
7. **Total Sulfur Dioxide**: Free + bound forms of SO2
8. **Density**: Depends on alcohol and sugar content
9. **pH**: Acidity level (0-14 scale)
10. **Sulphates**: Wine additive contributing to SO2 levels
11. **Alcohol**: Percentage of alcohol content

### Target Variable
- **Quality**: Score between 0-10 (integer)

## 🏗️ Project Architecture

```
wine-quality/
├── artifacts/              # Pipeline outputs
│   ├── data_ingestion/     # Downloaded and unzipped data
│   ├── data_validation/    # Validation status
│   ├── data_transformation/ # Processed train/test splits
│   ├── model_trainer/      # Trained model artifacts
│   └── model_evaluation/   # Evaluation metrics
├── config/
│   └── config.yaml         # Pipeline configuration
├── logs/                   # Application logs
├── mlruns/                 # MLflow experiment tracking
├── src/project/
│   ├── components/         # Core pipeline components
│   ├── config/             # Configuration management
│   ├── constants/          # Project constants
│   ├── entity/             # Configuration entities
│   ├── pipeline/           # Pipeline orchestration
│   └── utils/              # Helper utilities
├── templates/              # Flask HTML templates
├── app.py                  # Flask web application
├── main.py                 # Pipeline execution script
├── params.yaml             # Model hyperparameters
├── schema.yaml             # Data schema validation
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "wine quality"
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Training the Pipeline

Run the complete end-to-end pipeline:

```bash
python main.py
```

This executes all stages in sequence:
- **Stage 1**: Data Ingestion - Downloads and extracts wine quality dataset
- **Stage 2**: Data Validation - Validates schema and data types
- **Stage 3**: Data Transformation - Splits data into train/test sets
- **Stage 4**: Model Training - Trains ElasticNet regression model
- **Stage 5**: Model Evaluation - Evaluates model and logs metrics to MLflow

### 2. Running the Web Application

Launch the Flask web interface:

```bash
python app.py
```

Then navigate to `http://localhost:5000` in your browser.

#### Web Application Features:

- **Home Page** (`/`): Input wine characteristics for prediction
- **Train** (`/train`): Trigger pipeline training via web interface
- **Predict** (`/predict`): Get quality predictions for new wine samples

### 3. Making Predictions

#### Via Web Interface:
1. Navigate to `http://localhost:5000`
2. Enter wine characteristics in the form
3. Click "Predict" to get quality score

#### Programmatically:
```python
from src.project.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np

# Example wine features
data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

# Make prediction
pipeline = PredictionPipeline()
prediction = pipeline.predict(data)
print(f"Predicted Quality: {prediction}")
```

## 📋 Pipeline Stages

### Stage 1: Data Ingestion
- Downloads dataset from source URL
- Extracts ZIP file to artifacts directory
- Prepares raw data for validation

### Stage 2: Data Validation
- Validates column names and data types against schema
- Checks for missing values and data integrity
- Generates validation status report

### Stage 3: Data Transformation
- Performs train-test split (default: 80-20)
- Handles any necessary feature engineering
- Saves processed datasets for training

### Stage 4: Model Training
- Trains ElasticNet regression model with configurable hyperparameters
- Alpha: Regularization strength (default: 0.2)
- L1 Ratio: Balance between L1 and L2 regularization (default: 0.1)
- Saves trained model as `model.joblib`

### Stage 5: Model Evaluation
- Evaluates model on test set
- Tracks metrics: MAE, RMSE, R²
- Logs parameters and metrics to MLflow
- Registers model artifacts

## ⚙️ Configuration

### Model Hyperparameters (`params.yaml`)
```yaml
ElasticNet:
  alpha: 0.2      # Regularization strength
  l1_ratio: 0.1   # L1/L2 ratio (0=Ridge, 1=Lasso)
```

### Pipeline Configuration (`config/config.yaml`)
- Customize artifact directories
- Modify data source URLs
- Configure model paths and names

### Data Schema (`schema.yaml`)
- Defines expected column names and data types
- Specifies target column
- Used for automated data validation

## 📊 Experiment Tracking with MLflow

This project uses MLflow for comprehensive experiment tracking:

### View Experiments
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view:
- Model parameters (alpha, l1_ratio)
- Evaluation metrics (MAE, RMSE, R²)
- Model artifacts and versions
- Run comparisons and visualizations

### Tracked Metrics
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (R-squared): Proportion of variance explained

## 🔍 Project Components

### Core Components (`src/project/components/`)

- **data_ingest.py**: Handles data download and extraction
- **data_validation.py**: Validates data against schema
- **data_transformation.py**: Performs data preprocessing
- **model_trainer.py**: Trains ML models
- **model_evaluation.py**: Evaluates and tracks model performance

### Pipeline Modules (`src/project/pipeline/`)

- **data_ingestion.py**: Orchestrates data ingestion stage
- **data_validation.py**: Orchestrates validation stage
- **data_transformation.py**: Orchestrates transformation stage
- **model_trainer.py**: Orchestrates training stage
- **model_evaluation.py**: Orchestrates evaluation stage
- **prediction_pipeline.py**: Handles inference for new data

### Utilities (`src/project/utils/`)

- **common.py**: Shared utility functions (YAML reading, file operations, etc.)

## 📦 Dependencies

Major packages used:
- **Flask**: Web framework for the application
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **mlflow**: Experiment tracking and model registry
- **joblib**: Model serialization
- **PyYAML**: Configuration management

See [requirements.txt](requirements.txt) for complete list.

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t wine-quality-predictor .
```

### Run Container
```bash
docker run -p 5000:5000 wine-quality-predictor
```

## 📈 Model Performance

The ElasticNet model provides a good balance between prediction accuracy and model simplicity:

- **Regularization**: Prevents overfitting
- **Feature Selection**: L1 penalty enables automatic feature selection
- **Interpretability**: Linear model coefficients are interpretable

Check `artifacts/model_evaluation/metrics.json` for current model metrics.

## 🛠️ Development

### Project Structure Philosophy

- **Modular Design**: Each component handles a single responsibility
- **Configuration Driven**: Easy to modify behavior without code changes
- **Logging**: Comprehensive logging for debugging and monitoring
- **Exception Handling**: Robust error handling throughout the pipeline

### Adding New Features

1. **New Component**: Add to `src/project/components/`
2. **Pipeline Integration**: Create corresponding pipeline in `src/project/pipeline/`
3. **Configuration**: Update `config/config.yaml` and entity classes
4. **Execution**: Add stage to `main.py`

## 🔒 Best Practices Implemented

- ✅ **Version Control**: Git-based workflow
- ✅ **Virtual Environments**: Isolated dependencies
- ✅ **Configuration Management**: Separate config from code
- ✅ **Logging**: Centralized logging system
- ✅ **Error Handling**: Try-except blocks with proper logging
- ✅ **Modular Code**: Separation of concerns
- ✅ **Experiment Tracking**: MLflow integration
- ✅ **Data Validation**: Schema-based validation
- ✅ **Model Versioning**: Artifact storage and tracking

## 🚧 Future Enhancements

- [ ] Add more ML algorithms (Random Forest, XGBoost, Neural Networks)
- [ ] Implement hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Add feature engineering pipeline
- [ ] Create CI/CD pipeline with GitHub Actions
- [ ] Deploy to cloud (AWS, Azure, or GCP)
- [ ] Add API authentication and rate limiting
- [ ] Implement batch prediction endpoint
- [ ] Add model monitoring and drift detection
- [ ] Create comprehensive test suite
- [ ] Add data versioning with DVC

## 📝 Logging

Logs are stored in the `logs/` directory with timestamps. Each pipeline stage logs:
- Stage start/completion
- Errors and exceptions
- Key operations and decisions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository - Wine Quality Dataset
- Inspired by MLOps best practices and industry standards
- Built with modern Python ML stack

## 📞 Support

For questions or issues:
- Open an issue in the repository
- Contact: your.email@example.com

---

**⭐ If you find this project helpful, please consider giving it a star!**

