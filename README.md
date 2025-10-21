# MedEngTriage - Medical Triage System

A machine learning-powered medical triage system designed to assist healthcare providers in prioritizing patient care based on symptoms and clinical indicators. The system provides triage recommendations in Somali language for better accessibility in Somali-speaking communities.

## 🏥 Overview

MedEngTriage is an intelligent medical triage system that uses machine learning to classify patients into different urgency levels based on their symptoms. The system is designed to help healthcare providers make informed decisions about patient prioritization and care pathways.

## ✨ Features

- **Multilingual Support**: Interface available in Somali language for better accessibility
- **Machine Learning-Powered**: Uses trained models to provide accurate triage recommendations
- **Interactive Web Interface**: User-friendly Streamlit-based web application
- **Comprehensive Symptom Assessment**: Covers multiple symptom categories including:
  - Fever (Qandho)
  - Cough (Qufac)
  - Headache (Madax-xanuun)
  - Abdominal Pain (Calool-xanuun)
  - Fatigue (Daal)
  - Vomiting (Matag)
- **Risk Stratification**: Classifies patients into urgency levels:
  - **Green**: Home care (Xaalad fudud)
  - **Amber**: Outpatient care (Xaalad dhax dhaxaad)
  - **Red**: Emergency care (Xaalad deg deg ah)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MedEngTriage.git
   cd MedEngTriage
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
MedEngTriage/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
├── models/                         # Trained ML models
│   ├── best_pipe.joblib           # Best performing pipeline
│   ├── feature_columns.json       # Feature column definitions
│   └── label_encoder.joblib       # Label encoder for predictions
├── artifacts/                      # Model artifacts and results
│   ├── best_model_metrics.json    # Model performance metrics
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   └── feature_importance.csv     # Feature importance analysis
├── figs/                          # Generated figures and plots
├── ui_assets/                     # UI configuration files
│   └── feature_schema.json        # Feature schema for UI
├── medical_triage.ipynb          # Jupyter notebook for analysis
├── triage.csv                     # Training dataset
└── column_headers_map_so.yml     # Column mapping for Somali interface
```

## 🔧 Technical Details

### Machine Learning Pipeline

The system uses a comprehensive machine learning pipeline that includes:

- **Data Preprocessing**: Feature engineering and data cleaning
- **Model Training**: Multiple algorithms tested (Random Forest, XGBoost, etc.)
- **Model Selection**: Best performing model selected based on metrics
- **Feature Importance**: Analysis of which symptoms are most predictive
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score

### Model Performance

The system achieves high accuracy in triage classification with detailed performance metrics available in the `artifacts/` directory.

### Key Technologies

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python with scikit-learn for machine learning
- **Data Processing**: Pandas and NumPy for data manipulation
- **Model Persistence**: Joblib for model serialization
- **Visualization**: Matplotlib and Seaborn for plots

## 📊 Usage

1. **Select Patient Age Group**: Choose from child, adult, or elderly
2. **Select Symptoms**: Choose one or more symptoms from the available categories
3. **Answer Follow-up Questions**: Provide additional details about selected symptoms
4. **Get Triage Recommendation**: The system will provide a triage level and care recommendations

## 🎯 Triage Levels

- **🟢 Green (Home Care)**: Mild symptoms that can be managed at home with self-care
- **🟡 Amber (Outpatient)**: Moderate symptoms requiring medical evaluation within 24 hours
- **🔴 Red (Emergency)**: Severe symptoms requiring immediate medical attention

## 📈 Model Development

The machine learning model was developed using:

- **Training Data**: Comprehensive dataset of patient symptoms and triage outcomes
- **Feature Engineering**: Derived features including red flag counts and symptom combinations
- **Cross-Validation**: Robust validation to ensure model generalizability
- **Hyperparameter Tuning**: Optimized model parameters for best performance

## 🤝 Contributing

We welcome contributions to improve MedEngTriage! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏥 Medical Disclaimer

**Important**: This system is designed to assist healthcare providers and should not replace professional medical judgment. All triage recommendations should be reviewed by qualified medical professionals.

## 📞 Support

For questions, issues, or contributions, please:

- Open an issue on GitHub
- Contact the development team
- Review the documentation

## 🔮 Future Enhancements

- Integration with electronic health records (EHR)
- Multi-language support expansion
- Mobile application development
- Real-time data integration
- Advanced analytics dashboard

---

**MedEngTriage** - Empowering healthcare through intelligent triage systems.
