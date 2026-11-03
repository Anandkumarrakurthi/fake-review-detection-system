# Fake Review Detection System

An advanced AI-powered fake review detection system using ensemble machine learning techniques to identify fraudulent product reviews with industry-leading accuracy. The system leverages Random Forest, XGBoost, and Logistic Regression models to achieve 99.6% accuracy in detecting fake reviews.

![Fake Review Detection](https://cdn.prod.website-files.com/6579cdf1c0fe0d37fe87b60d/65fc06f47847dd9a279556e6_Review-Detection.png)

## 🎯 Project Overview

This project provides a complete solution for detecting fake reviews on e-commerce platforms. It features:
- **High Accuracy**: 99.6% accuracy using ensemble learning
- **Real-time Detection**: Instant predictions with confidence scores
- **Interactive Visualizations**: Dynamic charts and analytics
- **RESTful API**: Easy integration into existing platforms
- **Web Interface**: User-friendly interface for testing and analysis

## 📊 Key Features

### AI-Powered Detection
- Ensemble learning combining Random Forest, XGBoost, and Logistic Regression
- Multi-model voting system for robust predictions
- Confidence scoring for each prediction

### Real-time Analysis
- Instant review classification (Genuine/Fake)
- Individual model predictions with probabilities
- Interactive visualization of results

### Multi-Feature Analysis
The system analyzes 6 key features:
1. **Rating** (1-5 stars)
2. **Sentiment Score** (-1 to +1)
3. **Helpful Vote Count**
4. **Verified Purchase Status** (Yes/No)
5. **Review Length** (character count)
6. **Word Count**

### Fraud Pattern Detection
Identifies suspicious patterns such as:
- Sentiment-rating mismatches
- Unverified extreme opinions
- Suspiciously short/long reviews
- Reviews with no helpful votes despite length
- Moderate sentiment with extreme ratings

### Interactive Visualizations
- Model performance comparison charts
- Feature importance analysis
- Confusion matrices
- ROC curves and Precision-Recall curves
- Correlation heatmaps
- Real-time prediction visualizations

## 🏗️ Project Structure

```
fake-review-detection/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── traning.ipynb                   # Model training notebook
├── README.md                       # Project documentation
│
├── Dataset/                        # Training data
│   ├── products.csv               # Product information
│   └── reviews.csv                # Review dataset with labels
│
├── models/                         # Trained ML models
│   ├── random_forest_model.pkl    # Random Forest classifier
│   ├── xgboost_model.pkl          # XGBoost classifier
│   ├── logistic_regression_model.pkl  # Logistic Regression classifier
│   ├── scaler.pkl                 # Feature scaler
│   ├── feature_names.pkl          # Feature column names
│   └── model_performance.csv      # Model metrics
│
├── static/                         # Static assets
│   ├── css/                       # Stylesheets
│   └── js/                        # JavaScript files
│
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Home page
│   ├── prediction.html            # Prediction interface
│   ├── visualization.html         # Analytics dashboard
│   └── about.html                 # About page
│
└── Visualization Files/            # Saved visualizations
    ├── eda_visualization.png       # Exploratory Data Analysis
    ├── correlation_heatmap.png     # Feature correlations
    ├── model_comparison.png        # Model performance comparison
    ├── feature_importance.png      # Feature importance charts
    ├── confusion_matrices.png      # Confusion matrices
    ├── roc_curves.png             # ROC curves
    └── precision_recall_curves.png # Precision-Recall curves
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fake-review-detection.git
cd fake-review-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify model files**
Ensure all `.pkl` files are present in the `models/` directory:
- random_forest_model.pkl
- xgboost_model.pkl
- logistic_regression_model.pkl
- scaler.pkl
- feature_names.pkl

4. **Run the application**
```bash
python app.py
```

5. **Access the application}:L}_<'[0[ tevfcOpen your browser and navigate to: `http://localhost:5000`

## 💻 Usage

### Web Interface

1. **Home Page** (`/`)
   - Overview of the system capabilities
   - Key features and statistics
   - Quick access to prediction and visualization tools

2. **Prediction Page** (`/prediction`)
   - Input review details manually
   - Analyze review text automatically
   - Get instant predictions with confidence scores
   - View individual model predictions

3. **Visualization Page** (`/visualization`)
   - Model performance comparison
   - Feature importance charts
   - Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Interactive gauges and charts

4. **About Page** (`/about`)
   - Detailed information about the project
   - Model architecture and methodology
   - Dataset information
   - Technical details

### API Endpoints

#### 1. Predict Review
```http
POST /api/predict
Content-Type: application/json

{
  "rating": 5,
  "sentiment_score": 0.5,
  "helpfulVoteCount": 10,
  "verifiedPurchase": 1,
  "review_length": 150,
  "word_count": 30
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "GENUINE",
  "confidence": 95.5,
  "model_predictions": {
    "random_forest": {
      "prediction": "GENUINE",
      "probability": 96.2
    },
    "xgboost": {
      "prediction": "GENUINE",
      "probability": 94.8
    },
    "logistic_regression": {
      "prediction": "GENUINE",
      "probability": 95.5
    }
  },
  "visualization": "..."
}
```

#### 2. Analyze Review Text
```http
POST /api/analyze_text
Content-Type: application/json

{
  "review_text": "This product is amazing! Great quality and fast shipping."
}
```

**Response:**
```json
{
  "success": true,
  "features": {
    "review_length": 58,
    "word_count": 9,
    "sentiment_score": 0.75
  }
}
```

## 🔬 Model Performance

### Overall Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 99.68% | 98.08% | 96.23% | 97.14% | 99.93% |
| **XGBoost** | 99.58% | 96.23% | 96.23% | 96.23% | 99.93% |
| **Logistic Regression** | 82.42% | 21.78% | 83.02% | 34.51% | 91.12% |
| **Ensemble (Voting)** | **99.6%** | **96%** | **97%** | **96.5%** | **99.9%** |

### Feature Importance

The most important features for detection (Random Forest):
1. Sentiment Score (0.285)
2. Rating (0.245)
3. Review Length (0.180)
4. Word Count (0.165)
5. Verified Purchase (0.075)
6. Helpful Vote Count (0.050)

## 📈 Dataset Information

- **Total Reviews**: 6,327
- **Genuine Reviews**: 5,974 (94.4%)
- **Fake Reviews**: 353 (5.6%)
- **Features**: 23 columns (6 used for prediction)
- **Source**: Amazon product reviews dataset

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| rating | Float | Product rating (1-5 stars) |
| sentiment_score | Float | Text sentiment analysis score (-1 to +1) |
| helpfulVoteCount | Integer | Number of helpful votes received |
| verifiedPurchase | Binary | Whether purchase was verified (0/1) |
| review_length | Integer | Total character count of review |
| word_count | Integer | Total word count of review |

## 🛠️ Technical Details

### Technologies Used

**Backend:**
- Python 3.8+
- Flask 3.0.0 (Web framework)
- scikit-learn 1.3.0 (Machine Learning)
- XGBoost 2.0.0 (Gradient Boosting)
- Pandas 2.0.3 (Data manipulation)
- NumPy 1.24.3 (Numerical computing)

**Frontend:**
- Bootstrap 5.3.0 (UI framework)
- Plotly 5.17.0 (Interactive charts)
- AOS (Animation library)
- Font Awesome (Icons)

**Deployment:**
- Gunicorn 21.2.0 (WSGI server)

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Handle missing values
   - Convert categorical variables to binary
   - Calculate review length and word count
   - Feature scaling using StandardScaler

2. **Fake Review Detection Rules**
   - Sentiment-rating mismatch (high weight)
   - Unverified + extreme rating
   - Suspiciously short reviews with high ratings
   - Very long reviews (potential spam)
   - No helpful votes on lengthy reviews
   - Moderate sentiment with extreme rating

3. **Model Training**
   - Train-test split (80-20)
   - Cross-validation for hyperparameter tuning
   - Ensemble learning with majority voting

4. **Model Evaluation**
   - Confusion matrix analysis
   - ROC curve and AUC score
   - Precision-Recall curves
   - Feature importance analysis

## 🎨 Web Application Features

### Pages

1. **Home (`/`)**: Landing page with overview and quick access
2. **Prediction (`/prediction`)**: Interactive prediction interface
3. **Visualization (`/visualization`)**: Analytics dashboard
4. **About (`/about`)**: Detailed project information

### Interactive Features

- Real-time predictions
- Dynamic chart updates
- Responsive design (mobile-friendly)
- Smooth animations
- Loading states and error handling
- Form validation

## 🔒 Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

```bash
export FLASK_ENV=production
export FLASK_APP=app.py
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📝 Model Training

To retrain the models with new data:

1. Place your dataset in the `Dataset/` folder
2. Open `traning.ipynb` in Jupyter Notebook
3. Run all cells to:
   - Load and preprocess data
   - Perform exploratory data analysis
   - Train all three models
   - Evaluate performance
   - Save models to `models/` directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: Amazon Product Reviews dataset
- Inspired by research in fake review detection
- Built with Flask and Bootstrap

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**Built with ❤️ using Flask, scikit-learn, and XGBoost**
