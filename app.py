from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, session)
import joblib
import numpy as np
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os
import sqlite3
from datetime import timedelta
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-production-xyz123')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['DATABASE'] = 'users.db'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
#  DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

# ============================================================
#  AUTH HELPERS
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Session expired. Please log in again.'}), 401
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated

def current_user():
    if 'user_id' not in session:
        return None
    with get_db() as conn:
        row = conn.execute('SELECT id, name, email FROM users WHERE id = ?',
                           (session['user_id'],)).fetchone()
    return dict(row) if row else None

@app.context_processor
def inject_user():
    return {'current_user': current_user()}

# ============================================================
#  LOAD MODELS
# ============================================================
print("Loading models...")
rf_model  = joblib.load('models/random_forest_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
lr_model  = joblib.load('models/logistic_regression_model.pkl')
scaler    = joblib.load('models/scaler.pkl')
model_performance = pd.read_csv('models/model_performance.csv', index_col=0)
print("✓ All models loaded successfully!")

FEATURE_NAMES = ['rating', 'sentiment_score', 'helpfulVoteCount',
                 'verifiedPurchase', 'review_length', 'word_count']

# ============================================================
#  TEXT & RULE-BASED PREDICTION LOGIC
# ============================================================
def analyze_text_for_fake(text):
    if not text: return False, 0.0
    text_lower = text.lower()
    
    fake_phrases = [
        "best product ever", "worst product ever", "waste of money", 
        "do not buy", "buy it now", "amazing quality", "outstanding product",
        "unbelievable experience", "best purchase ever", "total garbage",
        "horrible product", "highly recommend"
    ]
    
    if len(text) < 80 and any(phrase in text_lower for phrase in fake_phrases):
        return True, 0.98  
    if text.count('!') > 3:
        return True, 0.95
    if len(text.split()) > 20 and len(text) > 100:
        return False, 0.90
    return False, 0.0

def predict_logic(rating, sentiment, votes, verified, length, words, text=''):
    # 1. Text analysis
    text_is_fake, text_confidence = analyze_text_for_fake(text)
    if text_is_fake:
        return True, text_confidence

    # 2. Rating/Sentiment Mismatch
    if (rating == 1 and sentiment > 0.5) or (rating == 5 and sentiment < -0.5):
        return True, 0.95 

    # 3. Long detailed text with verified purchase
    if len(text) > 100 and verified == 1:
        return False, 0.92

    # 4. Unverified and 0 votes
    if verified == 0 and votes == 0:
        return True, 0.85

    # 5. ML Models Fallback
    features = [rating, sentiment, votes, verified, length, words]
    scaled = scaler.transform([features])
    
    rf_p = float(rf_model.predict_proba(scaled)[0][1])
    xg_p = float(xgb_model.predict_proba(scaled)[0][1])
    lr_p = float(lr_model.predict_proba(scaled)[0][1])
    avg_p = (rf_p + xg_p + lr_p) / 3.0
    
    is_fake = avg_p > 0.5
    return is_fake, avg_p

# ============================================================
#  AUTH ROUTES
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('index'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        
        if password != confirm:
            flash('Passwords do not match.', 'danger')
        else:
            try:
                with get_db() as conn:
                    conn.execute('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                                 (name, email, generate_password_hash(password)))
                    conn.commit()
                flash('Account created! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Email already registered.', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================================================
#  PAGES
# ============================================================
@app.route('/')
def root(): return redirect(url_for('index') if 'user_id' in session else url_for('login'))

@app.route('/home')
@login_required
def index(): return render_template('index.html')

@app.route('/prediction')
@login_required
def prediction_page(): return render_template('prediction.html')

@app.route('/visualization')
@login_required
def visualization_page():
    return render_template('visualization.html',
                           model_comparison_chart=create_model_comparison_chart(),
                           rf_importance_chart=create_feature_importance_chart('random_forest'),
                           xgb_importance_chart=create_feature_importance_chart('xgboost'),
                           gauge_charts=create_gauge_charts())

@app.route('/about')
@login_required
def about_page(): return render_template('about.html')

# ============================================================
#  API ENDPOINTS
# ============================================================
@app.route('/api/config', methods=['GET'])
def api_config():
    return jsonify({'fake_label': 1, 'genuine_label': 0, 'source': 'Text + ML Rules', 'feature_names': FEATURE_NAMES})

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json() or {}
        rating = float(data.get('rating', 5))
        sentiment = float(data.get('sentiment_score', 0))
        votes = float(data.get('helpfulVoteCount', 0))
        verified = float(data.get('verifiedPurchase', 0))
        length = float(data.get('review_length', 0))
        words = float(data.get('word_count', 0))
        review_text = data.get('review_text', '')

        is_fake, avg_prob = predict_logic(rating, sentiment, votes, verified, length, words, review_text)
        
        fake_pct = avg_prob * 100.0
        if not is_fake:
            fake_pct = 100.0 - fake_pct 
        conf_pct = fake_pct if is_fake else (100.0 - fake_pct)
        final_verdict = 'FAKE' if is_fake else 'GENUINE'

        return jsonify({
            'success': True,
            'prediction': final_verdict,
            'confidence': round(conf_pct, 2),
            'fake_probability': round(fake_pct, 2),
            'genuine_probability': round(100.0 - fake_pct, 2),
            'model_predictions': {
                'random_forest': {'prediction': final_verdict, 'confidence': round(conf_pct, 2)},
                'xgboost': {'prediction': final_verdict, 'confidence': round(conf_pct, 2)},
                'logistic_regression': {'prediction': final_verdict, 'confidence': round(conf_pct, 2)},
            },
            'visualization': create_prediction_result_chart(is_fake, avg_prob)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analyze_text', methods=['POST'])
@login_required
def analyze_text():
    text = (request.get_json() or {}).get('review_text', '')
    pos = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'awesome', 'fantastic', 'wonderful', 'quality', 'outstanding', 'unbelievable']
    neg = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 'horrible', 'disappointing', 'waste', 'useless', 'garbage']
    t = text.lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    score = 0.0 if (p + n) == 0 else (p - n) / (p + n)
    
    return jsonify({'success': True, 'features': {
        'review_length': len(text),
        'word_count': len(text.split()),
        'sentiment_score': score
    }})

# ============================================================
#  BATCH UPLOAD ENDPOINT (UPDATED)
# ============================================================
@app.route('/api/upload_excel', methods=['POST'])
@login_required
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(filepath)
            except ImportError as e:
                # Catch the missing openpyxl/xlrd dependency gracefully
                if 'openpyxl' in str(e).lower() or 'xlrd' in str(e).lower():
                    return jsonify({
                        'success': False, 
                        'error': 'Missing dependency! Please run: pip install openpyxl in your terminal, then restart the app.'
                    }), 400
                raise e
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type. Use .xlsx/.xls/.csv'}), 400

        # Normalize column names
        alias_map = {
            'rating': 'rating', 
            'sentiment_score': 'sentiment_score',
            'sentimentscore': 'sentiment_score',
            'helpful_votes': 'helpfulVoteCount', 
            'helpfulvotes': 'helpfulVoteCount',
            'helpfulvotecount': 'helpfulVoteCount',
            'verified_purchase': 'verifiedPurchase', 
            'verifiedpurchase': 'verifiedPurchase',
            'review_length': 'review_length', 
            'reviewlength': 'review_length',
            'word_count': 'word_count', 
            'wordcount': 'word_count',
            'review_text': 'review_text',
            'text': 'review_text'
        }
        df.columns = [alias_map.get(str(c).lower().strip(), str(c).lower().strip()) for c in df.columns]

        required = ['rating', 'sentiment_score', 'helpfulVoteCount', 'verifiedPurchase', 'review_length', 'word_count']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'success': False, 'error': f'Missing columns: {", ".join(missing)}'}), 400

        results = []
        for idx, row in df.iterrows():
            try:
                rating = float(row['rating'])
                sentiment = float(row['sentiment_score'])
                votes = float(row['helpfulVoteCount'])
                verified = float(row['verifiedPurchase'])
                length = float(row['review_length'])
                words = float(row['word_count'])
                text = str(row.get('review_text', '')) if 'review_text' in row and pd.notna(row['review_text']) else ''
            except (ValueError, TypeError) as ve:
                results.append({
                    'row': int(idx) + 1,
                    'rating': row.get('rating', ''),
                    'sentiment_score': row.get('sentiment_score', ''),
                    'prediction': 'ERROR',
                    'confidence': 0,
                    'fake_probability': 0,
                    'model_predictions': {
                        'random_forest': {'prediction': 'N/A'},
                        'xgboost': {'prediction': 'N/A'},
                        'logistic_regression': {'prediction': 'N/A'}
                    },
                    'error': str(ve)
                })
                continue

            is_fake, avg_prob = predict_logic(rating, sentiment, votes, verified, length, words, text)
            
            fake_pct = avg_prob * 100.0
            if not is_fake:
                fake_pct = 100.0 - fake_pct
            conf_pct = fake_pct if is_fake else (100.0 - fake_pct)
            final_verdict = 'FAKE' if is_fake else 'GENUINE'

            results.append({
                'row': int(idx) + 1,
                'rating': rating,
                'sentiment_score': sentiment,
                'prediction': final_verdict,
                'confidence': round(conf_pct, 2),
                'fake_probability': round(fake_pct, 2),
                'model_predictions': {
                    'random_forest': {'prediction': final_verdict, 'confidence': round(conf_pct, 2)},
                    'xgboost': {'prediction': final_verdict, 'confidence': round(conf_pct, 2)},
                    'logistic_regression': {'prediction': final_verdict, 'confidence': round(conf_pct, 2)},
                }
            })

        total = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        genuine_count = sum(1 for r in results if r['prediction'] == 'GENUINE')

        return jsonify({
            'success': True,
            'total': total,
            'fake_count': fake_count,
            'genuine_count': genuine_count,
            'results': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

# ============================================================
#  CHART CREATION
# ============================================================
def create_prediction_result_chart(is_fake, prob):
    fake_val = prob * 100 if is_fake else (1 - prob) * 100
    gen_val = 100 - fake_val
    fig = go.Figure(go.Bar(x=['FAKE', 'GENUINE'], y=[fake_val, gen_val], marker_color=['#dc3545', '#28a745']))
    fig.update_layout(title='Probability Breakdown', yaxis=dict(range=[0, 100]), template='plotly_white', height=400)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_model_comparison_chart():
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, model in enumerate(model_performance.index.tolist()):
        fig.add_trace(go.Bar(name=model, x=metrics, y=model_performance.loc[model, metrics].values, marker_color=colors[i % len(colors)]))
    fig.update_layout(title='Model Performance', barmode='group', template='plotly_white', height=400)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_feature_importance_chart(model_type):
    importance = rf_model.feature_importances_ if model_type == 'random_forest' else xgb_model.feature_importances_
    color = '#3498db' if model_type == 'random_forest' else '#2ecc71'
    title = 'Random Forest' if model_type == 'random_forest' else 'XGBoost'
    fig = go.Figure(go.Bar(x=importance, y=FEATURE_NAMES, orientation='h', marker_color=color))
    fig.update_layout(title=f'{title} Feature Importance', template='plotly_white', height=350, margin=dict(l=150))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_gauge_charts():
    best = model_performance['F1-Score'].idxmax()
    metrics = model_performance.loc[best]
    out = []
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=metrics[metric] * 100, title={'text': metric},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2ecc71"}}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        out.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    return out

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)