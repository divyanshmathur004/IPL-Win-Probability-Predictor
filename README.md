# 🏏 IPL Win Probability Predictor

A machine learning web application that predicts the real-time win probability of an IPL cricket match using Logistic Regression.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url-here)

## 📊 Features

- **Real-time Predictions**: Get win probability after every over
- **User-friendly Interface**: Easy-to-use Streamlit web app
- **Comprehensive Validation**: Handles edge cases and invalid inputs
- **Visual Insights**: Color-coded results and match situation analysis
- **Team Mapping**: Supports both old and new IPL team names
- **Modern Teams**: Includes Gujarat Titans and Lucknow Super Giants

## 🚀 Quick Start

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/divyanshmathur004/IPL-Win-Probability-Predictor.git
   cd IPL-Win-Probability-Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud (Free)

### Step 1: Push to GitHub
Make sure all these files are in your repo:
- ✅ `app.py`
- ✅ `pipe.pkl`
- ✅ `requirements.txt`
- ✅ `.streamlit/config.toml`
- ✅ `README.md`

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - **Repository**: `divyanshmathur004/IPL-Win-Probability-Predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

Your app will be live in 2-3 minutes! 🎉

## 📁 Project Structure

```
IPL-Win-Probability-Predictor/
│
├── app.py                      # Streamlit web application
├── pipe.pkl                    # Trained ML model (Logistic Regression)
├── requirements.txt            # Python dependencies
├── ipl_data_train.ipynb       # Model training notebook
├── matches.csv                # IPL matches dataset
├── deliveries.csv             # Ball-by-ball dataset
├── README.md                  # This file
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## 🎯 How It Works

### Input Features
The model takes 9 features as input:
1. **Batting Team** - Team chasing the target
2. **Bowling Team** - Team defending the target
3. **City** - Venue where match is being played
4. **Target** - Total runs to chase
5. **Current Score** - Runs scored so far
6. **Overs Completed** - Overs bowled
7. **Wickets Lost** - Wickets fallen

### Derived Features (Calculated Automatically)
- **Runs Left** = Target - Current Score
- **Balls Left** = 120 - (Overs × 6)
- **Wickets Remaining** = 10 - Wickets Lost
- **Current Run Rate (CRR)** = Score / Overs
- **Required Run Rate (RRR)** = (Runs Left × 6) / Balls Left

### Model Pipeline
```
Input Data → OneHotEncoder (Teams & Cities) → Logistic Regression → Win Probability
```

## 🔧 What's Fixed in This Version

### ✅ Bug Fixes
1. **Team Name Mapping** - Handles old team names (Delhi Daredevils → Delhi Capitals)
2. **New Teams Support** - Gujarat Titans & Lucknow Super Giants supported
3. **Input Validation** - Prevents invalid inputs (same team batting/bowling, negative values)
4. **Edge Case Handling** - Handles match-over scenarios (all out, target achieved)
5. **Division by Zero** - Fixed crashes when overs = 0 or balls_left = 0
6. **Error Handling** - Graceful error messages instead of crashes

### ✅ Improvements
1. **Better UI** - Color-coded results, progress bars, match insights
2. **Responsive Design** - Clean layout with proper spacing
3. **Real-time Validation** - Instant feedback on invalid inputs
4. **Match Insights** - AI-powered suggestions based on match situation
5. **Model Caching** - Faster load times with `@st.cache_resource`

## 📊 Model Performance

- **Algorithm**: Logistic Regression
- **Accuracy**: ~80.7%
- **Training Data**: IPL matches (2008-2020)
- **Features**: 52 (after one-hot encoding)

## 🎮 Usage Example

1. **Select Teams**: Choose batting and bowling teams
2. **Choose Venue**: Select the city where match is being played
3. **Enter Target**: Input the target score to chase
4. **Match State**: Enter current score, overs, and wickets
5. **Predict**: Click "Predict Win Probability" button
6. **Results**: View win probability for both teams with insights

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** - Web app framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **NumPy** - Numerical computing

## 📝 Model Training

The model was trained using:
- **Dataset**: Kaggle IPL Dataset (2008-2020)
- **Algorithm**: Logistic Regression with L2 regularization
- **Preprocessing**: One-Hot Encoding for categorical features
- **Validation**: Train-test split (80-20)

To retrain the model, run:
```bash
jupyter notebook ipl_data_train.ipynb
```

## ⚠️ Known Limitations

1. Model trained on historical data (2008-2020) - doesn't include recent seasons
2. Doesn't account for:
   - Player form and injuries
   - Pitch conditions
   - Weather factors (dew)
   - Match importance (playoffs vs league)
3. New teams (Gujarat Titans, Lucknow Super Giants) are mapped to similar existing teams

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Divyansh Mathur**
- GitHub: [@divyanshmathur004](https://github.com/divyanshmathur004)

## 🙏 Acknowledgments

- Dataset: [Kaggle IPL Dataset](https://www.kaggle.com/ramjidoolla/ipl-data-set)
- Inspiration: CampusX IPL Win Predictor Tutorial

---

**Made with ❤️ using Streamlit & Machine Learning**
