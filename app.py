import streamlit as st
import pickle
import pandas as pd

# Team name mapping to handle variations and old names
TEAM_MAPPING = {
    # Current teams
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
    'Mumbai Indians': 'Mumbai Indians',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Kings XI Punjab': 'Kings XI Punjab',
    'Chennai Super Kings': 'Chennai Super Kings',
    'Rajasthan Royals': 'Rajasthan Royals',
    'Delhi Capitals': 'Delhi Capitals',
    
    # New teams - map to similar existing teams for prediction
    'Gujarat Titans': 'Rajasthan Royals',  # Similar team profile
    'Lucknow Super Giants': 'Kings XI Punjab',  # Similar team profile
    
    # Old names - map to current names
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
}

# All teams to show in dropdown (including new teams)
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

# Cities - updated to match model's expected values
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('pipe.pkl', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Model file 'pipe.pkl' not found! Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

pipe = load_model()

# Page config
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# Title and description
st.title("🏏 IPL Win Predictor")
st.markdown("### Predict real-time win probability during an IPL match")
st.markdown("---")

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Validate teams are different
if batting_team == bowling_team:
    st.error("⚠️ Batting and bowling teams must be different!")
    st.stop()

# City selection
selected_city = st.selectbox("Select host city", sorted(cities))

# Target score
target = st.number_input('Target Score', min_value=1, max_value=400, value=180, step=1)

# Match state inputs
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, max_value=400, value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=19.5, value=0.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('Wickets Lost', min_value=0, max_value=10, value=0, step=1)

st.markdown("---")

# Prediction button
if st.button("🎯 Predict Win Probability", use_container_width=True):
    
    # Validation checks
    if overs == 0:
        st.warning("⚠️ Match just started! Need at least 0.1 overs to make a prediction.")
        st.stop()
    
    if wickets >= 10:
        st.info(f"🏏 **{bowling_team}** won! All wickets fallen.")
        st.stop()
    
    if score >= target:
        st.success(f"🎉 **{batting_team}** won! Target achieved!")
        st.stop()
    
    # Calculate match metrics
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    
    if balls_left <= 0:
        st.info(f"🏏 **{bowling_team}** won! Innings completed.")
        st.stop()
    
    wickets_remaining = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left
    
    # Map team names to model's expected names
    batting_team_mapped = TEAM_MAPPING.get(batting_team, batting_team)
    bowling_team_mapped = TEAM_MAPPING.get(bowling_team, bowling_team)
    
    # Show match situation
    st.markdown("### 📊 Current Match Situation")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Runs Required", runs_left)
    with col_b:
        st.metric("Balls Remaining", balls_left)
    with col_c:
        st.metric("Wickets Left", wickets_remaining)
    
    col_d, col_e = st.columns(2)
    with col_d:
        st.metric("Current Run Rate", f"{crr:.2f}")
    with col_e:
        st.metric("Required Run Rate", f"{rrr:.2f}")
    
    st.markdown("---")
    
    # Create input dataframe
    input_df = pd.DataFrame({
        'batting_team': [batting_team_mapped],
        'bowling_team': [bowling_team_mapped],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    # Make prediction
    try:
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        
        # Display results
        st.markdown("### 🎯 Win Probability")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #00a65a; border-radius: 10px;'>
                <h2 style='color: white; margin: 0;'>{batting_team}</h2>
                <h1 style='color: white; margin: 10px 0; font-size: 3em;'>{round(win * 100)}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_y:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #dd4b39; border-radius: 10px;'>
                <h2 style='color: white; margin: 0;'>{bowling_team}</h2>
                <h1 style='color: white; margin: 10px 0; font-size: 3em;'>{round(loss * 100)}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(win)
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 💡 Match Insights")
        
        if rrr > crr + 3:
            st.warning(f"⚠️ Required run rate is significantly higher than current run rate. {batting_team} needs to accelerate!")
        elif rrr > crr:
            st.info(f"📈 {batting_team} needs to slightly increase the scoring rate.")
        else:
            st.success(f"✅ {batting_team} is ahead of the required rate!")
        
        if wickets_remaining <= 3:
            st.warning(f"⚠️ Only {wickets_remaining} wickets remaining. Batting team needs to be careful!")
        
        if balls_left <= 30:
            st.info(f"⏰ Only {balls_left} balls remaining! Every ball counts now.")
    
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {str(e)}")
        st.error("This might be due to team/city name mismatch with training data.")
        st.info("💡 Tip: Try different team or city combinations.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Made with ❤️ using Streamlit & Machine Learning</p>
    <p style='font-size: 0.8em;'>Model trained on IPL historical data (2008-2020)</p>
</div>
""", unsafe_allow_html=True)
