import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("(2)_rf_churn_model.pkl")

model = load_model()

# App Title
st.title("🎮 League of Legends Churn Predictor")

st.markdown("Fill out the player stats below to predict if the player is likely to **churn** or stay active.")

# Input Fields
LP = st.number_input("League Points (LP)", value=0)
LEVEL = st.number_input("Player Level", value=1)
AVSCORE = st.number_input("Average Score", value=0.0)
WIN_PER = st.number_input("Overall Win %", value=0.0)
LAST20_WIN_PER = st.number_input("Last 20 Matches Win %", value=0.0)
TOTAL_MATCH = st.number_input("Total Matches Played", value=0)
LOSING_STREAK = st.number_input("Losing Streak Count", value=0)
NO_CHAMPIONS_PLAYED = st.number_input("Champions Played", value=0)
TIER_ENCODED = st.selectbox("Tier", options=[3, 4, 5, 6, 7, 8, 9], format_func=lambda x: {
    3: "Gold", 4: "Platinum", 5: "Emerald", 6: "Diamond",
    7: "Master", 8: "Grandmaster", 9: "Challenger"
}[x])
NO_TEAM_GAMES = st.number_input("Team Games", value=0)
NO_TEAM_PARTICIPANTS = st.number_input("Team Participants", value=0)
TEAM_WIN = st.number_input("Team Wins", value=0)
TEAM_LOSE = st.number_input("Team Losses", value=0)

# Predict Button
if st.button("Predict Churn"):
    input_data = np.array([[LP, LEVEL, AVSCORE, WIN_PER, LAST20_WIN_PER, TOTAL_MATCH, LOSING_STREAK,
                            NO_CHAMPIONS_PLAYED, TIER_ENCODED, NO_TEAM_GAMES,
                            NO_TEAM_PARTICIPANTS, TEAM_WIN, TEAM_LOSE]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Prediction: The player is likely to CHURN.")
    else:
        st.success("✅ Prediction: The player is likely to STAY ACTIVE.")
