import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide")

st.title("🏏 IPL Analytics & Match Winner Prediction Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

# -----------------------------
# Merge Team Name Variations
# -----------------------------

team_name_corrections = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant"
}

# Fix matches dataset
for column in ['team1', 'team2', 'winner', 'toss_winner']:
    matches[column] = matches[column].replace(team_name_corrections)

# Fix deliveries dataset
for column in ['batting_team', 'bowling_team']:
    deliveries[column] = deliveries[column].replace(team_name_corrections)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔎 Filters")

seasons = sorted(matches['season'].dropna().unique())
season_option = st.sidebar.selectbox("Select Season", ["All Seasons"] + list(seasons))

if season_option != "All Seasons":
    filtered_matches = matches[matches['season'] == season_option]
    match_ids = filtered_matches['id']
    filtered_deliveries = deliveries[deliveries['match_id'].isin(match_ids)]
else:
    filtered_matches = matches
    filtered_deliveries = deliveries

# -----------------------------
# KPI Section
# -----------------------------
st.markdown("## 📊 Season Overview")

col1, col2, col3, col4 = st.columns(4)

total_matches = len(filtered_matches)
total_runs = filtered_deliveries['total_runs'].sum()
total_sixes = len(filtered_deliveries[filtered_deliveries['batsman_runs'] == 6])
total_fours = len(filtered_deliveries[filtered_deliveries['batsman_runs'] == 4])

col1.metric("Total Matches", total_matches)
col2.metric("Total Runs", total_runs)
col3.metric("Total Sixes", total_sixes)
col4.metric("Total Fours", total_fours)

st.markdown("---")

# Filter final matches
final_matches = matches[matches['match_type'] == "Final"]

if season_option != "All Seasons":
    st.subheader("🏆 IPL Season Winner")
    champion_row = final_matches[final_matches['season'] == season_option]

    if not champion_row.empty:
        champion = champion_row['winner'].values[0]
        st.success(f"🏆 {champion}")
        st.markdown("---")
    else:
        st.warning("Champion data not available for this season.")

else:
    st.subheader("🏆 IPL Winners by Season")

    final_matches = matches[matches['match_type'] == "Final"]
    season_winners = final_matches[['season', 'winner']]

    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(data=season_winners, y='winner', order=season_winners['winner'].value_counts().index, ax=ax)

    ax.set_title("Total Titles Won")
    ax.set_xlabel("Titles")
    ax.set_ylabel("Team")

    st.pyplot(fig)

    st.markdown("---")


# -----------------------------
# Most Successful Teams
# -----------------------------
st.subheader("🏆 Most Successful Teams")

wins = filtered_matches['winner'].value_counts().reset_index()
wins.columns = ['Team', 'Wins']

fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=wins.head(10), x='Wins', y='Team', ax=ax)

ax.set_title("Top 10 Winning Teams")
ax.set_xlabel("Wins")
ax.set_ylabel("Team")

st.pyplot(fig)


st.markdown("---")


# -----------------------------
# Orange Cap
# -----------------------------
st.subheader("Top Run Scorer")

orange_cap = (
    filtered_deliveries
    .groupby('batter')['batsman_runs']
    .sum()
    .sort_values(ascending=False)
    .head(1)
)

if not orange_cap.empty:
    st.success(f" {orange_cap.index[0]} - {orange_cap.values[0]} Runs")

st.markdown("---")


# -----------------------------
# Purple Cap
# -----------------------------
st.subheader("Highest Wicket Taker")

valid_wickets_pc = filtered_deliveries[
    filtered_deliveries['dismissal_kind'].notna() &
    (~filtered_deliveries['dismissal_kind'].isin(
        ['run out', 'retired hurt', 'obstructing the field']
    ))
]

purple_cap = (
    valid_wickets_pc
    .groupby('bowler')
    .size()
    .sort_values(ascending=False)
)

if not purple_cap.empty:
    top_bowler = purple_cap.index[0]
    wickets = purple_cap.iloc[0]

    st.success(f" {top_bowler} - {wickets} Wickets")
else:
    st.warning("No wicket data available.")

st.markdown("---")


# -----------------------------
# Top Run Scorers
# -----------------------------
st.subheader("🏏 Top 10 Run Scorers")

top_batsmen = (
    filtered_deliveries
    .groupby('batter')['batsman_runs']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=top_batsmen, x='batsman_runs', y='batter', ax=ax)

ax.set_title("Top 10 Run Scorers")
ax.set_xlabel("Runs")
ax.set_ylabel("Player")

st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Top 10 Wicket Takers
# -----------------------------
st.subheader("🎯 Top 10 Wicket Takers")

valid_wickets = filtered_deliveries[
    filtered_deliveries['dismissal_kind'].notna() &
    (~filtered_deliveries['dismissal_kind'].isin(
        ['run out', 'retired hurt', 'obstructing the field']
    ))
]

top_10_bowlers = (
    valid_wickets
    .groupby('bowler')
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name='Wickets')
)

fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=top_10_bowlers, x='Wickets', y='bowler', ax=ax)

ax.set_title("Top 10 Wicket Takers")
ax.set_xlabel("Wickets")
ax.set_ylabel("Bowler")

st.pyplot(fig)


st.markdown("---")

# -----------------------------
# Venue Analysis
# -----------------------------
st.subheader("🏟 Matches by Venue")

venue_counts = (
    filtered_matches['venue']
    .value_counts()
    .head(10)
    .reset_index()
)

venue_counts.columns = ['Venue', 'Matches']

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=venue_counts, x='Matches', y='Venue', ax=ax)

ax.set_title("Top Venues by Matches")
ax.set_xlabel("Number of Matches")
ax.set_ylabel("Venue")

st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Match Winner Prediction
# -----------------------------
st.subheader("🤖 Match Winner Prediction with Probability")

teams = sorted(matches['team1'].dropna().unique())

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

if team1 != team2 and st.button("Predict Winner"):

    model_data = matches[['team1', 'team2', 'winner']].dropna()
    model_data = pd.get_dummies(model_data)

    X = model_data.drop(columns=[col for col in model_data.columns if 'winner_' in col])
    y = model_data[[col for col in model_data.columns if 'winner_' in col]]
    y = y.idxmax(axis=1)

    model = RandomForestClassifier()
    model.fit(X, y)

    input_df = pd.DataFrame([[team1, team2]], columns=['team1', 'team2'])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    prob_df = pd.DataFrame({
        "Team": classes,
        "Win Probability": probabilities
    }).sort_values(by="Win Probability", ascending=False)

    predicted_winner = prob_df.iloc[0]['Team'].replace("winner_", "")

    st.success(f"🏆 Predicted Winner: {predicted_winner}")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=prob_df, x="Win Probability", y="Team", ax=ax)

    ax.set_title("Win Probability")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Team")

    st.pyplot(fig)


elif team1 == team2:
    st.warning("Please select two different teams.")

st.markdown("---")
st.markdown("Made with ❤️ by SHRIKRISHNA BADAVE")
