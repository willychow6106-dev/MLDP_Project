import joblib
import streamlit as st
import numpy as np
import pandas as pd

# -------------------------
# Load trained model + columns
# -------------------------
model = joblib.load("final_rf_model.pkl")
model_cols = joblib.load("final_model_columns.pkl")

# -------------------------
# Streamlit app
# -------------------------
st.title("Steam Game Price Prediction (Random Forest)")

st.write("Fill in the game details below and click Predict.")

# -------------------------
# Define input options (similar to your class practice)
# -------------------------
yes_no = [0, 1]
month_options = list(range(1, 13))

# Keep genre/category options SMALL + readable (you can add more later)
genre_options = [
    "Indie", "Casual", "Action", "Adventure", "RPG", "Simulation", "Strategy", "Sports", "Racing"
]

category_options = [
    "Single-player", "Multi-player", "Co-op", "Online PvP", "Steam Achievements",
    "Steam Cloud", "Full controller support", "Partial Controller Support", "Steam Trading Cards"
]

# -------------------------
# User inputs
# -------------------------
release_year = st.number_input("Release Year", min_value=1970, max_value=2035, value=2020)
release_month = st.selectbox("Release Month", month_options, index=5)

is_free = st.selectbox("Is Free?", yes_no, index=0)
has_multiplayer = st.selectbox("Has Multiplayer?", yes_no, index=0)
has_achievements = st.selectbox("Has Achievements?", yes_no, index=1)

# Optional: allow selecting one main genre/category (simple like class)
main_genre = st.selectbox("Main Genre (optional)", ["None"] + genre_options, index=0)
main_category = st.selectbox("Main Category (optional)", ["None"] + category_options, index=0)

# Numeric inputs
num_genres = st.slider("Number of Genres", min_value=0, max_value=20, value=3)
num_categories = st.slider("Number of Categories", min_value=0, max_value=50, value=5)

# Developer / Publisher (optional text like your dataset)
developer = st.text_input("Developer (optional)", value="Other")
publisher = st.text_input("Publisher (optional)", value="Other")

# -------------------------
# Predict button
# -------------------------
if st.button("Predict Game Price"):

    # Build 1-row input dataframe (same approach as your class practice)
    df_input = pd.DataFrame({
        "release_year": [release_year],
        "release_month": [release_month],
        "is_free": [is_free],
        "has_multiplayer": [has_multiplayer],
        "has_achievements": [has_achievements],
        "num_genres": [num_genres],
        "num_categories": [num_categories],
        "developer": [developer],
        "publisher": [publisher],
    })

    # If your training data has these engineered columns, we can add them too
    # (safe: only add if they exist in model_cols)
    if "genres_x_categories" in model_cols:
        df_input["genres_x_categories"] = df_input["num_genres"] * df_input["num_categories"]

    # Add optional selected genre/category into the one-hot space
    # Your training columns looked like "genre_Indie" and "cat_Single-player"
    if main_genre != "None":
        df_input[f"genre_{main_genre}"] = 1

    if main_category != "None":
        # Match your naming style from dataset: cat_Single-player etc.
        df_input[f"cat_{main_category}"] = 1

    # One-hot encode developer/publisher if user typed new values
    # (This mirrors your notebook preprocessing style)
    df_input = pd.get_dummies(df_input, columns=["developer", "publisher"], drop_first=True)

    # Align to training columns (MOST IMPORTANT step)
    df_input = df_input.reindex(columns=model_cols, fill_value=0)

    # Predict is in log1p(price), convert back to price
    pred_log = model.predict(df_input)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"Predicted Game Price: ${pred_price:,.2f}")

