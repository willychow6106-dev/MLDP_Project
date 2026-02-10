import joblib
import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# Load model + expected columns
# -----------------------------
model = joblib.load("final_rf_model.pkl")
model_cols = joblib.load("final_model_columns.pkl")

st.title("Steam Game Price Prediction (Random Forest)")
st.write("Enter game details and click Predict.")

# -----------------------------
# Helper: safely set a column to 1 if it exists
# -----------------------------
def set_if_exists(df, colname, value=1):
    if colname in df.columns:
        df[colname] = value

# -----------------------------
# User inputs (simple + aligned)
# -----------------------------
release_year = st.number_input("Release Year", min_value=1970, max_value=2035, value=2020)
release_month = st.selectbox("Release Month", list(range(1, 13)), index=5)

is_free = st.selectbox("Is Free?", [0, 1], index=0)

num_genres = st.slider("Number of Genres", min_value=0, max_value=20, value=3)
num_categories = st.slider("Number of Categories", min_value=0, max_value=50, value=5)

# These options match your training column naming style (genre_*, cat_*)
genre_options = [
    "Indie", "Casual", "Action", "Adventure", "RPG",
    "Simulation", "Strategy", "Sports", "Racing"
]
category_options = [
    "Single-player", "Multi-player", "Co-op", "Online PvP",
    "Steam Achievements", "Steam Cloud", "Full controller support",
    "Partial Controller Support", "Steam Trading Cards"
]

main_genre = st.selectbox("Main Genre (optional)", ["None"] + genre_options, index=0)
main_category = st.selectbox("Main Category (optional)", ["None"] + category_options, index=0)

developer = st.text_input("Developer (optional)", value="Other")
publisher = st.text_input("Publisher (optional)", value="Other")

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Game Price"):

    # Rule-based output: if free, show $0.00 (more intuitive for users)
    if is_free == 1:
        st.success("Predicted Game Price: $0.00 (Free-to-play)")
    else:
        # Build 1-row dataframe with base numeric + text fields
        df_input = pd.DataFrame([{
            "release_year": release_year,
            "release_month": release_month,
            "is_free": is_free,
            "num_genres": num_genres,
            "num_categories": num_categories,
            "developer": developer,
            "publisher": publisher
        }])

        # Optional engineered interaction (only if your model trained with it)
        if "genres_x_categories" in model_cols:
            df_input["genres_x_categories"] = df_input["num_genres"] * df_input["num_categories"]

        # One-hot for developer/publisher (matches your preprocessing idea)
        df_input = pd.get_dummies(df_input, columns=["developer", "publisher"], drop_first=True)

        # Add chosen genre/category as flags (only if those features exist in training)
        if main_genre != "None":
            chosen_genre_col = f"genre_{main_genre}"
            set_if_exists(df_input, chosen_genre_col, 1)

        if main_category != "None":
            chosen_cat_col = f"cat_{main_category}"
            set_if_exists(df_input, chosen_cat_col, 1)

        # Align to training columns (critical)
        df_input = df_input.reindex(columns=model_cols, fill_value=0)

        # Predict in log1p space, convert back
        pred_log = model.predict(df_input)[0]
        pred_price = np.expm1(pred_log)

        st.success(f"Predicted Game Price: ${pred_price:,.2f}")

# Optional debug (uncomment if you want to verify columns)
# st.write("Model expects", len(model_cols), "features")
