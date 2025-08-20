from flask import Flask, render_template, request  
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model & pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Load housing to get feature names & types
data = pd.read_csv("housing.csv")
features_df = data.drop("median_house_value", axis=1)

# Split numeric vs categorical
num_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()

# For this dataset we expect "ocean_proximity" as categorical
cat_choices = {}
for col in cat_cols:
    cat_choices[col] = sorted([str(x) for x in features_df[col].dropna().unique()])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = {}

    if request.method == "POST":
        form_data = {}
        # numeric fields
        for col in num_cols:
            val = request.form.get(col, "").strip()
            try:
                form_data[col] = float(val)
            except ValueError:
                errors[col] = "Enter a valid number"
                form_data[col] = None

        # categorical fields
        for col in cat_cols:
            form_data[col] = request.form.get(col)

        if not errors:
            df = pd.DataFrame([form_data])
            transformed = pipeline.transform(df)
            pred = model.predict(transformed)[0]
            prediction = round(float(pred), 2)

    return render_template(
        "index.html",
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_choices=cat_choices,
        prediction=prediction,
        errors=errors
    )

if __name__ == "__main__":
    app.run(debug=True)
