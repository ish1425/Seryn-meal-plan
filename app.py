from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# Meals and expected pickle filenames
meals = ['Breakfast', 'Lunch', 'Dinner', 'Snacks', 'Drinks']

# Dropbox direct download URLs for each model and mlb file
# Replace these URLs with your actual Dropbox links ending with '?dl=1'
dropbox_urls = {
    'Breakfast_model.pkl': 'https://www.dropbox.com/scl/fi/erj64m9qolqof0leb67ox/Breakfast_model.pkl?rlkey=td4sts0r10tyt0bdk8cykra7j&st=50mi2aff&dl=1',
    'Breakfast_mlb.pkl': 'https://www.dropbox.com/scl/fi/vnci3k737jny0uyzp4cvy/Breakfast_mlb.pkl?rlkey=vrg8x3g0s91x3uridc05aj42q&st=kuwn6jm8&dl=1',
    'Lunch_model.pkl': 'https://www.dropbox.com/scl/fi/3it5ovnczwklt750hshcz/Lunch_model.pkl?rlkey=eqszv5pkm2a5yy0ot14kyjl8h&st=64zc7w83&dl=1',
    'Lunch_mlb.pkl': 'https://www.dropbox.com/scl/fi/ms6oz7pbkc6pohrldcrmn/Lunch_mlb.pkl?rlkey=p9sgchcwmwyqodd0att1svd18&st=p6s7mjy1&dl=1',
    'Dinner_model.pkl': 'https://www.dropbox.com/scl/fi/6kwniu7w4e2lcs8nyekjw/Dinner_model.pkl?rlkey=ayzytyy5lp8w6n0ive9wlkdk3&st=o6b9xgww&dl=1',
    'Dinner_mlb.pkl': 'https://www.dropbox.com/scl/fi/1nltzycthy65bjxfzxzh3/Dinner_mlb.pkl?rlkey=crlcyvi2lgwdv5k5u8s6ju2a9&st=nmu3vhrj&dl=1',
    'Snacks_model.pkl': 'https://www.dropbox.com/scl/fi/dkuvj9h2b9c656uoh6cth/Snacks_model.pkl?rlkey=vh2t62ey5fxm4rxmploaup6hl&st=xxkekly1&dl=1',
    'Snacks_mlb.pkl': 'https://www.dropbox.com/scl/fi/vpr0dcozrsnetl1sr2dt3/Snacks_mlb.pkl?rlkey=jp7oho2axcl7msm1829oblm3e&st=0bnua5il&dl=1',
    'Drinks_model.pkl': 'https://www.dropbox.com/scl/fi/ig7dfaez8e92vy1vherpd/Drinks_model.pkl?rlkey=6z3i4kf7z05upcl5duiu4ali0&st=f7qz630s&dl=1',
    'Drinks_mlb.pkl': 'https://www.dropbox.com/scl/fi/av3exl3tq9dogs017ak1a/Drinks_mlb.pkl?rlkey=2h03ptkj85fxv7pg7loqs04hi&st=vpubiflq&dl=1',
}

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {dest}")
    else:
        print(f"{dest} already exists, skipping download.")

# Download all files
for meal in meals:
    model_filename = f"{meal}_model.pkl"
    mlb_filename = f"{meal}_mlb.pkl"
    download_file(dropbox_urls[model_filename], model_filename)
    download_file(dropbox_urls[mlb_filename], mlb_filename)

# Load models and mlbs
models = {}
mlbs = {}
for meal in meals:
    models[meal] = joblib.load(f'{meal}_model.pkl')
    mlbs[meal] = joblib.load(f'{meal}_mlb.pkl')

# Define feature columns (must match model input)
feature_cols = ['Age', 'BMI', 'Hba1c', 'Calorie Needs', 'Carb Tolerance',
                'Gender_Male', 'Pre Diabetic', 'Type1', 'Type2',
                'Low Activity', 'Moderate Activity',
                'Low Sugar Sensitivity', 'Medium Sugar Sensitivity']

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_features = []
    for col in feature_cols:
        val = data.get(col)
        if val is None:
            return jsonify({'error': f'Missing feature: {col}'}), 400
        user_features.append(float(val))
    
    user_features = np.array(user_features).reshape(1, -1)

    weekly_plan = {}
    for meal in meals:
        model = models[meal]
        mlb = mlbs[meal]
        probas = model.predict_proba(user_features)
        avg_proba = np.array([p[0][1] for p in probas])
        top_7_indices = np.argsort(avg_proba)[-7:][::-1]
        top_7_labels = [mlb.classes_[i] for i in top_7_indices]
        weekly_plan[meal] = top_7_labels

    return jsonify(weekly_plan)

if __name__ == '__main__':
    app.run(debug=True)
