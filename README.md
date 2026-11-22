# Phishing URL Detector

Flask + scikit-learn. Extracts lexical features from a URL, scores it 0-100.

## Run
    pip install -r requirements.txt
    python train_model.py --data data/urls.csv    # or --demo for a smoke test
    python app.py                                  # http://127.0.0.1:5000

## Train in Google Colab
Upload `features.py`, `train_model.py` and your CSV, run `!python train_model.py --data urls.csv`,
then download `models/phishing_rf.joblib` into this project's `models/` folder.

## API (for a browser extension)
    curl -X POST localhost:5000/api/check -H "Content-Type: application/json" -d '{"url":"http://bit.ly/abc"}'
