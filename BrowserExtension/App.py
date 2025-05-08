from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import Extractor

app = Flask(__name__)
CORS(app) 

model = joblib.load('svm_linear_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'html' not in data:
            return jsonify({'error': 'No HTML provided'}), 400

        html_to_analyze = data['html']
        feature_vector = Extractor.ExtractFeatures(html_to_analyze)
        
        if feature_vector.empty:
            return jsonify({'error': 'Feature extraction failed'}), 500

        prediction = model.predict(feature_vector)
        readable_prediction = map_prediction_to_label(prediction[0])

        print(readable_prediction)
        return jsonify({'prediction': readable_prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def map_prediction_to_label(numeric_label):
    label_map = {
        0: 'Adult',
        1: 'Computers',
        2: 'Games',
        3: 'Health',
        4: 'News',
        5: 'Recreation',
        6: 'Reference',
        7: 'Science',
        8: 'Shopping',
        9: 'Society',
        10: 'Sports'
    }
    return label_map.get(numeric_label, 'Unknown')

def make_prediction(feature_vector):
    return model.predict(feature_vector)

if __name__ == '__main__':
    app.run(debug=True)