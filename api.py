from flask import Flask, request, jsonify, make_response
from flask.logging import create_logger
from flask_limiter import Limiter
from flask_cors import CORS
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields, ValidationError
import joblib
import pandas as pd

mlpipeline = Flask(__name__)
CORS(mlpipeline)
log = create_logger(mlpipeline)
limiter = Limiter(
    get_remote_address,
    app=mlpipeline,
    default_limits=["2 per minute", "1 per second"]
)

# Load the trained model
model = joblib.load("california_rf_model.pkl")
scaler = joblib.load('scaler.pkl')


class PredictionSchema(Schema):
    MedInc = fields.Float(required=True)
    HouseAge = fields.Float(required=True)
    AveRooms = fields.Float(required=True)
    AveBedrms = fields.Float(required=True)
    Population = fields.Float(required=True)
    AveOccup = fields.Float(required=True)
    Latitude = fields.Float(required=True)
    Longitude = fields.Float(required=True)


@mlpipeline.route('/predict', methods=['POST'])
@limiter.limit("1 per second")
def predict():
    schema = PredictionSchema()
    try:
        # Validate the request data
        data = schema.load(request.get_json(force=True))
    except ValidationError as e:
        return make_response(jsonify({'error': 'Invalid request data', 'details': e.messages}), 400)

    try:
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame([data])

        # Scale the input data
        scaled_data = scaler.transform(df)

        # Make the prediction
        prediction = model.predict(scaled_data)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Log the error and return a 500 response
        log.error(f"An error occurred: {e}")
        return make_response(jsonify({'error': 'An error occurred, please try again later'}), 500)


if __name__ == '__main__':
    mlpipeline.run(debug=True)
