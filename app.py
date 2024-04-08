from flask import Flask, request, render_template, redirect, url_for
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import logging

application = Flask(__name__)

app = application

# Configure logging
logging.basicConfig(level=logging.INFO)


#@app.route('/')
#def index():
    #return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Create CustomData instance from form data
            data = CustomData(**request.form)

            # Get data as DataFrame
            pred_df = data.get_data_as_data_frame()

            logging.info("Received input data: %s", pred_df)

            predict_pipeline = PredictPipeline()

            logging.info("Making prediction...")
            results = predict_pipeline.predict(pred_df)

            logging.info("Prediction results: %s", results)

            # If you expect multiple results, handle them accordingly
            result = results[0] if results else None

            return render_template('home.html', result=result)

        except CustomException as ce:
            # Log custom exceptions
            logging.error("Custom exception occurred: %s", ce)
            return redirect(url_for('index', error="An error occurred during prediction."))

        except Exception as e:
            # Log other exceptions
            logging.error("Exception occurred: %s", e)
            return redirect(url_for('index', error="An unexpected error occurred."))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
