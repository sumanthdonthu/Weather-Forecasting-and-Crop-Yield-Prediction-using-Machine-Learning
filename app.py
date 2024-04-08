from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            YEAR=int(request.form.get('YEAR')),
            STATE=request.form.get('STATE'),
            REGION=request.form.get('REGION'),
            PRCP_Apr=float(request.form.get('PRCP Apr')),
            PRCP_Aug=float(request.form.get('PRCP Aug')),
            PRCP_Dec=float(request.form.get('PRCP Dec')),
            PRCP_Feb=float(request.form.get('PRCP Feb')),
            PRCP_Jan=float(request.form.get('PRCP Jan')),
            PRCP_Jul=float(request.form.get('PRCP Jul')),
            PRCP_Jun=float(request.form.get('PRCP Jun')),
            PRCP_Mar=float(request.form.get('PRCP Mar')),
            PRCP_May=float(request.form.get('PRCP May')),
            PRCP_Nov=float(request.form.get('PRCP Nov')),
            PRCP_Oct=float(request.form.get('PRCP Oct')),
            PRCP_Sep=float(request.form.get('PRCP Sep')),
            SNOW_Apr=float(request.form.get('SNOW Apr')),
            SNOW_Aug=float(request.form.get('SNOW Aug')),
            SNOW_Dec=float(request.form.get('SNOW Dec')),
            SNOW_Feb=float(request.form.get('SNOW Feb')),
            SNOW_Jan=float(request.form.get('SNOW Jan')),
            SNOW_Jul=float(request.form.get('SNOW Jul')),
            SNOW_Jun=float(request.form.get('SNOW Jun')),
            SNOW_Mar=float(request.form.get('SNOW Mar')),
            SNOW_May=float(request.form.get('SNOW May')),
            SNOW_Nov=float(request.form.get('SNOW Nov')),
            SNOW_Oct=float(request.form.get('SNOW Oct')),
            SNOW_Sep=float(request.form.get('SNOW Sep')),
            SNWD_Apr=float(request.form.get('SNWD Apr')),
            SNWD_Aug=float(request.form.get('SNWD Aug')),
            SNWD_Dec=float(request.form.get('SNWD Dec')),
            SNWD_Feb=float(request.form.get('SNWD Feb')),
            SNWD_Jan=float(request.form.get('SNWD Jan')),
            SNWD_Jul=float(request.form.get('SNWD Jul')),
            SNWD_Jun=float(request.form.get('SNWD Jun')),
            SNWD_Mar=float(request.form.get('SNWD Mar')),
            SNWD_May=float(request.form.get('SNWD May')),
            SNWD_Nov=float(request.form.get('SNWD Nov')),
            SNWD_Oct=float(request.form.get('SNWD Oct')),
            SNWD_Sep=float(request.form.get('SNWD Sep')),
            TAVG_Apr=float(request.form.get('TAVG Apr')),
            TAVG_Aug=float(request.form.get('TAVG Aug')),
            TAVG_Dec=float(request.form.get('TAVG Dec')),
            TAVG_Feb=float(request.form.get('TAVG Feb')),
            TAVG_Jan=float(request.form.get('TAVG Jan')),
            TAVG_Jul=float(request.form.get('TAVG Jul')),
            TAVG_Jun=float(request.form.get('TAVG Jun')),
            TAVG_Mar=float(request.form.get('TAVG Mar')),
            TAVG_May=float(request.form.get('TAVG May')),
            TAVG_Nov=float(request.form.get('TAVG Nov')),
            TAVG_Oct=float(request.form.get('TAVG Oct')),
            TAVG_Sep=float(request.form.get('TAVG Sep')),
            TMAX_Apr=float(request.form.get('TMAX Apr')),
            TMAX_Aug=float(request.form.get('TMAX Aug')),
            TMAX_Dec=float(request.form.get('TMAX Dec')),
            TMAX_Feb=float(request.form.get('TMAX Feb')),
            TMAX_Jan=float(request.form.get('TMAX Jan')),
            TMAX_Jul=float(request.form.get('TMAX Jul')),
            TMAX_Jun=float(request.form.get('TMAX Jun')),
            TMAX_Mar=float(request.form.get('TMAX Mar')),
            TMAX_May=float(request.form.get('TMAX May')),
            TMAX_Nov=float(request.form.get('TMAX Nov')),
            TMAX_Oct=float(request.form.get('TMAX Oct')),
            TMAX_Sep=float(request.form.get('TMAX Sep')),
            TMIN_Apr=float(request.form.get('TMIN Apr')),
            TMIN_Aug=float(request.form.get('TMIN Aug')),
            TMIN_Dec=float(request.form.get('TMIN Dec')),
            TMIN_Feb=float(request.form.get('TMIN Feb')),
            TMIN_Jan=float(request.form.get('TMIN Jan')),
            TMIN_Jul=float(request.form.get('TMIN Jul')),
            TMIN_Jun=float(request.form.get('TMIN Jun')),
            TMIN_Mar=float(request.form.get('TMIN Mar')),
            TMIN_May=float(request.form.get('TMIN May')),
            TMIN_Nov=float(request.form.get('TMIN Nov')),
            TMIN_Oct=float(request.form.get('TMIN Oct')),
            TMIN_Sep=float(request.form.get('TMIN Sep')),
            COMMODITY=request.form.get('COMMODITY'),
            ACRES_HARVESTED=float(request.form.get('ACRES HARVESTED')),
            ACRES_PLANTED=float(request.form.get('ACRES PLANTED'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
