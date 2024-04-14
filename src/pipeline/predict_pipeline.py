import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            logging.info("Before Loading")  # Add logging before loading objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("After Loading")  # Add logging after loading objects
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            logging.error(f"Exception occurred: {e}")  # Log the exception
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 YEAR: int,
                 STATE: str,
                 REGION: str,
                 PRCP_Apr: float,
                 PRCP_Aug: float,
                 PRCP_Dec: float,
                 PRCP_Feb: float,
                 PRCP_Jan: float,
                 PRCP_Jul: float,
                 PRCP_Jun: float,
                 PRCP_Mar: float,
                 PRCP_May: float,
                 PRCP_Nov: float,
                 PRCP_Oct: float,
                 PRCP_Sep: float,
                 SNOW_Apr: float,
                 SNOW_Aug: float,
                 SNOW_Dec: float,
                 SNOW_Feb: float,
                 SNOW_Jan: float,
                 SNOW_Jul: float,
                 SNOW_Jun: float,
                 SNOW_Mar: float,
                 SNOW_May: float,
                 SNOW_Nov: float,
                 SNOW_Oct: float,
                 SNOW_Sep: float,
                 SNWD_Apr: float,
                 SNWD_Aug: float,
                 SNWD_Dec: float,
                 SNWD_Feb: float,
                 SNWD_Jan: float,
                 SNWD_Jul: float,
                 SNWD_Jun: float,
                 SNWD_Mar: float,
                 SNWD_May: float,
                 SNWD_Nov: float,
                 SNWD_Oct: float,
                 SNWD_Sep: float,
                 TAVG_Apr: float,
                 TAVG_Aug: float,
                 TAVG_Dec: float,
                 TAVG_Feb: float,
                 TAVG_Jan: float,
                 TAVG_Jul: float,
                 TAVG_Jun: float,
                 TAVG_Mar: float,
                 TAVG_May: float,
                 TAVG_Nov: float,
                 TAVG_Oct: float,
                 TAVG_Sep: float,
                 TMAX_Apr: float,
                 TMAX_Aug: float,
                 TMAX_Dec: float,
                 TMAX_Feb: float,
                 TMAX_Jan: float,
                 TMAX_Jul: float,
                 TMAX_Jun: float,
                 TMAX_Mar: float,
                 TMAX_May: float,
                 TMAX_Nov: float,
                 TMAX_Oct: float,
                 TMAX_Sep: float,
                 TMIN_Apr: float,
                 TMIN_Aug: float,
                 TMIN_Dec: float,
                 TMIN_Feb: float,
                 TMIN_Jan: float,
                 TMIN_Jul: float,
                 TMIN_Jun: float,
                 TMIN_Mar: float,
                 TMIN_May: float,
                 TMIN_Nov: float,
                 TMIN_Oct: float,
                 TMIN_Sep: float,
                 COMMODITY: str,
                 ACRES_HARVESTED: float,
                 ACRES_PLANTED: float):

        self.YEAR = YEAR
        self.STATE = STATE
        self.REGION = REGION
        self.PRCP_Apr = PRCP_Apr
        self.PRCP_Aug = PRCP_Aug
        self.PRCP_Dec = PRCP_Dec
        self.PRCP_Feb = PRCP_Feb
        self.PRCP_Jan = PRCP_Jan
        self.PRCP_Jul = PRCP_Jul
        self.PRCP_Jun = PRCP_Jun
        self.PRCP_Mar = PRCP_Mar
        self.PRCP_May = PRCP_May
        self.PRCP_Nov = PRCP_Nov
        self.PRCP_Oct = PRCP_Oct
        self.PRCP_Sep = PRCP_Sep
        self.SNOW_Apr = SNOW_Apr
        self.SNOW_Aug = SNOW_Aug
        self.SNOW_Dec = SNOW_Dec
        self.SNOW_Feb = SNOW_Feb
        self.SNOW_Jan = SNOW_Jan
        self.SNOW_Jul = SNOW_Jul
        self.SNOW_Jun = SNOW_Jun
        self.SNOW_Mar = SNOW_Mar
        self.SNOW_May = SNOW_May
        self.SNOW_Nov = SNOW_Nov
        self.SNOW_Oct = SNOW_Oct
        self.SNOW_Sep = SNOW_Sep
        self.SNWD_Apr = SNWD_Apr
        self.SNWD_Aug = SNWD_Aug
        self.SNWD_Dec = SNWD_Dec
        self.SNWD_Feb = SNWD_Feb
        self.SNWD_Jan = SNWD_Jan
        self.SNWD_Jul = SNWD_Jul
        self.SNWD_Jun = SNWD_Jun
        self.SNWD_Mar = SNWD_Mar
        self.SNWD_May = SNWD_May
        self.SNWD_Nov = SNWD_Nov
        self.SNWD_Oct = SNWD_Oct
        self.SNWD_Sep = SNWD_Sep
        self.TAVG_Apr = TAVG_Apr
        self.TAVG_Aug = TAVG_Aug
        self.TAVG_Dec = TAVG_Dec
        self.TAVG_Feb = TAVG_Feb
        self.TAVG_Jan = TAVG_Jan
        self.TAVG_Jul = TAVG_Jul
        self.TAVG_Jun = TAVG_Jun
        self.TAVG_Mar = TAVG_Mar
        self.TAVG_May = TAVG_May
        self.TAVG_Nov = TAVG_Nov
        self.TAVG_Oct = TAVG_Oct
        self.TAVG_Sep = TAVG_Sep
        self.TMAX_Apr = TMAX_Apr
        self.TMAX_Aug = TMAX_Aug
        self.TMAX_Dec = TMAX_Dec
        self.TMAX_Feb = TMAX_Feb
        self.TMAX_Jan = TMAX_Jan
        self.TMAX_Jul = TMAX_Jul
        self.TMAX_Jun = TMAX_Jun
        self.TMAX_Mar = TMAX_Mar
        self.TMAX_May = TMAX_May
        self.TMAX_Nov = TMAX_Nov
        self.TMAX_Oct = TMAX_Oct
        self.TMAX_Sep = TMAX_Sep
        self.TMIN_Apr = TMIN_Apr
        self.TMIN_Aug = TMIN_Aug
        self.TMIN_Dec = TMIN_Dec
        self.TMIN_Feb = TMIN_Feb
        self.TMIN_Jan = TMIN_Jan
        self.TMIN_Jul = TMIN_Jul
        self.TMIN_Jun = TMIN_Jun
        self.TMIN_Mar = TMIN_Mar
        self.TMIN_May = TMIN_May
        self.TMIN_Nov = TMIN_Nov
        self.TMIN_Oct = TMIN_Oct
        self.TMIN_Sep = TMIN_Sep
        self.COMMODITY = COMMODITY
        self.ACRES_HARVESTED = ACRES_HARVESTED
        self.ACRES_PLANTED = ACRES_PLANTED

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "YEAR": [self.YEAR],
                "STATE": [self.STATE],
                "REGION": [self.REGION],
                "PRCP Apr": [self.PRCP_Apr],
                "PRCP Aug": [self.PRCP_Aug],
                "PRCP Dec": [self.PRCP_Dec],
                "PRCP Feb": [self.PRCP_Feb],
                "PRCP Jan": [self.PRCP_Jan],
                "PRCP Jul": [self.PRCP_Jul],
                "PRCP Jun": [self.PRCP_Jun],
                "PRCP Mar": [self.PRCP_Mar],
                "PRCP May": [self.PRCP_May],
                "PRCP Nov": [self.PRCP_Nov],
                "PRCP Oct": [self.PRCP_Oct],
                "PRCP Sep": [self.PRCP_Sep],
                "SNOW Apr": [self.SNOW_Apr],
                "SNOW Aug": [self.SNOW_Aug],
                "SNOW Dec": [self.SNOW_Dec],
                "SNOW Feb": [self.SNOW_Feb],
                "SNOW Jan": [self.SNOW_Jan],
                "SNOW Jul": [self.SNOW_Jul],
                "SNOW Jun": [self.SNOW_Jun],
                "SNOW Mar": [self.SNOW_Mar],
                "SNOW May": [self.SNOW_May],
                "SNOW Nov": [self.SNOW_Nov],
                "SNOW Oct": [self.SNOW_Oct],
                "SNOW Sep": [self.SNOW_Sep],
                "SNWD Apr": [self.SNWD_Apr],
                "SNWD Aug": [self.SNWD_Aug],
                "SNWD Dec": [self.SNWD_Dec],
                "SNWD Feb": [self.SNWD_Feb],
                "SNWD Jan": [self.SNWD_Jan],
                "SNWD Jul": [self.SNWD_Jul],
                "SNWD Jun": [self.SNWD_Jun],
                "SNWD Mar": [self.SNWD_Mar],
                "SNWD May": [self.SNWD_May],
                "SNWD Nov": [self.SNWD_Nov],
                "SNWD Oct": [self.SNWD_Oct],
                "SNWD Sep": [self.SNWD_Sep],
                "TAVG Apr": [self.TAVG_Apr],
                "TAVG Aug": [self.TAVG_Aug],
                "TAVG Dec": [self.TAVG_Dec],
                "TAVG Feb": [self.TAVG_Feb],
                "TAVG Jan": [self.TAVG_Jan],
                "TAVG Jul": [self.TAVG_Jul],
                "TAVG Jun": [self.TAVG_Jun],
                "TAVG Mar": [self.TAVG_Mar],
                "TAVG May": [self.TAVG_May],
                "TAVG Nov": [self.TAVG_Nov],
                "TAVG Oct": [self.TAVG_Oct],
                "TAVG Sep": [self.TAVG_Sep],
                "TMAX Apr": [self.TMAX_Apr],
                "TMAX Aug": [self.TMAX_Aug],
                "TMAX Dec": [self.TMAX_Dec],
                "TMAX Feb": [self.TMAX_Feb],
                "TMAX Jan": [self.TMAX_Jan],
                "TMAX Jul": [self.TMAX_Jul],
                "TMAX Jun": [self.TMAX_Jun],
                "TMAX Mar": [self.TMAX_Mar],
                "TMAX May": [self.TMAX_May],
                "TMAX Nov": [self.TMAX_Nov],
                "TMAX Oct": [self.TMAX_Oct],
                "TMAX Sep": [self.TMAX_Sep],
                "TMIN Apr": [self.TMIN_Apr],
                "TMIN Aug": [self.TMIN_Aug],
                "TMIN Dec": [self.TMIN_Dec],
                "TMIN Feb": [self.TMIN_Feb],
                "TMIN Jan": [self.TMIN_Jan],
                "TMIN Jul": [self.TMIN_Jul],
                "TMIN Jun": [self.TMIN_Jun],
                "TMIN Mar": [self.TMIN_Mar],
                "TMIN May": [self.TMIN_May],
                "TMIN Nov": [self.TMIN_Nov],
                "TMIN Oct": [self.TMIN_Oct],
                "TMIN Sep": [self.TMIN_Sep],
                "COMMODITY": [self.COMMODITY],
                "ACRES HARVESTED": [self.ACRES_HARVESTED],
                "ACRES PLANTED": [self.ACRES_PLANTED]
            }
            pred_df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")  # Add logging after dataframe creation
            return pred_df

        except Exception as e:
            logging.error(f"Exception occurred: {e}")  # Log the exception
            raise CustomException(e, sys)
