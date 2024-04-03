# Import all the required libraries
import os
import sys
import warnings
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

warnings.filterwarnings("ignore")

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            weather_directory = "notebook/Cleaned Weather Datasets"
            agriculture_data = pd.read_csv('notebook/Cleaned Agriculture Datasets/crop.csv')
            
            # List of states
            states = [
                'ALABAMA', 'ARKANSAS', 'ARIZONA', 'CALIFORNIA', 'COLORADO', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'IDAHO',
                'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MARYLAND', 'MICHIGAN', 'MINNESOTA',
                'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK',
                'NORTH CAROLINA', 'NORTH DAKOTA','OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'SOUTH CAROLINA',
                'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN',
                'WYOMING'
            ]

            # Function to read CSV files
            def read_csv_files(weather_directory, states):
                dataframes = {}
                for state in states:
                    filename = os.path.join(weather_directory, state + '.csv')
                    if os.path.exists(filename):
                        dataframes[state] = pd.read_csv(filename)
                    else:
                        print(f"File not found: {filename}")
                return dataframes

            # Read CSV files
            state_data = read_csv_files(weather_directory, states)

            # Combine data from all states
            merged_dfs = []

            for state, data in state_data.items():
                # Convert the 'DATE' column to datetime format
                data['DATE'] = pd.to_datetime(data['DATE'])

                # Extract the year and month from the 'DATE' column and create new 'YEAR' and 'MONTH' columns
                data['YEAR'] = data['DATE'].dt.year
                data['MONTH'] = data['DATE'].dt.month

                # Define a dictionary to map month numbers to month names
                month_names = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }

                # Replace numerical month values with month names
                data['MONTH'] = data['MONTH'].map(month_names)

                # Group the data by 'YEAR' and 'MONTH' and calculate the mean for each group
                monthly_average = data.groupby(['YEAR', 'MONTH']).agg({
                    'PRCP': 'mean',
                    'SNOW': 'mean',
                    'SNWD': 'mean',
                    'TAVG': 'mean',
                    'TMAX': 'mean',
                    'TMIN': 'mean',
                    'STATE': 'first',  # retain the STATE column
                    'REGION': 'first',  # retain the REGION column
                }).reset_index()

                # Pivot the data to have separate columns for each weather type and month
                pivot_data = monthly_average.pivot_table(index=['YEAR', 'STATE', 'REGION'], columns=['MONTH'], values=['PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN'], aggfunc='mean').reset_index()

                # Flatten the multi-level columns
                pivot_data.columns = [' '.join(col).strip() for col in pivot_data.columns.values]

                # Append to list of merged DataFrames
                merged_dfs.append(pivot_data)

            # Concatenate all DataFrames into one
            final_merged_df = pd.concat(merged_dfs, ignore_index=True)
            
            final_merged_df_1 = final_merged_df[final_merged_df['STATE'] == 'DELAWARE']

            # Exclude non-numeric columns
            numeric_cols = final_merged_df_1.select_dtypes(include='number').columns

            final_merged_df_1[numeric_cols] = final_merged_df_1[numeric_cols].fillna(final_merged_df_1[numeric_cols].mean())

            # Drop rows where STATE is DELAWARE
            final_merged_df_3 = final_merged_df[final_merged_df['STATE'] != 'DELAWARE']

            # Assuming final_merged_df_2 and final_merged_df_3 are your DataFrames
            # Concatenate the two DataFrames
            weather_clean = pd.concat([final_merged_df_1, final_merged_df_3], ignore_index=True)


            agriculture_clean = agriculture_data.pivot_table(index=['YEAR', 'STATE', 'COMMODITY'],
                                                            columns='DATA ITEM',
                                                            values='VALUE',
                                                            aggfunc='sum')

            # Reset index to make the resulting DataFrame cleaner
            agriculture_clean.reset_index(inplace=True)

            # Rename the columns to remove leading spaces
            agriculture_clean.columns = agriculture_clean.columns.str.strip()

            agriculture_clean = agriculture_clean[agriculture_clean['STATE'] != 'OTHER STATES']

            # Merge final_merged_df with soybeans_data
            df_1 = pd.merge(weather_clean,agriculture_clean, on=['YEAR', 'STATE'])
            logging.info('Dataset read as pandas Dataframe')

            # Define conversion factors
            conversion_factors = {'SOYBEANS': 60, 'WHEAT': 60}

            # Update yield values for soybeans and wheat based on the crop name in the 'commodity' column
            for index, row in df_1.iterrows():
                if row['COMMODITY'] in conversion_factors:
                    df_1.at[index, 'YIELD'] *= conversion_factors[row['COMMODITY']]            

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df_1.to_csv(self.ingestion_config.raw_data_path,index=False)

            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df_1, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))