import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            label_encoder = LabelEncoder()
            train_df['STATE'] = label_encoder.fit_transform(train_df['STATE'])
            test_df['STATE'] = label_encoder.transform(test_df['STATE'])
            train_df['REGION'] = label_encoder.fit_transform(train_df['REGION'])
            test_df['REGION'] = label_encoder.transform(test_df['REGION'])
            train_df['COMMODITY'] = label_encoder.fit_transform(train_df['COMMODITY'])
            test_df['COMMODITY'] = label_encoder.transform(test_df['COMMODITY'])            

            target_column_name = "YIELD"

            # Define numerical columns
            numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols.remove('YIELD')

            input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Creating preprocessing pipeline")
            num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

            preprocessing_obj = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_cols)
                ]
            )

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            logging.info('Exception occurred in initiate_data_transformation function')
            raise CustomException(e, sys)
