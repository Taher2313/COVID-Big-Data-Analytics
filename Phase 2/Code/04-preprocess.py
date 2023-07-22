from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, when
from pyspark.ml.feature import Imputer

def load_data(file_path):
    spark = SparkSession.builder.appName("Load Data").getOrCreate()
    data = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(file_path)
    return data
        
def convert_classification(df):
    # Convert CLASIFFICATION_FINAL to binary classification
    binary_classification = df.withColumn('CLASIFFICATION_FINAL', when(col('CLASIFFICATION_FINAL').isin([1,2,3]), 2).otherwise(1))
    return binary_classification

def drop_columns(df):
    # Drop columns that are not needed
    drop_columns = ['ICU', 'INTUBED', 'PREGNANT']
    df = df.drop(*drop_columns)
    return df
                     
def perform_multiple_imputation(data):

    # Select columns with missing values
    missing_cols = [col for col in data.columns if data.filter(data[col].isNull()).count() > 0]

    # Create an instance of Imputer
    imputer = Imputer(inputCols=missing_cols, outputCols=[col+"_imputed" for col in missing_cols])

    # Fit the imputer on the data
    imputer_model = imputer.fit(data)

    # Transform the data with the fitted imputer
    imputed_data = imputer_model.transform(data)

    # drop original columns
    imputed_data = imputed_data.drop(*missing_cols)

    # rename imputed columns
    for col in missing_cols:
        imputed_data = imputed_data.withColumnRenamed(col+"_imputed", col)

    # move the CLASIFFICATION_FINAL column to the end
    imputed_data = imputed_data.select([col for col in imputed_data.columns if col != 'DIED'] + ['DIED'])

    return imputed_data

def cast_to_int(df):
    # Cast columns to integer
    for col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast('int'))
    return df

def drop_rows_with_null(df):
    # Drop rows with null values
    df = df.dropna()
    return df

def clean_age_outliers(df):
    # Drop rows with age outliers
    df = df.filter(df.AGE < 85)
    return df

def convert_data_to_binary_zero_one(df):
    # Convert all columns to 0 and 1
    columns = ["USMER","SEX","PATIENT_TYPE","PNEUMONIA","DIABETES","COPD","ASTHMA","INMSUPR","HIPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO","DIED"]
    for col_name in columns:
        df = df.withColumn(col_name, when(col(col_name) == 2, 1).otherwise(0))
    return df

def save_preprocessed_data(data, file_path):
    preprocessed_data = data.toPandas()
    preprocessed_data.to_csv(file_path, index=False)

def preprocess_data(input_file_path, output_file_path):
    df = load_data(input_file_path)
    # binary_classification = convert_classification(df)
    # imputed_df = perform_multiple_imputation(df)
    columns_droped_df = drop_columns(df)
    casted_df = cast_to_int(columns_droped_df)
    nan_dropped_df = drop_rows_with_null(casted_df)
    age_cleaned_df = clean_age_outliers(nan_dropped_df)
    zero_one_df = convert_data_to_binary_zero_one(age_cleaned_df)
    save_preprocessed_data(zero_one_df, output_file_path)

# Preprocess training data
preprocess_data('dataset/03-splited/train.csv', 'dataset/04-preprocessed/train.csv')

# Preprocess testing data
preprocess_data('dataset/03-splited/test.csv', 'dataset/04-preprocessed/test.csv')