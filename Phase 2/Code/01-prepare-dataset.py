from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, when
from pyspark.ml.feature import Imputer

def load_data(file_path):
    spark = SparkSession.builder.appName("Prepare Data").getOrCreate()
    data = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(file_path)
    return data

def replace_with_null(df):

    # make columns with boolean values that have 97, 98, 99 as null values
    boolean_cols = ['SEX', 'PATIENT_TYPE',  'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
                'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
    
    for column_name in boolean_cols:
        df = df.withColumn(column_name, when((col(column_name) == 97) | (col(column_name) == 98) | (col(column_name) == 99), None)
                           .otherwise(col(column_name)))
        
    return df

def convert_death_date_to_boolean(df):
    # Convert DEATH_DATE to boolean
    df = df.withColumn('DATE_DIED', when(col('DATE_DIED') == '9999-99-99', 1).otherwise(2))
    df = df.withColumnRenamed('DATE_DIED', 'DIED')
    return df

def save_data(data, file_path):
    data.toPandas().to_csv(file_path, index=False)

def prepare_data(input_file_path, output_file_path):
    # Load the dataset
    df = load_data(input_file_path)

    df_with_null = replace_with_null(df)
    binary_death = convert_death_date_to_boolean(df_with_null)
    save_data(binary_death, output_file_path)

prepare_data('dataset/01-original/covid-dataset.csv', 'dataset/02-prepared/covid-dataset.csv')