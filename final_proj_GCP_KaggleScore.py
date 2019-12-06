# imports
import time
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from pyspark.sql.functions import conv, mean, max, min, udf
from pyspark.sql.types import LongType, IntegerType, DoubleType, FloatType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, VectorIndexer, StringIndexer, StandardScaler, OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier

app_name = "final_proj"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

#conf = pyspark.SparkConf().setAll([ ('spark.executor.pyspark.memory', '11g'), ('spark.driver.memory','11g')])
#sc = pyspark.SparkContext(conf=conf)

BUCKET="jyu-mids-w261-2019fall"

# create spark dataframe
trainDF = spark.read.load('gs://'+BUCKET+'/train_80.parquet')
testDF = spark.read.load('gs://'+BUCKET+'/test.parquet')

# Process I-1 through I-13 by casting to integers ###
def prep_df(DF):
    # Fill all Integer columns `NA` to `0'
    DF = DF.fillna(0)
    
    # Replace all categorical string `empty` to `null`
   # DF = DF.replace('', 'null')

    #convert all Label, Integer columns to Ints and Categorical to Longs. 
    DF = DF.withColumn("Label", DF["Label"].cast(IntegerType()))
    DF = DF.withColumn("I-1", DF["I-1"].cast(IntegerType()))
    DF = DF.withColumn("I-2", DF["I-2"].cast(IntegerType()))
    DF = DF.withColumn("I-3", DF["I-3"].cast(IntegerType()))
    DF = DF.withColumn("I-4", DF["I-4"].cast(IntegerType()))
    DF = DF.withColumn("I-5", DF["I-5"].cast(IntegerType()))
    DF = DF.withColumn("I-6", DF["I-6"].cast(IntegerType()))
    DF = DF.withColumn("I-7", DF["I-7"].cast(IntegerType()))
    DF = DF.withColumn("I-8", DF["I-8"].cast(IntegerType()))
    DF = DF.withColumn("I-9", DF["I-9"].cast(IntegerType()))
    DF = DF.withColumn("I-10", DF["I-10"].cast(IntegerType()))
    DF = DF.withColumn("I-11", DF["I-11"].cast(IntegerType()))
    DF = DF.withColumn("I-12", DF["I-12"].cast(IntegerType()))
    DF = DF.withColumn("I-13", DF["I-13"].cast(IntegerType()))
    
    # Fill all Integer columns `NA` to `0'
    DF = DF.fillna(0)
    return DF
    
trainDF = prep_df(trainDF)
testDF = prep_df(testDF)


# Input: Spark Dataframe and list of features to ignore
# Output: Transformed feature format to feed to Decision Tree model
# Function builds a vector from the given Spark Dataframe, excluding the features
# in 'ignore_list'. It uses VectorAssembler to build a vector and then transform
# to 'label | [feature 1, feature 2, etc]' format.
def vector_transform(DF, ignore_list):
    # Create a list of features 
    assemblerInputs = []
    for i in range(1,14): assemblerInputs.append('I-'+str(i))
    for i in range(1,27): assemblerInputs.append('C-'+str(i))
        
    # Build vector for decision tree for dataset
    assembler = VectorAssembler(inputCols=[x for x in DF.columns if x not in ignore],
                                outputCol='features')

    # Transform the data for train data set
    output = assembler.transform(DF)
    #print(output.select("Label", "features").show(truncate=False))
    return output

# Drop I-12 and C-22 with 76% missing values
ignore = ['I-12', 'C-22', 'Label']

# Also ignore all categorical for testing purposes.
for i in range(1,27): ignore.append('C-'+str(i))
trainDF_trans = vector_transform(trainDF, ignore)

# Select Integer features only.
temp = ['Label']
for i in range(1,14): temp.append('I-'+str(i))
testDF_trans = vector_transform(testDF.select(temp), ignore)

time0 = time.time()
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'Label', maxDepth = 10)
dtModel = dt.fit(trainDF_trans)
predictions = dtModel.transform(testDF_trans)
predictions.select('Label', 'rawPrediction', 'prediction', 'probability').show(20)
print('Wall time: ', time.time()-time0)

predictions.write.parquet("gs://"+BUCKET+"/predictions.parquet")