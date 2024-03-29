# imports
import time
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from pyspark.sql.functions import conv, mean, udf
from pyspark.sql.types import LongType, IntegerType, DoubleType, FloatType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, VectorIndexer, StringIndexer, StandardScaler, OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

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
toyDF = spark.read.load('gs://'+BUCKET+'/toy_naga.parquet')
miniValDF = spark.read.load('gs://'+BUCKET+'/miniValidation.parquet')

# Process Label and I-1 through I-13 by casting to integers ###
def prep_df(DF):
    #convert Label and all Integer columns to Ints and Categorical to Longs. 
    DF = DF.withColumn('Label', DF['Label'].cast(IntegerType()))
    for i in range(1,14):
        feature = 'I-'+str(i)
        DF = DF.withColumn(feature, DF[feature].cast(IntegerType()))
    
    # Replace all categorical string `empty` to `null`
   # DF = DF.replace('', 'null')
    
    # Fill all Integer columns `NA` to `0'
    DF = DF.fillna(0)
    return DF
    
toyDF = prep_df(toyDF)
#trainDF = prep_df(trainDF)
#validationDF = prep_df(validationDF)
#testDF = prep_df(testDF)
miniValDF = prep_df(miniValDF)


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

# Compute log loss on the given Spark Dataframe.
def logLoss(predDF):
    # Define a function clamp to restrict the values of probability to be greater than 0 and less than one
    def clamp(n):
        epsilon = .000000000000001
        minn = 0 + epsilon
        maxn = 1 - epsilon
        return max(min(maxn, n), minn)
    
    # Define a UDF to extract the first element of the probability array returned which is probability of one
    firstelement=udf(lambda v:clamp(float(v[1])))   #,FloatType() after [] was inserted and removed for epsilon
    
    # Create a new dataframe that contains a probability of one column (true)
    predict_df = predDF.withColumn('prob_one', firstelement(predDF.probability))
    
    # Compute the log loss for the spark dataframe for each row
    row_logloss = (predict_df.withColumn(
        'logloss', -f.col('Label')*f.log(f.col('prob_one')) - (1.-f.col('Label'))*f.log(1.-f.col('prob_one'))))

    logloss = row_logloss.agg(f.mean('logloss').alias('ll')).collect()[0]['ll']
    return logloss

# Explore the optimal number of trees. 
RF_results = []

# Drop I-12 and C-22 with 76% missing values
ignore = ['I-12', 'C-22', 'Label']

# Also ignore all categorical for testing purposes.
for i in range(1, 26): ignore.append('C-'+str(i))
    
def string_indexers(dataframe):
    indexers = [StringIndexer(inputCol=all_columns[i], outputCol=all_columns[i]+"-Index").fit(dataframe) for 
                i in category_indexes ]
    pipeline = Pipeline(stages=indexers)
    indexed_dataframe = pipeline.fit(dataframe).transform(dataframe)
    for i in category_indexes:
        indexed_dataframe = indexed_dataframe.drop(all_columns[i])
    return indexed_dataframe

all_columns = ['Label']
for i in range(1,14): all_columns.append('I-'+str(i))
for i in range(1,27): all_columns.append('C-'+str(i))
    
continuous_indexes = [i for i in range(1,14)]
category_indexes = [i for i in range(14,40)]

def assembler_indexers(dataframe):
    assemblers = [VectorAssembler(inputCols=[all_columns[i]], outputCol=all_columns[i]+"-Vector") for i in continuous_indexes ]
    pipeline = Pipeline(stages=assemblers)
    assembled_dataframe = pipeline.fit(dataframe).transform(dataframe)
    return assembled_dataframe

#apply standard scaler to numeric columns
def scaler_indexers(dataframe):
    indexers = [StandardScaler(inputCol=all_columns[i]+"-Vector", outputCol=all_columns[i]+"-Scaled",
                        withStd=True, withMean=False).fit(dataframe) for 
                i in continuous_indexes ]
    pipeline = Pipeline(stages=indexers)
    scaled_dataframe = pipeline.fit(dataframe).transform(dataframe)
    #for i in continuous_indexes:
        #indexed_dataframe = indexed_dataframe.drop(all_columns[i])
    return scaled_dataframe

toyDF_scaled_indexed   = scalar_indexers(assembler_indexers(string_indexers(toyDF)))
toyDF_scaled_indexed.show(5)

# Select Integer features only.
temp = ['Label']
for i in range(1,14): temp.append('I-'+str(i))
toyDF_trans = vector_transform(toyDF.select(temp), ignore)

miniValDF_trans = vector_transform(miniValDF.select(temp), ignore)

print('DFs transformed')
for i in range(25,301, 25):
    time0 = time.time()
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'Label', maxDepth = 8, numTrees=i)
    rfModel = rf.fit(toyDF_trans)
    predictions = rfModel.transform(miniValDF_trans)
    log_loss = logLoss(predictions)
    evaluator = BinaryClassificationEvaluator(labelCol='Label')
    auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    auprc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
    wall_time = time.time()-time0
    RF_results.append([i, log_loss, auroc, auprc, wall_time])
    print('finished training', i)
RF_base_PD = pd.DataFrame(RF_results, columns=['# of Trees', 'Log Loss', 'Area Under ROC', 'Area Under PR', 'Wall Time'])
print(RF_base_PD)
print(RF_results)
#RF_base_PD.to_csv('gs://'+BUCKET+'/RF_base_PD.csv', index=False)
#RF_base_PD.to_parquet('gs://'+BUCKET+'/RF_base_PD.parquet')
#RF_base_PD.to_pickle('gs://'+BUCKET+'/RF_base_PD.pkl')





# Select Integer features only.
#temp = ['Label']
#for i in range(1,14): temp.append('I-'+str(i))
#testDF_trans = vector_transform(testDF.select(temp), ignore)

#predictions.write.parquet("gs://"+BUCKET+"/predictions.parquet")