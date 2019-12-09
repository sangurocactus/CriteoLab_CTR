#!/usr/bin/env python


import time
import numpy as np
import pyspark

from pyspark.sql import SQLContext
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
import pyspark.sql.functions as f
# from pyspark.sql.functions import conv, mean, max, min
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType, IntegerType, DoubleType, FloatType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, VectorIndexer, StringIndexer, StandardScaler, OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import when
import re
import ast
import time
import numpy as np
import pandas as pd
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import lit,avg
from pyspark.sql.window import Window
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import Imputer

app_name = "final_proj"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()

conf = pyspark.SparkConf().setAll([ ('spark.executor.pyspark.memory', '11g'), ('spark.driver.memory','11g'), ('spark.driver.maxResultSize', '11g')])



#set new runtime options
spark.conf.set("spark.executor.pyspark.memory", "11g")
spark.conf.set("spark.executor.memory", "11g")
spark.conf.set("spark.driver.maxResultSize", "11g")


sc = spark.sparkContext
sqlContext = SQLContext(sc)
############## YOUR BUCKET HERE ###############

BUCKET="w261-ml"

############## (END) YOUR BUCKET ###############


trainDF = spark.read.load("gs://"+BUCKET+"/train_80.parquet")

print(trainDF.count())

def prep_df(DF):
    #convert Label and all Integer columns to Ints and Categorical to Longs. 
    DF = DF.withColumn('Label', DF['Label'].cast(IntegerType()))
    for i in range(1,14):
        feature = 'I-'+str(i)
        DF = DF.withColumn(feature, DF[feature].cast(IntegerType()))
    return DF
    
trainDF = prep_df(trainDF)


all_columns = ['Label']
for i in range(1,14): all_columns.append('I-'+str(i))
for i in range(1,27): all_columns.append('C-'+str(i))
    
continuous_indexes = [i for i in range(1,14)]
category_indexes = [i for i in range(14,40)]

#Fill all Nulls or NAs with mean values
def imputers(dataframe):
    inputCols = []
    outputCols = []
    for i in range(1,14):
        feature = 'I-'+str(i)
        dataframe =  dataframe.withColumn(feature, dataframe[feature].cast(DoubleType())) 
        inputCols.append(feature)
        outputCols.append(feature)
    imputer = Imputer(strategy="mean",
        inputCols=inputCols,
        outputCols=outputCols)
    return imputer.fit(dataframe).transform(dataframe)


train_imputed_DF = imputers(trainDF)

def assembler_indexers(dataframe):
    assemblers = [VectorAssembler(inputCols=[all_columns[i]], outputCol=all_columns[i]+"-Vector") for 
                i in continuous_indexes ]
    pipeline = Pipeline(stages=assemblers)
    assembled_dataframe = pipeline.fit(dataframe).transform(dataframe)
    return assembled_dataframe

train_assembled_indexedDF   = assembler_indexers(train_imputed_DF)

def scaler_indexers(dataframe):
    indexers = [StandardScaler(inputCol=all_columns[i]+"-Vector", outputCol=all_columns[i]+"-Scaled",
                        withStd=True, withMean=True).fit(dataframe) for 
                i in continuous_indexes ]
    pipeline = Pipeline(stages=indexers)
    scaled_dataframe = pipeline.fit(dataframe).transform(dataframe)
    for i in continuous_indexes:
        scaled_dataframe = scaled_dataframe.drop(all_columns[i])
    for i in continuous_indexes:
        scaled_dataframe = scaled_dataframe.drop(all_columns[i]+"-Vector")
    return scaled_dataframe


train_scaled_indexedDF   = scaler_indexers(train_assembled_indexedDF)
train_scaled_indexedDF.show(10)

del train_assembled_indexedDF

def string_indexers(dataframe):
    indexers = [StringIndexer(inputCol=all_columns[i], outputCol=all_columns[i]+"-Index").fit(dataframe) for 
                i in category_indexes ]
    pipeline = Pipeline(stages=indexers)
    indexed_dataframe = pipeline.fit(dataframe).transform(dataframe)
    for i in category_indexes:
        indexed_dataframe = indexed_dataframe.drop(all_columns[i])
    return indexed_dataframe


train_S_indexedDF   = string_indexers(train_scaled_indexedDF)

def drop_threshold(df, threshold, replace_val, column_names):
    '''Help function that replace all values greater than a threshold with the replacement value'''
    for col in column_names:
        df=df.withColumn(col, when(df[col]>threshold, replace_val).otherwise(df[col]))
    return df

categorical_columns=["C-1-Index","C-2-Index","C-3-Index","C-4-Index","C-5-Index","C-6-Index","C-7-Index","C-8-Index"
                     ,"C-9-Index","C-10-Index","C-11-Index","C-12-Index", "C-13-Index","C-14-Index","C-15-Index",
                     "C-16-Index","C-17-Index","C-18-Index","C-19-Index","C-20-Index","C-21-Index", "C-22-Index",
                     "C-23-Index","C-24-Index","C-25-Index","C-26-Index"]

train_S_indexed_ranked_DF = drop_threshold (train_S_indexedDF, 100, 999, categorical_columns)


def Breiman(df, label_column, column_names):
    '''This function calculates the average of a given column conditional on the value of another column'''
    for col in column_names:
        print(col)
        w = Window().partitionBy(col)
        df = df.withColumn(col+"B", avg(label_column).over(w))
        df.drop(col)
        df.persist()
    return df;

train_S_indexed_breiman_DF = Breiman(train_S_indexed_ranked_DF, "Label", categorical_columns)

train_S_indexed_breiman_DF.show(10)


train_S_indexed_breiman_DF.write.parquet("gs://"+BUCKET+"/train_preprocessed.parquet")

trainDF = spark.read.load("gs://"+BUCKET+"/train_preprocessed.parquet")

trainDF.cache()
print(toyDF.count())

trainDF.show()