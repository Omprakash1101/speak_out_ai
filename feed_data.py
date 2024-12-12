from pyspark.sql import SparkSession
import pandas as pd
import os

spark= SparkSession.builder.appName("Train").getOrCreate()
while(1):
    inp=input("Input:")
    cat=input("Category:")
    ton=input("Tone:")
    lan=input("Lang:")
    res=input("Result:")
    data = [(inp,cat,ton,lan,res)]
    columns = ["input", "category", "tone","lang","result"]
    df = spark.createDataFrame(data, columns)
    pdf=df.toPandas()
    pdf.to_csv("training_data.csv", mode="a", header=False, index=False)
    con=input("Do you want to continue(Y/N):")
    if con=="y" or con=="Y":
        pass
    else:
        break
