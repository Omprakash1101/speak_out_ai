from pyspark.ml.linalg import DenseVector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("AI_Response_Cosine_Similarity").getOrCreate()
df = spark.read.csv("training_data.csv", header=True, inferSchema=True)
print(df.show())

df = df.withColumn('question', F.lower(F.col('question')))
print(df.show())

tokenizer = Tokenizer(inputCol="question", outputCol="words")
hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=20)
idf = IDF(inputCol="raw_features", outputCol="features")
pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
model = pipeline.fit(df)
result = model.transform(df)
result.show(truncate=False)

def extract_features(row):
    return np.array(row['features'].toArray())
features = result.rdd.map(extract_features).collect()
def get_similar_answer(input_question):
    input_question = input_question.strip().lower()
    input_vector = model.transform(spark.createDataFrame([(input_question,)], ['question'])).select('features').head()[0].toArray().reshape(1, -1)
    similarities = cosine_similarity(input_vector, features)
    most_similar_idx = similarities.argmax()
    answer = result.select('answer').collect()[most_similar_idx][0]
    return answer

response = get_similar_answer("hello")
print(response)
