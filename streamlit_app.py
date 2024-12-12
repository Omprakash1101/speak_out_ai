import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
import os

os.system("apt-get update")
os.system("apt-get install openjdk-8-jdk -y")
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'

spark = SparkSession.builder.appName("AI_Response").master("local[*]").getOrCreate()
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
# Function to find the most relevant answer based on input question
def get_answer(input_question):
    input_question = input_question.strip().lower()

    # Convert to lower case and match input question with questions in the DataFrame
    result = df.filter(F.lower(df['question']).contains(input_question.lower()))

    # Check if a match was found
    if result.count() > 0:
        answer = result.select('answer').first()[0]
        return answer
    else:
        return get_similar_answer(input_question.lower())


st.header("Speak Out AI")
st.write("I am Speak out AI which is made by Mr. G. Omprakash from EEC,Chennai")
st.write("My aim is to provide you a better mental health analysis")
st.write("To help you out from mental health issues and feel free to talk.")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):
    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    o=get_answer(prompt)+"\n"
    # Generate a response using the OpenAI API.
    stream = [o,"\nCurrently We are working on it"]
    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})