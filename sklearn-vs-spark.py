from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  as RF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Global variables
# Sample size
SAMPLE_SIZE = 30000000

# Number of trees in the random forest
N_TREES = 100

# Number of cores to use
N_CORES = 10

def build_RF_with_sklearn(df, sample_size=100000):
    """
    Trains a Random Forest model using scikit-learn, evaluates its performance, and returns the model and metrics.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing features and the target variable. The target column should be named 'hotel_cluster'.
    sample_size : int
        The number of rows to sample from the input DataFrame. Default is 100,000

    Returns:
    -------
    clf : RandomForestClassifier
        The trained Random Forest model.
    accuracy : float
        The accuracy of the model on the test set.
    metric : float
        The MAP@5 evaluation score for the model on the test set.
    """
    # sample the data
    df = df.sample(n=sample_size, random_state=42)
    
    # Convert all columns to integers
    for column in df:
        df[column] = df[column].astype(str).astype(int)

    # Splitting features and target
    X = df.drop(['hotel_cluster'], axis=1)
    y = df['hotel_cluster'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Training the Random Forest
    start = datetime.now()
    clf = RF(n_jobs=N_CORES, n_estimators=N_TREES, random_state=42, 
             max_depth=4)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    end = datetime.now()
    time_taken = (end - start).total_seconds() / 60

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, pred)
    
    # # Evaluate using MAP@5
    # probs = clf.predict_proba(X_test)
    # actual = y_test
    # predicted = np.argsort(probs, axis=1)[:, -np.arange(5)]
    # metric = 0.0
    # for i in range(5):
    #     metric += np.sum(actual == predicted[:, i]) / (i + 1)
    # metric /= actual.shape[0]

    # print()
    # print(f"MAP@5: {metric:.4f}")

    num_rows = X.shape[0]
    print("="*60)
    print(' TRAINING WITH *SKLEARN* TOOK {:.2f} MINUTES, DETAILS BELOW'.format(time_taken))
    print("="*60)
    print(f" Num_rows: {num_rows:,}")
    print(f" Num_trees: {N_TREES:,}")
    print(f" Num_cores: {N_CORES:,}")
    print(f" Accuracy: {accuracy:,.2f}")
    print("-"*60)

def build_RF_with_spark(sdf, sample_size=100000):
    """
    Trains a Random Forest model using Spark MLlib, evaluates its performance, and returns the model and metrics.

    Parameters:
    ----------
    sdf : pyspark.sql.DataFrame
        The input DataFrame containing features and the target variable. The target column should be named 'hotel_cluster'.
    Returns:
    -------
    model : pyspark.ml.classification.RandomForestClassificationModel
        The trained Random Forest model.
    accuracy : float
        The accuracy of the model on the test set.
    time_taken : float
        The time taken to train the model (in seconds).
    """
    # Sample the data
    total_size = 37670293
    sample_fraction = sample_size/total_size
    sdf = sdf.sample(False, sample_fraction)
    # Prepare the data for Spark MLlib
    feature_columns = [col for col in sdf.columns if col != 'hotel_cluster']

    # Convert all columns to integers
    for column in sdf.columns:
        sdf = sdf.withColumn(column, sdf[column].cast(IntegerType()))
    
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    spark_df = assembler.transform(sdf).select("features", col("hotel_cluster").alias("label"))

    # Split the data into training and testing sets
    train_df, test_df = spark_df.randomSplit([0.75, 0.25], seed=42)

    # Train the Random Forest model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", 
                                numTrees=N_TREES, maxDepth=4)
    start_time = datetime.now()
    model = rf.fit(train_df)
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()/60

    # Make predictions
    predictions = model.transform(test_df)

    # Evaluate the model using accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    # Count the total rows processed
    num_rows = spark_df.count()
    print("="*60)
    print(' TRAINING WITH *SPARK* TOOK {:.2f} MINUTES, DETAILS BELOW'.format(time_taken))
    print("="*60)
    print(f" Num_rows: {num_rows:,}")
    print(f" Num_trees: {N_TREES:,}")
    print(f" Num_cores: {N_CORES:,}")
    print(f" Accuracy: {accuracy:,.2f}")
    print("-"*60)
    
def main():
    # Load the data
    DATA_FILE = Path.cwd().joinpath("data/kaggle-expedia-train.csv")
    # Define the columns to be used
    COLS = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt','srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']

    # =======================================
    # RUN WITH SKLEARN
    # =======================================
    pdf = pd.read_csv(DATA_FILE, usecols=COLS)
    pdf = pdf[COLS]
    build_RF_with_sklearn(pdf, sample_size=SAMPLE_SIZE)
    print("Done with sklearn")
    
    # =======================================
    # RUN WITH SPARK
    # =======================================
    # Create a Spark session
    spark = SparkSession.builder\
                .appName("LargeDatasetProcessing")\
                .master(f"local[{N_CORES}]")\
                .config("spark.driver.memory", "56g")\
                .config("spark.executor.memory", "56g")\
                .config("spark.memory.offHeap.enabled", "true")\
                .config("spark.sql.shuffle.partitions", "40")\
                .config("spark.memory.offHeap.size", "4g")\
                .getOrCreate()
    # Set log level to ERROR
    spark.sparkContext.setLogLevel("ERROR")
    
    # Load and parse the data file, converting it to a DataFrame.
    sdf = spark.read.csv(str(DATA_FILE), header=True)

    # Select the columns to be used
    sdf = sdf.select(COLS)

    # drop rows with missing values
    sdf = sdf.dropna()
    
    # Run the Random Forest model with Spark MLlib
    build_RF_with_spark(sdf, sample_size=SAMPLE_SIZE)
    
    # Stop the Spark session
    spark.stop()
if __name__ == "__main__":
    main()