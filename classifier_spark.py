from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('logisticsRegression').getOrCreate()

from pyspark.ml.classification import LogisticRegression

import time

start_loading_data_time = time.time()

training_set = spark.read.csv('hdfs://localhost:9000/BigDataProject/dataset/train.csv', header = True, inferSchema = True)

print('Data loading time: ' + str(time.time() - start_loading_data_time))


# Training

# Transforming the dataset- creating an array for each row in the dataset. Each such array will be assigned to a new column names 'features'




from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


feature_columns = training_set.columns[1:]
label_column = training_set.columns[0]
# print(feature_columns)
# print(label_column)

print(label_column)

vector_assembler = VectorAssembler(inputCols = feature_columns, outputCol = 'features')


final_train_set = vector_assembler.transform(training_set)

final_train_set.count()

logisticRegressor = LogisticRegression(featuresCol = 'features', labelCol = 'label')

start_training_time = time.time()


fitted_logisticRegressor = logisticRegressor.fit(final_train_set)

print('training time: ' + str(time.time() - start_training_time))


test_data = spark.read.csv('hdfs://localhost:9000/BigDataProject/dataset/test.csv', header = True, inferSchema = True)


# test_data.dtypes


# #### Transforming the test dataset in the same way as we did for the training set

# test_data.columns

feature_columns = test_data.columns[1:]
label_column = test_data.columns[0]


# feature_columns

vector_assembler_2 = VectorAssembler(inputCols = feature_columns, outputCol= 'features')


final_test_data = vector_assembler_2.transform(test_data)

final_test_data.select("features").show()


prediction_and_labels = fitted_logisticRegressor.evaluate(final_test_data)

prediction_and_labels.predictions.select(['label','prediction']).show()

type(prediction_and_labels.predictions.select('label'))

list_of_labels = prediction_and_labels.predictions.select('label').collect()

num_rows_in_list = prediction_and_labels.predictions.select('label').count()
num_rows_in_list

list_of_labels = [list_of_labels[i].label for i in range(num_rows_in_list)]

# list_of_prediction_and_labels


list_of_predictions = prediction_and_labels.predictions.select('prediction').collect()
list_of_predictions = [int(list_of_predictions[i].prediction) for i in range(num_rows_in_list)]

# list_of_predictions

list_of_labels_prediction_pairs = zip(list_of_labels, list_of_predictions)


list_of_labels_prediction_pairs = list(list_of_labels_prediction_pairs)

# list_of_labels_prediction_pairs


# Evaluating the classifer

from pyspark.mllib.evaluation import MulticlassMetrics

predictions_and_labels_parallelized = sc.parallelize(list_of_labels_prediction_pairs)

metrics = MulticlassMetrics(predictions_and_labels_parallelized)

correct_predictions = 0;

for i in range(num_rows_in_list):
    if (list_of_labels_prediction_pairs[i][0] == list_of_labels_prediction_pairs[i][1]):
        correct_predictions += 1
        
print(correct_predictions)

testset_num_rows = num_rows_in_list
test_accuracy = correct_predictions/testset_num_rows
print(test_accuracy)

import pandas as pd

labels_pd = pd.DataFrame(list_of_labels)
predictions_pd = pd.DataFrame(list_of_predictions)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(predictions_pd, labels_pd)

conf_mat


sum = 0
for i in range(10):
    sum += conf_mat[i, i]
    
sum # total rows in test set = 10k

