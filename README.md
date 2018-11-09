## Logistic Regression on Hadoop using PySpark

* Dataset: MNIST dataset for Handwritten Digit Classification

The project was done as a part of a college course- Big Data Analytics (VIT, Vellore, Tamil Nadu, India).
The goal was to run a Logistic Regression model on Hadoop, using the PySpark API, and compare its performance
with the models that were run without Hadoop, i.e., on local file system (Linux).

Three different implementations of Logistic Regression were run:
  * Written from scratch in Python
  * Using scikit-learn LogisticRegression class
  * Using Spark LogisticRegression

While the first one did not finish training, scikit-learn took 28 minutes, while Spark completed training on the
entire dataset in 25 seconds (about 65x faster than sklearn!). This kind of contrasting difference was not
expected to be observed, and it was amazing.
