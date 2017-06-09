"""
Gradient Boosted Trees Classification Example.
"""
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="PythonGradientBoostedTreesClassificationExample")
    # $example on$
    # Load and parse the data file.
    data = sparkContext.textFile("s3n://trytrytrytry/sample_libsvm_data.txt")
    #data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a GradientBoostedTrees model.
    #  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
    #         (b) Use more iterations in practice.
    model = GradientBoostedTrees.trainClassifier(trainingData,
                                                 categoricalFeaturesInfo={}, numIterations=3)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification GBT model:')
    print(model.toDebugString())

    # Save and load model
    model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
    sameModel = GradientBoostedTreesModel.load(sc,
                                               "target/tmp/myGradientBoostingClassificationModel")
    # $example off$