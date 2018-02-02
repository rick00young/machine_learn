# from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

base_data_dir = '/Users/rick/src/ml_data'

spark = SparkSession.builder.master('local[2]').appName('CTRLogisticRegression') \
	.config("spark.sql.warehouse.dir", "./spark-warehouse").getOrCreate()
sc = spark.sparkContext
# spark = SparkSession.builder.appName('ppp').master('local[2]').getOrCreate()
# sc = spark


def parsePoint(line):
	values = [float(x) for x in line.split(',')]
	features = values[0:1]
	features.extend(values[2:])
	return LabeledPoint(values[1], features)


data = sc.textFile(base_data_dir + '/data/kaggle_click_through_rate/xae')
# test = sc.textFile(base_data_dir + '/data/kaggle_click_through_rate/test.csv')

train_data = data.map(parsePoint)

model = LogisticRegressionWithLBFGS.train(train_data)

label_predict = train_data.map(lambda p: (p.label,
                                          model.predict(p.features)))
train_err = label_predict.filter(lambda v: v[0] != v[1]) \
	            .count() / float(train_data.count())

print('train error = ', str(train_err))
model.save(sc, 'target/tmp/CTR')
