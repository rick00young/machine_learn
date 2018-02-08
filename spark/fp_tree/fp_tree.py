'''
http://www.cnblogs.com/pinard/p/6340162.html
'''
import os
import sys

os.environ['SPARK_HOME'] = "/Users/rick/src/hadoop/spark-2.2.1-bin-hadoop2.7"


from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.fpm import FPGrowth
sc = SparkContext('local', 'testing')

data = [["A", "B", "C", "E", "F","O"], ["A", "C", "G"], ["E","I"], ["A", "C","D","E","G"], ["A", "C", "E","G","L"],
       ["E","J"],["A","B","C","E","F","P"],["A","C","D"],["A","C","E","G","M"],["A","C","E","G","N"]]
rdd = sc.parallelize(data, 2)
# #支持度阈值为20%
# model = FPGrowth.train(rdd, 0.2, 2)
#
# print(sorted(model.freqItemsets().collect()))

from pyspark.mllib.fpm import PrefixSpan
data = [
   [['a'],["a", "b", "c"], ["a","c"],["d"],["c", "f"]],
   [["a","d"], ["c"],["b", "c"], ["a", "e"]],
   [["e", "f"], ["a", "b"], ["d","f"],["c"],["b"]],
   [["e"], ["g"],["a", "f"],["c"],["b"],["c"]]
   ]
rdd = sc.parallelize(data, 2)
model = PrefixSpan.train(rdd, 0.5,4)
print(sorted(model.freqSequences().collect()))