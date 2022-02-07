from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
import process
import train
import test
import config
import utils
from time import time


if __name__ == '__main__':
    
    spark = SparkSession \
            .builder \
            .getOrCreate()


    args = utils.get_args()
    



    data = process.execute(spark,args.data_path)
    train_data, test_data = data.randomSplit([.8,.2],seed=1234)

    # train the model
    start_time = time()

    model = train.execute(train_data,args.k)

    end_time = time()
    elapsed_time = end_time - start_time
    print("Time to train model: %.3f seconds" % elapsed_time)

    test.execute(model,test_data)

    # save the model
    model.save(args.save_path)


    print("Closing Spark Session ...")
    spark.stop()


    