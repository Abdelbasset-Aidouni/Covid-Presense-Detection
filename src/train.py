from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
import config
import utils



def execute(data,k=config.DEFAULT_K):


    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()

    # Create ParamGrid for Cross Validation
    LRparamGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.01,0.2,0.3,0.4,0.5]).build())
    RFparamGrid = (ParamGridBuilder()\
                .addGrid(rf.maxDepth, [10, 20, 25, 30])
                .addGrid(rf.maxBins, [100, 50, 25])\
                .addGrid(rf.numTrees, [5, 20, 30])\
                .addGrid(rf.impurity, ['gini', 'entropy'])\
                .build())
    DTparamGrid = (ParamGridBuilder()\
                .addGrid(dt.maxDepth, [10, 20, 25, 30])
                .addGrid(dt.maxBins, [100, 50, 25])\
                .addGrid(dt.impurity, ['gini', 'entropy'])\
                .build())



    models = [
        {
            "name" : "Logistic Regression",
            "model":lr,
            "params":LRparamGrid,
        },
        {
            "name" : "Decision Tree",
            "model":dt,
            "params":DTparamGrid,
        },
        {
            "name" : "Random Forest",
            "model":rf,
            "params":RFparamGrid,
        }
    ]



    print("Running K-fold Cross Validation ...")
    print(f"================== {k}-fold Cross Validation ==================")
    evaluator = BinaryClassificationEvaluator()

    bestModel = None
    max_avg = 0
    for model in models:
        # Create 5-fold CrossValidator
        cv = CrossValidator(estimator=model["model"],
                            estimatorParamMaps=model["params"],
                            evaluator=evaluator, 
                            numFolds=k)

        # Run cross validations
        cvModel = cv.fit(data)
        print(f"{model['name']} \nAvg Metrics : {cvModel.avgMetrics[0]}")
        if cvModel.avgMetrics[0] > max_avg:
            bestModel = cvModel
            max_avg = cvModel.avgMetrics[0]

    print("Finshed Training !")

    return bestModel