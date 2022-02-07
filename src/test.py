from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
import utils


def execute(model,data):
    # Evaluate model
    predictions = model.transform(data)
    acc = utils.accuracy(predictions=predictions)
    print("Model accuracy: %.3f%%" % (acc * 100))


    print(f"================== Evaluation Summary ==================")
    
    preds = predictions.select('label','prediction').rdd.map(lambda row: (float(row['prediction']), float(row['label'])))
    bcm = BinaryClassificationMetrics(preds)
    mcm = MulticlassMetrics(preds)

    print(">>> Area Under Precision-Recall : ",bcm.areaUnderPR)
    print(">>> Area Under ROC : ",bcm.areaUnderROC)
    print(">>> Precision : ", mcm.weightedPrecision)
    print(">>> Recall : ", mcm.weightedRecall)
    print(">>> F1 Score : ", mcm.weightedFMeasure())