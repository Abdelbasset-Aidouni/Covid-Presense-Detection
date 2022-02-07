from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import config

def execute(spark,data_path):

    
    print("Processing the data ...")
    
    df = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(data_path)


    # These two variables have one value : NO
    df.drop(*["Wearing Masks","Sanitization from market"])



    # Take into consideration variables that have a correlation > 0.10 with the Label (COVID-19 presence)
    FEATURES = [
        "Breathing Problem",
        "Fever",
        "Dry Cough",
        "Sore throat",
        "Abroad travel",
        "Contact with COVID Patient",
        "Attended Large Gathering",
        "Family working in Public Exposed Places",
        "Visited Public Exposed Places"
        ]


    # Stage 01 : Encode the features Columns

    
    stages = [] # stages in our Pipeline
    for col in FEATURES:
        stringIndexer = StringIndexer(inputCol=col, outputCol=col + "Index")
        stages += [stringIndexer]
    
    
    # Stage 02 : Encode the Label
    stringIndexer = StringIndexer(inputCol=config.LABEL_NAME, outputCol="label")
    stages += [stringIndexer]

    
    # Stage 03 : Assemble the features into one dense vector
    indexedFeatures = [feature + "Index" for feature in FEATURES]

    assembler = VectorAssembler(inputCols=indexedFeatures, outputCol="features")
    stages += [assembler]

    # Building The PIPELINE

    

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    model = pipelineModel.transform(df)
    data = model.select("label","features")
    return data