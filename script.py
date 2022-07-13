from sparknlp.annotator import *

tokenClassifier = BertForTokenClassification.loadSavedModel(
     f'./{PROJECT_NAME}/{MODEL_NAME_TF}/model_weights',
     spark)\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")\
  .setCaseSensitive(True)\
  .setMaxSentenceLength(128) # 512

tokenClassifier.write().overwrite().save("spark_nlp")

 #Test the imported model in Spark NLP

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols("sentence")\
  .setOutputCol("token")

tokenClassifier = BertForTokenClassification.load("spark_nlp")\
  .setInputCols("token", "sentence")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","label"])\
  .setOutputCol("ner_chunk")


pipeline =  Pipeline(
    stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  tokenClassifier,
  ner_converter
    ]
)

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))


text = A Caucasian man in his 40s was admitted in the spring of 2021 to his local hospital, 6 days after the first non-specific symptoms of COVID-19. His medical history included previous hemithyroidectomy (in 2016, benign histology), surgical resection of thymoma (2017) and pneumonia caused by Mycoplasma pneumoniae (2016) and a successfully treated visceral leishmaniasis from which he suffered after a sailing holiday in the Mediterranean Sea (2017). During routine follow-up after thymectomy, a CT thorax with intravenous contrast in 2020 was unremarkable except for dependent atelectasis. '

result = p_model.transform(spark.createDataFrame([[text]]).toDF('text'))

result.select(F.explode(F.arrays_zip('token.result', 'label.result')).alias("cols")) \
      .select(F.expr("cols['0']").alias("token"),
              F.expr("cols['1']").alias("label")).show(50, truncate=False)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)