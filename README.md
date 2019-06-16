# BERT-toxic-classification
# Bag-of-words
In order to run the Bag-of-Words model in Peregrine, the jobscriptBOW.sh file can be used. Also, the following packages should be installed: matplotlib, numpy, pandas, sklearn and seaborn.

# BERT
1)Run tfrecordgenertor.py to generate tensorflow tfrecord of the dataset
2)Run BERT_tensorflow.py to train the model
3)Run BERT_evaluator.py to check the how well your network is trained
4)Run BERT_predict.py to generate a csv file of the predictions from BERT.
5)Run accuracy_calculator.py to calculate metrics. 
