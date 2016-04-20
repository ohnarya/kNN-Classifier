## kNN-Classifier from scratch
k Nearest Neighbor Classifier

* Used [5-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) to evaluation performance
* Report precision, recall, [micro-avaraged F1-measure](https://en.wikipedia.org/wiki/F1_score)
* Written in ```Python```

## Input
data_matrix.csv

## Result 
```
Accuracy: 85.0%
Accuracy: 75.0%
Accuracy: 85.0%
Accuracy: 70.0%
Accuracy: 80.0%

Average Accuracy: 79.0%

Confusion Matrix : 
   64       4
   17      15

     TP rate   FP rate
Yes | 0.9412   0.5312 
No  | 0.4688   0.0588 

Precision : 0.7901
Recall    : 0.9412
Micro Averaged F1 score : 0.8591 # manually calculated

micro-averaged F1 score : 0.8591 # from sklearn.metrics import f1_score
```
