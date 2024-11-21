import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report

# Load the data set into the dataframe
chrn_df = pd.read_csv('./datasets/ChurnData.csv')

# Selecting the columns that are required
chrn_df = chrn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
print(chrn_df.head())

# Let's define X and Y
X = np.asarray(chrn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(chrn_df[['churn']])

# Let's normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Train/test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Flatten y_train to a 1D array
y_train = y_train.ravel()

# Modelling logistic regression with scikit-learn
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# Print confirmation
print("Model training completed.")

# Predicting yhat
yhat = LR.predict(X_test)
print("Predictions: ", yhat)

# Predicting probability
yhat_prob = LR.predict_proba(X_test)
print("Prediction probabilities: \n", yhat_prob)

# Jaccard Index
jaccard = jaccard_score(y_test, yhat, pos_label=0)
print("Jaccard Index: ", jaccard)

# Confusion matrix plotting function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'], normalize=False, title='Confusion matrix')

# Show the plot
plt.show()

# Classification report
print("\nClassification Report:\n", classification_report(y_test, yhat))
