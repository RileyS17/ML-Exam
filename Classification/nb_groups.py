# sklearn stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# other libraries
import pandas as pd
import numpy as np

def load_csv(file_name):
    url = file_name
    dataset = pd.read_csv(url)
    dataset = dataset.values
    return dataset

def preprocess_dataset(dataset):
    x = dataset[:, 0:45]
    # removing Q12 from the dataset
    x = np.delete(x, [33,34,35], 1)
    y = dataset[:, 45]
    category_list = np.unique(y)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # seperate 30% of the original dataset for validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test, category_list

def evaluate_prediction(y_test, y_predict):
    print('Naive Bayes Evaulation')
    print('-----------------------------------------')
    print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
    print("Precision:",metrics.precision_score(y_test, y_predict, average='weighted'))
    print("Recall:",metrics.recall_score(y_test, y_predict, average='weighted'))
    print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_predict))

def main():
    file_name = 'dataset.csv'
    dataset = load_csv(file_name)
    x_train, x_test, y_train, y_test, category_list = preprocess_dataset(dataset)
    model = GaussianNB()
    model.fit(x_train, y_train)
    print(model.predict_proba(x_test))
    y_predict = model.predict(x_test)
    evaluate_prediction(y_test, y_predict)
    

if __name__ == "__main__":
    main()