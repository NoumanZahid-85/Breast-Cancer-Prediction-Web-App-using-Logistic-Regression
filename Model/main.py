import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle



def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    ## Scale the data bcz features are on different scales
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    ## Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ## Create the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    ## Make predictions on the test set
    y_pred = model.predict(X_test)

    ## Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    class_report_str = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report_str)
    # X_train = scaler.transform(X_train)

    ## Return the trained model with the scaled data
    return model, scaler


def get_clean_data():
    data = pd.read_csv('./data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

  

def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    with open('./Model/logistic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('./Model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()