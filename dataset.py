# dataset preprocessing section
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values  #input features
    y = data.iloc[:, -1].values   #labels
    
    #transform text target labels into numerical ones
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    #normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
