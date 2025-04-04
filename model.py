# model.py
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y, use_knn=True):  # Default to KNN for speed
    print("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_knn:
        print("Using KNN classifier...")
        classifier = KNeighborsClassifier(n_neighbors=5)
    else:
        print("Using SVM classifier...")
        classifier = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
    
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return classifier

def predict_digit(model, input_data):
    return model.predict([input_data])[0]