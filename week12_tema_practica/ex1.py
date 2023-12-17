from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage import io, color, transform
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits


def preprocess_image(image_path):
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    resized_image = transform.resize(gray_image, (8, 8), mode='constant')

    flattened_image = resized_image.flatten()
    return flattened_image


def train_adaboost_classifier():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Increase max_depth and n_estimators
    base_classifier = DecisionTreeClassifier(max_depth=5)
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42, learning_rate=0.1)

    adaboost_classifier.fit(X_train, y_train)

    y_pred_train = adaboost_classifier.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy on training set: {accuracy_train}")

    y_pred_test = adaboost_classifier.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy on test set: {accuracy_test}")

    return adaboost_classifier


def predict_digit(image_path, adaboost_classifier):
    input_image = preprocess_image(image_path)

    predicted_digit = adaboost_classifier.predict([input_image])

    return predicted_digit


adaboost_classifier = train_adaboost_classifier()
for i in range(1, 10):
    image_path = f"{i}.png"
    predicted_digit = predict_digit(image_path, adaboost_classifier)
    print(f"Predicted digit for {i}.png: {predicted_digit}")
