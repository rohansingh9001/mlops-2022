import flask
from flask import Flask, jsonify, request
# import argparse
import matplotlib.pyplot as plt
import joblib
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#parser = argparse.ArgumentParser()

#parser.add_argument("-cn", "--clf_name", help="Name of model, supported - svm or tree.")
#parser.add_argument("-rs", "--random_state", help="Seed for the random number generator.")

#args = parser.parse_args()

def get_train_test_split(ratio=0.2, seed=0):
	digits = datasets.load_digits()
	# flatten the images
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	return train_test_split(
	    data, digits.target, test_size=ratio, shuffle=True, random_state=seed
	)


def train_svm(data, gamma=0.001, C=1):
	X_train, X_test, y_train, y_test = data


	# Create a classifier: a support vector classifier
	clf = svm.SVC(gamma=gamma, C=C)

	# Learn the digits on the train subset
	clf.fit(X_train, y_train)

	return clf

def train_tree(data):
        X_train, X_test, y_train, y_test = data


        # Create a classifier: a support vector classifier
        clf = DecisionTreeClassifier(random_state=0)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        return clf

def test_model(model, data, args):
	_, X_test, __, y_test = data
	print("Model Accuracy:", metrics.accuracy_score(y_test, model.predict(X_test)))

	model_path = f'./models/{args.clf_name}_gamma=0.001_C=1_random_state={args.random_state}.joblib'

	with open(f"results/{args.clf_name}_{args.random_state}.txt", "w") as file:
		print("test accurancy:", metrics.accuracy_score(y_test, model.predict(X_test)), file=file)
		print("test macro-f1:", metrics.f1_score(y_test, model.predict(X_test), average="macro"), file=file)
		print(f"model saved at {model_path}", file=file)

	joblib.dump(model, model_path)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return "Machine Learning Prediction app"

@app.route('/predict', methods=['POST'])
def predict_image():
	prediction = 1
	return f"Prediction: {prediction}"

@app.route('/', methods=['GET'])
def best_model():
	mode_name = "svm_gamma=0.001_C=1_random_state=0.joblib"
        return "Best Model: {model_name}

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0')
