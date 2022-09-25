import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
print(digits.images.shape)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

parameters = []
accuracies = []

for i in range(4):
	for j in range(3):
		parameters.append((0.0008 + 0.0001*i, 0.8 + 0.1*i))
print("parameters:", parameters)

for parameter in parameters:
	gamma, C = parameter
	# Create a classifier: a support vector classifier
	clf = svm.SVC(gamma=gamma, C=C)

	# Split data into 50% train and 50% test subsets
	X_train, X_test, y_train, y_test = train_test_split(
	    data, digits.target, test_size=0.5, shuffle=False
	)

	# Learn the digits on the train subset
	clf.fit(X_train, y_train)

	# Predict the value of the digit on the test subset
	predicted2 = clf.predict(X_test)

	#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
	#for ax, image, prediction in zip(axes, X_test, predicted):
	#    ax.set_axis_off()
	#    image = image.reshape(8, 8)
	#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
	#    ax.set_title(f"Prediction: {prediction}")
	accuracy2 = metrics.accuracy_score(y_test, predicted2)
	predicted1 = clf.predict(X_train)
	accuracy1 = metrics.accuracy_score(y_train, predicted1)
	accuracies.append((accuracy1, accuracy2))

	# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
	# disp.figure_.suptitle("Confusion Matrix")
	# print(f"Confusion matrix:\n{disp.confusion_matrix}")

	# plt.show()
print("gamma    C    train_acc    test_acc")
for parameter, accuracy in zip(parameters, accuracies):
	gamma, C = parameter
	train_acc, test_acc = accuracy
	print("%.4f" % gamma, '|', "%.1f" % C, '|', "%.7f" % train_acc, '|', "%.6f" % test_acc)

test_acc = [b for a, b in accuracies]
test_acc.sort()
print("max accuracy:", max(test_acc))
print("min accuracy:", min(test_acc))
print("mean accuracy:", sum(test_acc)/len(test_acc))
print("median accuracy:", test_acc[len(test_acc)//2])
