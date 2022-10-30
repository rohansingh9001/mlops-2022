import matplotlib.pyplot as plt
import math
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, tree, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

svm_accuracies = []
print("SVM")
for i in range(5):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, predicted)
    print(acc)
    svm_accuracies.append(acc)

svm_mean = sum(svm_accuracies)/5
svm_std = math.sqrt(sum([(i-svm_mean)**2 for i in svm_accuracies])/5)
print("SVM Mean: ", svm_mean)
print("SVM std: ", svm_std, "\n")

tree_accuracies = []
print("Decision Tree")
for i in range(5):
    # Create a classifier: a support vector classifier
    clf = tree.DecisionTreeClassifier(max_features=500, max_depth=70, max_leaf_nodes=1200)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, predicted)
    print(acc)
    tree_accuracies.append(acc)

tree_mean = sum(tree_accuracies)/5
tree_std = math.sqrt(sum([(i-tree_mean)**2 for i in tree_accuracies])/5)
print("Tree Mean: ", tree_mean)
print("Tree std: ", tree_std, "\n")
