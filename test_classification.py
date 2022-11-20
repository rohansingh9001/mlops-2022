import numpy as np
import random
from classification import get_train_test_split, train_svm

def test_same_seed():
	seed = random.randint(0, 999999)

	X_train1, X_test1, y_train1, y_test1 = get_train_test_split(seed=seed)
	X_train2, X_test2, y_train2, y_test2 = get_train_test_split(seed=seed)

	X_train = X_train1 == X_train2
	X_test = X_test1 == X_test2
	y_train = y_train1 == y_train2
	y_test = y_test1 == y_test2

	assert X_train.all() and X_test.all() and y_train.all() and y_test.all()

def test_different_seed():
	seed = random.randint(0, 999999)

	X_train1, X_test1, y_train1, y_test1 = get_train_test_split(seed=seed)
	X_train2, X_test2, y_train2, y_test2 = get_train_test_split(seed=seed+1)

	X_train = np.array_equal(X_train1, X_train2)
	X_test = np.array_equal(X_test1, X_test2)
	y_train = np.array_equal(y_train1, y_train2)
	y_test = np.array_equal(y_test1, y_test2)

	assert not X_train and not X_test and not y_train and not y_test
