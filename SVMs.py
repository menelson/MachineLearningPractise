import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC

# Make use of the smaller dataset, the iris dataset.
# In general SVMs work best with small- to medium-sized 
# datasets, because they have complexity O(m*n), for m 
# instances of n features 

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

# Setup a pipeline which first rescales the dataset, and then uses
# the SVM classifier. Also note that we're using the hinge loss
# as the cost function for the SVM. Also recall that the
# scaler is used to rescale the feature values such that they are 
# all within the same order of magnitude, typically between 0 and
# 1, or thereabouts. SVMs are sensitive to the feature scaling.
# The C-parameter also acts as a form of regularization.

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
])

svm_clf.fit(X, y)

# Can then make relevant predictions, following the above training (example below).
# A 2D array is expected, hence [[a, b]]
print('Classifier prediction: ', svm_clf.predict([[5.5, 1.7]]))

# The above can be adopted for work with non-linear classification problems, by
# either transforming the non-linear data so that the decision boundary becomes
# linear (using a transformer in the pipeline), or by employing a kernel. 

# Another approach to linear separation is to employ a similarity function.
# For example, one can employ a standard rescaling of the features and then
# also add a Gaussian RBF kernel.

# Use the popular Gaussian Radial Basis Function kernel, where gamma also serves as
# a form of regularization. 
rbf_kernel_svm_clf = Pipeline([
			("scaler", StandardScaler()),
			("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)) # gamma and C act like regularization parameters
	             ])

rbf_kernel_svm_clf.fit(X,y)

# The linear kernel and the Gaussian RBF kernels are recommended as the ones to use, particularly
# the Gaussian RBF if the dataset is small- to medium-sized.

# Now one can also use SVMs for regression by essentially inverting the problem. 
# Try and fit to the street/margin. 
svm_reg = LinearSVR(epsilon=1.5) # Epsilon sets the margin width
svm_reg.fit(X,y)
print('Regressor prediction: ', svm_reg.predict([[5.5, 1.7]]))
