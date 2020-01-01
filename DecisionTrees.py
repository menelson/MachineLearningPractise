from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Want this for the visualization of the decision tree
from sklearn.tree import export_graphviz

# Load the iris data-set
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

# Train a standard decision tree classifier
# It's a very simple tree, with a depth of only two
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

'''
export_graphviz(
        tree_clf,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
'''

# SkLearn uses the CART algorithm to train decision trees, based
# on a cost function which seeks to minimize the impurities between
# the left/right subsets. Then it continues splitting the subsets further,
# based on this same cost function. 

# Two common impurity measures are used: Gini impurity (default), and 
# the Shannon entropy. Also need to ensure the tree is not overtraining to
# the data; this is common since the model is non-parametric. One can 
# introduce a natural regularization by restricting the maximum depth of
# the tree. 

# Can also perform regression using a decision tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)

# Note that the CART cost function for regression now
# tries to minimize the MSE

# The extension to the decision tree formalism is the Random Forest ... 
