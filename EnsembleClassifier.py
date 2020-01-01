'''
One can construct an ensemble classifier based on the systems of
hard or soft voting.
'''

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier # Averages over a large set of decision tree predictors
from sklearn.ensemble import VotingClassifier # Used to combined classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Pick up the moons dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# Boosting: train several weak learners sequentially, each trying to correct its predecessor by
# updating the weights corresponding to the worst predictors. A popular choice is AdaBoost.
from sklearn.ensemble import AdaBoostClassifier

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Instantiate the three classifiers 
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

# Construct and fit the voting classifier
voting_clf = VotingClassifier(
		estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
		voting='hard') # Can have hard or soft voting set here 
voting_clf.fit(X_train, y_train)

# Can also use bagging (sampling with replacement, or 'bootstrapping') on the training set, so that
# the same training set can be used to train multiple classifiers, from which
# the result of the ensemble can be combined (using soft voting) to produce
# a better combined predictor.


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500, # Bagging ensemble of 500 trees
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42, # 100 training instances, use all available cores (n_jobs = -1)
    oob_score=True) # Request an automatic out-of-bag score after training (training instances that are not sampled, and therefore can be used as a validation set)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# A combination of bagging and decision tree classification gives rise to a Random Forrest classifier. 
# Example:
rnd_clf_new = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf_new.fit(X_train, y_train)

y_pred_rf = rnd_clf_new.predict(X_test)
# Note: there is also a nice inbuilt method for measuring the relative importance of the different 
# features considered in the Random Forrest. This can be useful for e.g. feature selection.


ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200, # Trees of max_depth=1 (single tree, plus two leaf nodes) are 'Decision stumps'; here 200 are used in the training
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
