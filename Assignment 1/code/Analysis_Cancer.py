import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

fig_path = '../figures/'

# load dataset[1] Breast Cancer Data
df = pd.read_csv('../data/cancer.csv')

# drop un-needed columns
df.drop(['id'], axis=1, inplace=True)
df.drop(['Unnamed: 32'], axis=1, inplace=True)

# update target variable from text to binary expression (Malignant to 1; benign to 0)
df.loc[df['diagnosis'] == "M", 'diagnosis'] = 1
df.loc[df['diagnosis'] == "B", 'diagnosis'] = 0

# update the target column datatype from object to float otherwise cannot fit the data
df['diagnosis'] = df['diagnosis'].astype(float)

df.head()

id = 903657078

# split dataset into training and testing
x = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=id)

# check the dataset size and target distribution
print("total number of examples in the dataset is:", x.shape[0])
print()
print("porpotion of Malignant cases is: %.2f%% " % (y[y==1].shape[0]/y.shape[0]*100.0))

# normalize the data
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


# plot algorithms as a function of training sizes
def modeltime(x, y, model, title):
    train_time = []
    predict_time = []
    percentage = np.arange(0.2, 1.0, 0.2)

    for i in percentage:
        train_rows = int(i * df.shape[0])
        test_rows = df.shape[0] - train_rows

        xtrain = x[:train_rows]
        ytrain = y[:train_rows]
        xtest = x[:test_rows]

        # set up learner and calculate training and querying time
        train_start = time.time()
        model.fit(xtrain, ytrain)
        train_end = time.time()

        y_pred = model.predict(xtest)
        predict_end = time.time()

        train_time.append(train_end - train_start)
        predict_time.append(predict_end - train_end)

    plt.plot(percentage, train_time, 'o-', label='Training time', lw=1)
    plt.plot(percentage, predict_time, 'o-', label='Predicting time', lw=1)
    plt.xlabel('Dataset Usage %')
    plt.ylabel('Speed (sec)')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Model Time: ' + title)
    plt.savefig(fig_path + 'model time_' + title)
    plt.clf()

# ========================== Decision Tree ==============================
# default decision tree
dtree = DecisionTreeClassifier(random_state=id, criterion='gini')
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)
dtree_accuracy = accuracy_score(y_test, y_pred)
print('The accuracy score of default decision tree is %.2f%%' % (dtree_accuracy*100))

# Validation
tree_depth = np.arange(16)
train_scores, test_scores = validation_curve(dtree, x_train, y_train,
                                             param_name='max_depth', param_range=tree_depth,
                                             scoring='accuracy', cv=5)
plt.figure()

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(tree_depth, train_scores_mean, 'o-', color='red', label='training score')
plt.fill_between(tree_depth, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='red')

plt.plot(tree_depth, test_scores_mean, 'o-', color='blue', label='validation score')
plt.fill_between(tree_depth, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='blue')

plt.title('Validation Curve: Decision Tree for Cancer')
plt.xlabel('tree_depth')
plt.ylabel('accuracy score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'dtree_validation_curve_cancer.png')
plt.clf()

# Parameter Tuning
params_name={'max_depth': range(1, 16, 1),
             'min_samples_split': list(range(2, 10))}
dtree_search = GridSearchCV(estimator=dtree,
                            param_grid=params_name,
                            scoring='accuracy',
                            n_jobs=4,
                            cv=5)

dtree_search.fit(x_train, y_train)

optimal_dtree_params = dtree_search.best_params_
print('The optimal depth for decision tree is:')
print(optimal_dtree_params)

optimal_dtree_score = dtree_search.best_score_
print()
print('The best score for decision tree is %.2f%%' % (optimal_dtree_score*100))

y_pred = dtree_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print()
print('Test accuracy of decision tree is %.2f%%' % (accuracy * 100))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
dt_clf = DecisionTreeClassifier(max_depth=optimal_dtree_params['max_depth'],
                                min_samples_split=optimal_dtree_params['min_samples_split'],
                                criterion='gini',
                                random_state=id)
train_sizes, train_scores, test_scores = learning_curve(dt_clf, x_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='Validation score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Learning Curves: Decision Tree for Cancer')
plt.xlabel('Training Sizes')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'dtree_learning_curve_cancer.png')
plt.clf()

# model time
modeltime(x_train, y_train, dt_clf, 'Decision Tree for Cancer' )

# Cost Complexity Pruning
dtree = DecisionTreeClassifier(random_state=id)
path = dtree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

param_name = {'ccp_alpha': [i for i in ccp_alphas]}
ccp_alphas_search = RandomizedSearchCV(dtree, scoring='accuracy', param_distributions=param_name, cv=5, n_jobs=4)
ccp_alphas_search.fit(x_train, y_train)
print('best ccp_alpha for decision tree is: ', ccp_alphas_search.best_params_)

train_sizes = np.linspace(0.1, 1.0, 5)
tree = DecisionTreeClassifier(ccp_alpha=ccp_alphas_search.best_params_['ccp_alpha'],
                                criterion='gini',
                                random_state=id)
train_sizes, train_scores, test_scores = learning_curve(tree, x_train, y_train, train_sizes=train_sizes, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='Validation score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Learning Curves: Decision Tree ccp_alpha adjusted for Cancer')
plt.xlabel('Training Sizes')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'dtree_ccp_alpha_adjusted_for_cancer.png')
plt.clf()

# ================================== Neural Network ============================
nn = MLPClassifier(random_state=id, max_iter=500)
nn.fit(x_train, y_train)
y_pred = nn.predict(x_test)
nn_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy score of default neural network is %.2f%%' % (nn_accuracy * 100))

# validation Curve
# Regularization parameter
alpha_range = np.logspace(-5, 3, 5)
train_scores, test_scores = validation_curve(nn, x_train, y_train, param_name="alpha", param_range=alpha_range, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.semilogx(alpha_range, train_scores_mean, 'o-', color='r', label='Training score')
plt.fill_between(alpha_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.semilogx(alpha_range, test_scores_mean, 'o-', color='b', label='Validation score')
plt.fill_between(alpha_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Validation Curve: Neural Network for Cancer')
plt.xlabel('L2 penalty alpha')
plt.ylabel("Score")
plt.legend(loc="best")
plt.savefig(fig_path + 'nn_validation_curve_alpha_cancer.png')
plt.clf()

# Learning rate
lr_range = np.logspace(-4, 0, 6)
train_scores, test_scores = validation_curve(nn, x_train, y_train, param_name="learning_rate_init", param_range=lr_range, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.semilogx(lr_range, train_scores_mean, 'o-', color='r', label='Training score')
plt.fill_between(lr_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.semilogx(lr_range, test_scores_mean, 'o-', color='b', label='Validation score')
plt.fill_between(lr_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Validation Curve: Neural Network for Cancer')
plt.xlabel('Initial learning rate')
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(fig_path + 'nn_validation_curve_lr_cancer.png')
plt.clf()

# Parameter Tuning
# Define grid for search after validation curves
alpha_range = np.logspace(-4, 0, 5)
lr_range = np.logspace(-4, -2, 5)
nn = MLPClassifier(random_state=id, early_stopping=True)
params_name = {'alpha': alpha_range,
               'learning_rate_init': lr_range,
               'hidden_layer_sizes': [(5,), (15,), (30, )],
               'activation': ['logistic', 'relu'],
               'solver': ['sgd', 'adam']}

nn_search = GridSearchCV(nn, param_grid=params_name, n_jobs=4, cv=5)

nn_search.fit(x_train, y_train)
optimal_nn_params = nn_search.best_params_
print("Best parameters for Neural Netwrok are:")
print(optimal_nn_params)

optimal_nn_score = nn_search.best_score_
print()
print('The best score for Neural Network is %.2f%%' % (optimal_nn_score*100))

y_pred = nn_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy of neural network is %.2f%%' % (accuracy * 100))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
nn_clf = MLPClassifier(hidden_layer_sizes=optimal_nn_params['hidden_layer_sizes'],
                       activation=optimal_nn_params['activation'],
                       solver=optimal_nn_params['solver'],
                       alpha=optimal_nn_params['alpha'],
                       learning_rate_init=optimal_nn_params['learning_rate_init'],
                       random_state=id,
                       max_iter=500)
train_sizes, train_scores, test_scores = learning_curve(nn_clf, x_train, y_train, train_sizes=train_sizes, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='Validation score')
plt.title('Learning Curve: Neural Network Cancer')
plt.xlabel('Fraction of training examples')
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(fig_path + 'nn_learning_curve_cancer.png')
plt.clf()

# Loss Curve
nn = MLPClassifier(hidden_layer_sizes=optimal_nn_params['hidden_layer_sizes'], random_state=id, max_iter=1, warm_start=True)
nn.set_params(alpha=optimal_nn_params['alpha'],
              learning_rate_init=optimal_nn_params['learning_rate_init'],
              activation=optimal_nn_params['activation'],
              solver=optimal_nn_params['solver'])
num_epochs = 300
train_loss = np.zeros(num_epochs)
train_scores = np.zeros(num_epochs)
val_scores = np.zeros(num_epochs)
# Split training set into training and validation
x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=id)
for i in range(num_epochs):
    nn.fit(x_train1, y_train1)
    train_loss[i] = nn.loss_
    train_scores[i] = accuracy_score(y_train1, nn.predict(x_train1))
    val_scores[i] = accuracy_score(y_val, nn.predict(x_val))

y_pred = nn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of neural network is %.2f%%' % (accuracy * 100))

xrange = np.arange(num_epochs) + 1
plt.figure()
plt.plot(xrange, train_loss)
plt.title('Training Loss: Neural Network for Cancer')
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.grid()
plt.savefig(fig_path + 'nn_training_loss_cancer.png')
plt.clf()

plt.figure()
plt.plot(xrange, train_scores, label='Training score', color='r')
plt.plot(xrange, val_scores, label='Validation score', color='b')
plt.title('Training and Validation Score: Neural Network for Cancer')
plt.xlabel('Epochs')
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(fig_path + 'nn_training_validation_score_cancer.png')
plt.clf()

# Modeltime
modeltime(x_train, y_train, nn_clf, 'Neural Network for Cancer Data')

# ========================== Boosting ====================================

adaboost = AdaBoostClassifier(base_estimator=dt_clf, random_state=id)
adaboost.fit(x_train, y_train)
y_pred = adaboost.predict(x_test)
adaboost_accuracy = accuracy_score(y_test, y_pred)
print('The accuracy score of default AdaBoost is %.2f%%' %(adaboost_accuracy*100))

# Validation Curve
# Plot training and cv scores for the number of learners.
learners = np.arange(100)+1
train_scores, test_scores = validation_curve(adaboost, x_train, y_train,
                                             param_name='n_estimators', param_range=learners,
                                             scoring='accuracy', n_jobs=4, cv=5)
plt.figure()
plt.xticks(np.arange(min(learners), max(learners), 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(learners, train_scores_mean, color='r', label='Training score')
plt.fill_between(learners, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.plot(learners, test_scores_mean, color='b', label='Validation score')
plt.fill_between(learners, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Validation Curve: AdaBoost for Cancer')
plt.xlabel('Number of Learners')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'ab_validation_curve_cancer.png')
plt.clf()

# Parameter Tuning
params_name = {'n_estimators': range(1, 101, 1)}
ab_search = GridSearchCV(estimator=adaboost, param_grid=params_name, scoring='accuracy', n_jobs=4, cv=5)

ab_search.fit(x_train, y_train)

optimal_ab_params = ab_search.best_params_
print("Best parameters for AdaBoosting are:")
print(optimal_ab_params)

optimal_ab_score = ab_search.best_score_
print()
print('The best score for AdaBoosting is %.2f%%' % (optimal_ab_score*100))

y_pred = ab_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy of AdaBoosting is %.2f%%' % (accuracy * 100))

# Learning Curve
train_sizes = np.arange(0.2, 1.0, 0.2)
ab_clf = AdaBoostClassifier(base_estimator=dt_clf,
                            n_estimators=optimal_ab_params['n_estimators'],
                            random_state=id)
train_sizes, train_scores, test_scores = learning_curve(ab_clf, x_train, y_train, train_sizes=train_sizes, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='validation score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Learning Curves: AdaBoosting for Cancer')
plt.xlabel('Fraction of training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'ab_learning_curve_cancer.png')
plt.clf()

# Modeltime
modeltime(x_train, y_train, ab_clf, 'AdaBoosting for Cancer')

# =========================== SVM ==================================
# start with SVM linear kernel
def hyperSVM(x_train, y_train, x_test, y_test):
    accuracy_train = []
    accuracy_test = []

    kernel_func = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in kernel_func:
        if i == 'poly':
            for j in [2, 4, 6]:
                clf = SVC(kernel=i, degree=j, random_state=id)
                clf.fit(x_train, y_train)
                y_pred_train = clf.predict(x_train)
                y_pred_test = clf.predict(x_test)
                accuracy_train.append(accuracy_score(y_train, y_pred_train))
                accuracy_test.append(accuracy_score(y_test, y_pred_test))
        else:
            clf = SVC(kernel=i, random_state=id)
            clf.fit(x_train, y_train)
            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            accuracy_train.append(accuracy_score(y_train, y_pred_train))
            accuracy_test.append(accuracy_score(y_test, y_pred_test))

    xvals = ['linear', 'poly2', 'poly4', 'poly6', 'rbf', 'sigmoid']
    plt.plot(xvals, accuracy_train, 'o-', color='b', label='Training Score')
    plt.plot(xvals, accuracy_test, 'o-', color='r', label='Testing Score')

    plt.xlabel('Kernel Function')
    plt.ylabel('Model Accuracy Score')

    plt.title('SVM different kernels: Cancer')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.savefig(fig_path + 'svm_kernel_accuracy_cancer.png')
    plt.clf()

hyperSVM(x_train, y_train, x_test, y_test)

# Validation Curve
C_range = np.logspace(-3, 3, 7)

train_scores, test_scores = validation_curve(SVC(random_state=id),
                                             x_train, y_train,
                                             param_name="C", param_range=C_range,
                                             scoring="accuracy",
                                             n_jobs=4,
                                             cv=5)
plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(C_range, train_scores_mean, color='r', label='Training score')
plt.fill_between(C_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.plot(C_range, test_scores_mean, color='b', label='Validation score')
plt.fill_between(C_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Validation Curve: SVM for Cancer')
plt.xlabel('Regularization parameter')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'svm_validation_curve_cancer.png')
plt.clf()

# Parameter Tuning
parameters = {'C': [0.001, 0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.1, 1, 10, 100],
              'kernel': ['rbf', 'linear']}

svm_search = GridSearchCV(estimator=SVC(random_state=id), param_grid=parameters, scoring='accuracy', n_jobs=4, cv=5)
svm_search.fit(x_train, y_train)

optimal_svm_params = svm_search.best_params_
print("Best parameters for SVM are:")
print(optimal_svm_params)

optimal_svm_score = svm_search.best_score_
print()
print('The best score for SVM is %.2f%%' % (optimal_svm_score*100))

y_pred = svm_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy of SVM is %.2f%%' % (accuracy * 100))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
svm_clf = SVC(C=optimal_svm_params['C'],
              kernel=optimal_svm_params['kernel'],
              gamma=optimal_svm_params['gamma'],
              random_state=id)

train_sizes, train_scores, test_scores = learning_curve(svm_clf, x_train, y_train, train_sizes=train_sizes, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='validation score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Learning Curves: SVM for Cancer')
plt.xlabel('Fraction of training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'svm_learning_curve_cancer.png')
plt.clf()

# iterate over times
iteration = range(100)
train_scores = []
test_scores = []

for it in iteration:
    svc = SVC(kernel='rbf', max_iter=it)
    svc.fit(x_train, y_train)
    train_score = accuracy_score(y_train, svc.predict(x_train))
    test_score = accuracy_score(y_test, svc.predict(x_test))
    train_scores.append(train_score)
    test_scores.append(test_score)

plt.plot(iteration, train_scores, color='r', label='train')
plt.plot(iteration, test_scores, color='b', label='test')
plt.xticks(np.arange(0, 110, 10))
plt.title('Learning Curves: SVM iteration for Cancer')
plt.xlabel('iterations')
plt.ylabel('train test accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'svm_iteration_cancer.png')
plt.clf()

# Modeltime
modeltime(x_train, y_train, svm_clf, 'SVM for Cancer')

# =============================== KNN ==============================
# set up a default KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, y_pred)
print('The accuracy score of default KNN is %.2f%%' %(knn_accuracy*100))

# Validation Curve
num_neighbors = np.arange(1, 101)

train_scores, test_scores = validation_curve(knn, x_train, y_train, param_name='n_neighbors', param_range=num_neighbors, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(num_neighbors, train_scores_mean, color='r', label='Training score')
plt.fill_between(num_neighbors, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.plot(num_neighbors, test_scores_mean, color='b', label='Validation score')
plt.fill_between(num_neighbors, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Validation Curve: KNN for Cancer')
plt.xlabel('Number of Neighbors')
plt.ylabel('Scores')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'knn_validation_curve_cancer.png')
plt.clf()

# Parameter Tuning
parameters = {'n_neighbors': np.arange(1,20),
             'weights': ['uniform', 'distance']}

knn_search = GridSearchCV(knn, param_grid=parameters, scoring='accuracy', cv=5)

knn_search.fit(x_train, y_train)

optimal_knn_params = knn_search.best_params_
print("Best parameters for KNN are:")
print(optimal_knn_params)

optimal_knn_score = knn_search.best_score_
print()
print('The best score for KNN is %.2f%%' %(optimal_knn_score*100))


y_pred = knn_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy of KNN is %.2f%%' % (accuracy * 100))

# Learning Curve
train_sizes = np.arange(0.1, 1.0, 0.1)
knn_clf = KNeighborsClassifier(n_neighbors=optimal_knn_params['n_neighbors'],
                               weights=optimal_knn_params['weights'])
train_sizes, train_scores, test_scores = learning_curve(knn, x_train, y_train, train_sizes=train_sizes, n_jobs=4, cv=5)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='validation score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')

plt.title('Learning Curves: KNN for Cancer')
plt.xlabel('Fraction of training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig(fig_path + 'knn_learning_curve_cancer.png')
plt.clf()

# Modeltime
modeltime(x_train, y_train, knn_clf, 'KNN for Cancer')

# ===================== Algorithms Comparison =========================
# Accuracy
time_dict = dict()
test_dict = dict()

start_time = time.time()
dt_clf.fit(x_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
time_dict['DecisionTreeClassifier'] = [time_taken]

start_time = time.time()
dt_clf.predict(x_test)
end_time = time.time()
time_taken = end_time - start_time
test_dict['DecisionTreeClassifier'] = [time_taken]

start_time = time.time()
nn_clf.fit(x_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
time_dict['MLPClassifier'] = [time_taken]

start_time = time.time()
nn_clf.predict(x_test)
end_time = time.time()
time_taken = end_time - start_time
test_dict['MLPClassifier'] = [time_taken]

start_time = time.time()
ab_clf.fit(x_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
time_dict['AdaBoostClassifier'] = [time_taken]

start_time = time.time()
ab_clf.predict(x_test)
end_time = time.time()
time_taken = end_time - start_time
test_dict['AdaBoostClassifier'] = [time_taken]

start_time = time.time()
svm_clf.fit(x_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
time_dict['SVMClassifier'] = [time_taken]

start_time = time.time()
svm_clf.predict(x_test)
end_time = time.time()
time_taken = end_time - start_time
test_dict['SVMClassifier'] = [time_taken]

start_time = time.time()
knn_clf.fit(x_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
time_dict['KNNClassifier'] = [time_taken]

start_time = time.time()
knn_clf.predict(x_test)
end_time = time.time()
time_taken = end_time - start_time
test_dict['KNNClassifier'] = [time_taken]

clf_dict = {'DecisionTreeClassifier': dt_clf,
            'MLPClassifier': nn_clf,
            'AdaBoostClassifier': ab_clf,
            'SVMClassifier': svm_clf,
            'KNNClassifier': knn_clf}


# Function to generate training and testing accuracy scores
def fun(clf_dict, x_train, y_train, x_test, y_test):
    d = dict()
    for i in clf_dict.keys():
        l = list()
        train_pred = clf_dict[i].predict(x_train)
        auc_score = accuracy_score(y_train, train_pred)
        l.append(auc_score)

        test_pred = clf_dict[i].predict(x_test)
        auc_score = accuracy_score(y_test, test_pred)
        l.append(auc_score)

        # Adding difference of score (train-test)
        diff = l[0] - l[1]
        l.append(diff)

        d[i] = l

    return d

# Calling the above function fun() to predict and then compare roc_auc score of different models
scores = fun(clf_dict, x_train, y_train, x_test, y_test)

# Merging training time in scores dictionary for better visualization
for i in scores.keys():
    scores[i] = scores[i] + time_dict[i] + test_dict[i]

df = pd.DataFrame.from_dict(scores)
df = df.rename(index = {0: 'Train_Score', 1: 'Test_Score', 2: 'Score_Diff', 3: 'Training_Time', 4: 'Predicting_Time'})
df.plot(kind='barh', figsize=(10, 8), zorder=2, width=0.85)
plt.title('Model performance for cancer')
plt.savefig(fig_path + 'algorithms_performance_cancer.png')
plt.clf()