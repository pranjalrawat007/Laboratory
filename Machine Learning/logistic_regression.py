'''
Covers core aspects of Andrew Ng's Coursera Course
Topic: Week 3: Logistic Regression
Data: CSV from http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
Problem: Classification/SMS Spam-Ham Classification
Features = 4 Word Count Frequecy Based Features
Observations = 5.5k
'''

PATH = '/Users/pranjal/Google Drive/python_projects/projects/courses/andrew_ng/machine_learning/algorithms/logistic_regression/spam.csv'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.metrics import roc_auc_score


def import_df(PATH):
    data_pd = pd.read_csv(PATH, encoding="ISO-8859-1")
    data_pd['response'] = 0
    data_pd.loc[data_pd['v1'] == 'spam', 'response'] = 1
    data = data_pd[['response', 'v2']]
    data.columns = ['response', 'text']
    # text feature engineering
    data['text_len'] = data['text'].str.len()
    data['text_tokens'] = data['text'].apply(lambda x: len(str(x).split(" ")))

    def avg_word(sentence):
        words = sentence.split()
        return (sum(len(word) for word in words) / len(words))

    data['text_avg_word_len'] = data['text'].apply(lambda x: avg_word(str(x)))
    data['text_numerics'] = data['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    data = data[['response', 'text_len', 'text_tokens', 'text_avg_word_len', 'text_numerics']]
    df = np.array(data, 'float32')
    return df


def Xy(df):
    X = df[:, 1:]
    y = df[:, 0]
    y = y.reshape(y.shape[0], 1)
    return X, y


def featureScale(X):
    temp0 = X - X.mean(axis=0)
    temp1 = X.max(axis=0) - X.min(axis=0)
    return np.divide(temp0, temp1)


def randomTheta(X):
    m = X.shape[0]  # training examples
    n = X.shape[1] + 1  # parameters
    θ = np.random.normal(0, 1, (n, 1))
    return θ


def sigmoid(z):
    val = 1 + np.e ** -z
    return np.power(val, -1)


def hypothesis(X, θ):
    m = X.shape[0]  # training examples
    ones = np.ones((m, 1))
    Xd = np.hstack((ones, X))
    h = sigmoid(np.dot(Xd, θ))
    return h, Xd


def squaredLoss(X, y, θ, λ):
    m = X.shape[0]  # training examples
    h, Xd = hypothesis(X, θ)
    J = np.multiply(y, np.log(h)) + np.multiply(1 - y, np.log(1 - h))
    J = - sum(J)
    θ_temp = θ.copy()
    θ_temp[0] = 1
    J = J + λ * sum(np.multiply(θ_temp, θ_temp))
    J = J[0] / (1 * m)
    return J


def gradient_descent(X, y, eval_set, θ, λ, iterations, α, mom):
    X_val, y_val = eval_set[0], eval_set[1]
    m = X.shape[0]  # training examples
    delta = 0
    cost_path = []
    eval_path = []
    for i in range(iterations):
        cost_path.append([squaredLoss(X, y, θ, λ), squaredLoss(X_val, y_val, θ, λ)])
        h, Xd = hypothesis(X, θ)
        h_val, Xd_val = hypothesis(X_val, θ)
        eval_path.append([roc_auc_score(y, h), roc_auc_score(y_val, h_val)])
        θ_gradient = np.dot(Xd.T, h - y) + λ * θ
        θ_gradient = θ_gradient / m
        delta = mom * delta + θ_gradient
        θ= θ - α * delta
    return θ, cost_path, eval_path


def train_test_split(X, y, size):
    ''' X_train, X_test, y_train, y_test '''
    df = np.hstack((y, X))
    np.random.shuffle(df)
    X, y = Xy(df)
    m = round(y.shape[0] * size)
    return X[m:, :], X[0:m, :], y[m:, :], y[0:m, :]


def analyticalSolution(X, y):
    '''Unregularized Analytical Solution'''
    m = X.shape[0]  # training examples
    ones = np.ones((m, 1))
    Xd = np.hstack((ones, X))
    P = np.dot(Xd.T, Xd)
    P = np.linalg.inv(P)
    θ = np.dot(np.dot(P, Xd.T), y)
    return θ


# Load the data
np.random.seed(42)
print('\n\n')
print('----LOGISTIC REGRESSION MODEL----')

print('\n\n')
print('----LOAD DATA----')
df = import_df(PATH)
X, y = Xy(df)
print('X shape: ', X.shape)
print('y shape: ', y.shape)

print('\n\n')
print('----PREPROCESSING----')
X = featureScale(X)
print('Feature Scaling Completed!')


print('\n\n')
print('----INITIALISE----')
θ = randomTheta(X)
λ = 0
print('Initial L2 Regularization: ', λ)
print('Initial θ (random)(top 5 rows): \n', θ[0:5])
h, Xd = hypothesis(X, θ)
print('Initial Hypothesis (top 5 rows): \n', h[0:5])
print(f'Initial Cost at θ: {squaredLoss(X, y, θ, λ):.2f}')


print('\n\n')
print('----TRAIN-TEST SPLIT----')
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)


# Gradient Descent
θ, cost_path, eval_path = gradient_descent(X_train, y_train, [X_test, y_test], θ, λ, 500, 0.01, 0.9)

# Sklearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# Comparision
print('\n\n')
print('----RESULTS COMPARISION----')
print(f'Test AUC from Sklearn: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}')
print(f'Test AUC from Gradient Descent: {eval_path[-1][1]: .4f}')

# Plot Learning Curves for Gradient Descent Soln
plt.subplot(1, 2, 1)
cost_train = [i[0] for i in cost_path]
cost_val = [i[1] for i in cost_path]
plt.plot(cost_train, label='Cost Train')
plt.plot(cost_val, label='Cost Val')
plt.legend()
plt.title('Learning Curve - Cost')

plt.subplot(1, 2, 2)
R2_train = [i[0] for i in eval_path]
R2_val = [i[1] for i in eval_path]
plt.plot(R2_train, label='AUC Train')
plt.plot(R2_val, label='AUC Val')
plt.legend()
plt.title('Learning Curve - Eval Metric')
plt.show()
