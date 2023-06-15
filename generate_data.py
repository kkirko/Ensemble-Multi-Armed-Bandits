import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data(n_samples=50000, n_features=50, random_state=42):
    np.random.seed(random_state)

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=18, random_state=random_state)
    num_features = X.shape[1]
    group_size = num_features // 3

    X_noisy_1 = X[:, :group_size] + 2 * np.random.normal(loc=0, scale=1, size=(X.shape[0], group_size))
    X_noisy_2 = X[:, group_size:(2 * group_size)] + 5 * np.random.normal(loc=0, scale=1, size=(X.shape[0], group_size))
    X_noisy_3 = X[:, (2 * group_size):] + 0.7 * np.random.normal(loc=0, scale=1, size=(X.shape[0], num_features - 2 * group_size))

    X_noisy = np.hstack([X_noisy_1, X_noisy_2, X_noisy_3])

    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.7, random_state=random_state)

    return X_train, X_test, y_train, y_test