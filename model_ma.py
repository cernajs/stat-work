import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot', type=bool, default=False, const=True, nargs='?')
parser.add_argument('--grid_search', type=bool, default=False, const=True, nargs='?')
parser.add_argument('--dist', type=bool, default=False, const=True, nargs='?')

args = parser.parse_args()

columns_wind = ["DATE","FG","moving_average_FG"]
columns_cloud = ["DATE","CC","moving_average_CC"]

cloud = pd.read_csv("CC_ma.csv", names=columns_cloud)
wind = pd.read_csv("FG_ma.csv", names=columns_wind)

cloud['DATE'] = cloud['DATE'].astype(str)
wind['DATE'] = wind['DATE'].astype(str)

prices = pd.read_csv('price.csv')

prices['date'] = prices['date'].astype(str)

oil_prices = pd.read_csv('oil.csv')

oil_prices['date'] = oil_prices['date'].astype(str)

temperatures = pd.read_csv('TG_ma.csv')

temperatures['DATE'] = temperatures['DATE'].astype(str)

#temperatures fill na
temperatures = temperatures.fillna(0)

merged_df = pd.merge(cloud, wind, on='DATE')
merged_df = pd.merge(merged_df, prices, left_on='DATE', right_on='date')
merged_df = pd.merge(merged_df, oil_prices, left_on='DATE', right_on='date')
#merged_df = pd.merge(merged_df, temperatures, left_on='DATE', right_on='DATE')

"""
merged_df = pd.merge(wind, prices, left_on='DATE', right_on='date')
merged_df = pd.merge(merged_df, oil_prices, left_on='DATE', right_on='date')
merged_df = pd.merge(merged_df, cloud, left_on='DATE', right_on='DATE')
"""

merged_df['price_lag_1'] = merged_df['price'].shift(1)
#merged_df['price_lag_2'] = merged_df['price'].shift(2)
merged_df['price_lag_3'] = merged_df['price'].shift(3)
merged_df['price_lag_4'] = merged_df['price'].shift(4)
merged_df['price_lag_5'] = merged_df['price'].shift(5)
merged_df['price_lag_6'] = merged_df['price'].shift(6)
merged_df['price_lag_7'] = merged_df['price'].shift(7)

merged_df['price_lag_20'] = merged_df['price'].shift(20)
merged_df['price_lag_50'] = merged_df['price'].shift(50)

merged_df['price_lag_ma_50'] = merged_df['price_lag_1'].rolling(window=50).mean()

X = merged_df.drop(columns=['price'])
y = merged_df['price']

#X = X.drop(columns=['DATE','date_x','date_y','moving_average_FG'])
X = X.drop(columns=['DATE','date_x','date_y'])
#X = X.drop(columns=['date'])

#X = X.drop(columns=['FG','moving_average_FG','CC','moving_average_CC'])
X = X.drop(columns=['moving_average_FG','moving_average_CC'])

X['moving_average_FG'] = X['FG'].rolling(window=50).mean()

#X['moving_average_CC'] = X['CC'].rolling(window=14).mean()

#select only [val,FG] from X
#X = X[['val','FG', 'price_lag']]

#add to X feature of its own lag
#X['val_lag_1'] = X['val'].shift(1)
#X['val_lag_2'] = X['val'].shift(2)
#X['val_lag_3'] = X['val'].shift(3)
#X['val_lag_4'] = X['val'].shift(4)
#X['val_lag_5'] = X['val'].shift(5)
#X['val_lag_6'] = X['val'].shift(6)
#X['val_lag_7'] = X['val'].shift(7)

X = X.fillna(0)

import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Lasso(alpha=1)

"""
model = sklearn.pipeline.Pipeline([
    ("polynomial", sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge", Ridge(alpha=100))
])
"""

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'R^2: {r2_score(y_test, y_pred)}')
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')

#poly_features = model.named_steps['polynomial'].get_feature_names_out(input_features=X.columns)
#coefficients = pd.Series(model.named_steps['ridge'].coef_, index=poly_features)
coefficients = pd.Series(model.coef_, index=X.columns)

significant_stations = coefficients[coefficients.abs() > 0].sort_values(ascending=False)

print("Top stations affecting price:")
print(significant_stations.head(10))

import numpy as np
from scipy import stats

X_train_np = X_train.to_numpy()

#convert all columns to float
X_train_np = X_train_np.astype(float)

""""
X_train_poly = model.named_steps['polynomial'].transform(X_train_np)

XtX_inv = np.linalg.inv(np.dot(X_train_poly.T, X_train_poly))
sigma_squared = np.var(y_train - model.predict(X_train))  # Residual variance
standard_errors = np.sqrt(np.diagonal(XtX_inv) * sigma_squared)
"""
standard_errors = np.sqrt(np.diagonal(np.linalg.inv(np.dot(X_train_np.T, X_train_np)) * np.var(y_train - model.predict(X_train))))
t_stats = coefficients / standard_errors
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=len(X_train) - len(X.columns) - 1)) for t in t_stats]

significance = pd.DataFrame({'Coefficient': coefficients, 't_stat': t_stats, 'p_value': p_values})
print("\nCoefficient Significance:")
print(significance)

"""
ssr = np.sum((model.predict(X_train) - y_train.mean()) ** 2)
sse = np.sum((y_train - model.predict(X_train)) ** 2)
f_stat = (ssr / (len(X.columns) - 1)) / (sse / (len(X_train) - len(X.columns)))
f_p_value = 1 - stats.f.cdf(f_stat, len(X.columns) - 1, len(X_train) - len(X.columns))

print(f"\nF-statistic: {f_stat}")
print(f"F-test p-value: {f_p_value}")
"""

daily_increase = merged_df['price'].diff().dropna()

import matplotlib.pyplot as plt
import seaborn as sns

data = merged_df['price'].diff().dropna()

#standart scaler
#data = (data - data.mean()) / data.std()

#remove from data outliers
data = data[(np.abs(stats.zscore(data)) < 3)]

#k-fold cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')

for i, score in enumerate(scores):
    print(f'Fold {i+1}: {score}')

if args.dist:
    # Histogram
    sns.histplot(data, kde=True)
    plt.title('Histogram')
    plt.show()

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()

    ks_statistic, p_value = stats.kstest(data, 'norm')
    print(f"KS Statistic: {ks_statistic} P-Value: {p_value}")

    if p_value < 0.05:
        print("Data is not normally distributed")

    #scipy.stats.normaltest
    stat, p = stats.normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

if args.plot:


    # Calculate correlation matrix
    correlation_matrix = merged_df.drop(columns=['DATE','date_x','date_y']).corr()
    #correlation_matrix = merged_df.drop(columns=['date']).corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features and Electricity Price')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')

    plt.legend()
    plt.show()

    """
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    # Plot the original data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Original Data')

    # Plot the polynomial regression curve
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Fit')

    # Add labels and title
    plt.xlabel('Temperature (TG)')
    plt.ylabel('Price')
    plt.title('Polynomial Regression Fit')

    # Show legend
    plt.legend()

    # Show plot
    plt.show()
    """

if args.grid_search:
    import sklearn
    import sklearn.pipeline
    import sklearn.preprocessing

    model = sklearn.pipeline.Pipeline([
        ("polynomial", sklearn.preprocessing.PolynomialFeatures()),
        ("ridge", Ridge())
    ])

    model = sklearn.model_selection.GridSearchCV(
        model,
        {"polynomial__degree": [1, 2, 3, 4, 5],
        "ridge__alpha": [0.01, 0.1, 1, 10, 100]},
        #"ridge__solver": ["lbfgs", "sag"]},
    )

    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")

    for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
                                        model.cv_results_["mean_test_score"],
                                        model.cv_results_["params"]):
            print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                *("{}: {:<5}".format(key, value) for key, value in params.items()))
