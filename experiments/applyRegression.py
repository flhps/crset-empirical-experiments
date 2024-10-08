from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
import pickle
import sys


def test_regressor(regressor, X_train, X_test, y_train, y_test):
    scores = cross_validate(regressor, X_train, y_train, cv=5)
    print("Cross-validation Score Mean:", scores["test_score"].mean())
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    if len(y_test.shape) > 1:
        for i in range(0, y_test.shape[1]):
            print("MAE (", i, "): ", mean_absolute_error(y_test[:, i], y_pred[:, i]))
    print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))
    if len(y_test.shape) > 1:
        for i in range(0, y_test.shape[1]):
            print(
                "MAPE (",
                i,
                "): ",
                mean_absolute_percentage_error(y_test[:, i], y_pred[:, i]),
            )
    print("R2: ", regressor.score(X_test, y_test))


def build_and_eval_regressor(X, y):
    # scikit seems to expect one-value predictions as (n_samples, ) instead of (n_samples, 1)
    if y.shape[1] == 1:
        y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
    print("--------------------------")
    print("Random Forest Regressor")
    print("--------------------------")
    reg1 = RandomForestRegressor(random_state=1)
    test_regressor(reg1, X_train, X_test, y_train, y_test)
    print("--------------------------")
    print("Linear Regressor")
    print("--------------------------")
    reg2 = LinearRegression()
    test_regressor(reg2, X_train, X_test, y_train, y_test)
    print("--------------------------")
    print("Decision Tree Regressor")
    print("--------------------------")
    reg3 = DecisionTreeRegressor()
    test_regressor(reg3, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as inp:
        d = pickle.load(inp)

    print("Training and evaluating...")
    build_and_eval_regressor(d[0], d[1])
    print("Done.")
