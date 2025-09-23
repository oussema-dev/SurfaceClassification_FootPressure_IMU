from sklearn.feature_selection import VarianceThreshold

def reduce_features(train_set, features, variance_threshold=0.01, max_corr=0.80):
    """this is the feature reduction method

    Parameters:
    train_set (dataframe): train set used to reduce the features
    features (list): initial feature names
    variance_threshold (float): variance threshold used to identify quasi-constant features
    max_corr (float): threshold to keep non-correlated features

    Returns:
    reduced_features (list): reduced feature names
    """

    print("Reducing features...")

    X_train = train_set[features].copy()
    # Removing Constant features using variance threshold
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    # Get the names of the constant features
    constant_columns = [
        column
        for column in X_train.columns
        if column not in X_train.columns[constant_filter.get_support()]
    ]
    # Drop constant features
    X_train = X_train.drop(labels=constant_columns, axis=1)  # Reassign X_train

    # Drop quasi-constant features
    qconstant_filter = VarianceThreshold(threshold=variance_threshold)
    qconstant_filter.fit(X_train)
    # Get the names of the qconstant features
    qconstant_columns = [
        column
        for column in X_train.columns
        if column not in X_train.columns[qconstant_filter.get_support()]
    ]
    # Drop qconstant features
    X_train = X_train.drop(labels=qconstant_columns, axis=1)  # Reassign X_train

    # Drop duplicate features
    transpose = X_train.T
    unique_features = transpose.drop_duplicates(keep="first").T.columns
    duplicate_features = [x for x in X_train.columns if x not in unique_features]
    X_train = X_train.drop(labels=duplicate_features, axis=1)
    # Drop correlated features
    X_train = train_set[[*unique_features]].copy()
    correlated_features = set()
    correlation_matrix = X_train.corr()
    # Add the columns with a correlation value of max_corr to the correlated_features set
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > max_corr:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    X_train = X_train.drop(labels=correlated_features, axis=1)
    reduced_features = X_train.columns
    return reduced_features