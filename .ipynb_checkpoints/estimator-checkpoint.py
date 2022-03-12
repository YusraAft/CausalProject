import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import pdb


def naive(df):
    """
    Naive estimator that assumes E[Y^a] = E[Y | A]
    Should work when A is randomized
    """
    results = []
    for i in sorted(np.unique(df["a"])):
        est = np.mean(df[df["a"] == i]["y"])
        results.append(est)
    return np.array(results)


def naive_linear(df):
    """
    A naive linear estimator for E[Y^a]
    Assumes E[Y^a] = E[Y | A]
    Should work when A is randomized and A->Y is linear
    """
    params = smf.ols("y~a", data=df).fit().params.to_numpy()
    n_a = np.unique(df["a"]).shape[0]
    return np.stack([np.ones(n_a), np.arange(n_a)], axis=1).dot(params)


def bootstrap(df, function, n=1000, ci=95, **kwargs):
    """
    Resample the dataframe `n` times. For each resampled dataframe,
        call `function(new_df, **kwargs)` and save its value.
    Return the confidence interval that covers the central `ci`% of these values

    If the passed `function()` returns an array of shape [6,],
        then `bootstrap` should return an array of shape [2, 6]
        where the first row contains the bottom of the confidence interval
        and the second row contains the top of the confidence interval

    You may want to use `df.sample` and `np.percentile`
    """
    # empty.unique() -> 1d np array

    np.random.seed(42)  # Keep this random seed call for reproducibility
    values = []
    for i in range(n):
        length = df.shape[0]
        # number of samples in first dimension of df
        new_df = df.sample(length, replace=True)
        value = function(new_df, **kwargs)
        values.append(value)
    dif = (100 - ci) / 2
    low_bound = 0 + dif
    upper_bound = 100 - dif
    values = np.array(values)
    # print("values ", values)
    confidence = np.percentile(values, [low_bound, upper_bound], axis=0)
    ##print("confidence ", confidence)
    return confidence


def backdoor(df, confounders=["towncode", "tipuach"], intervention="classize", outcome="avgmath"):
    """
    A backdoor adjustment  estimator for E[Y^a]
    Use smf.ols to train an outcome model E(Y | A, confounders)

    Arguments
      df: a data frame for which to estimate the causal effect
      confounders: the variables to treat as confounders
          For the data we consider, if you include both c and d as confounders,
          this estimator should be unbiased. If you only include one,
          you would expect to see more bias.

    Returns
        results: an array of E[Y^a] estimates
    """
    # pdb.set_trace()
    # expectation of y intervened on a

    # model = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
    # A backdoor adjustment  estimator for E[Y^a]
    # Use smf.ols to train an outcome model E(Y | A, confounders)

    results = []

    # Create new model that will train to predict y and fit the model
    a_dim = [13, 26, 39, 52]

    # print(a_dim)  # should be 0,1,2,3,4,5
    prev = 0
    for a_ in a_dim:
        # print("params are intercept, a_param, confounders  ", res.params)
        # =1/n sum(i=1) to n ;model average of sum from 1 thru n
        subset = df[(df[intervention] <= a_) & (df[intervention] > prev)]
        prev = a_
        #df[df[intervention] >= a_]
        # print("subset ", subset)
        std = np.std(subset, axis=0)
        #print("std", std)

        formula = outcome + " ~ + {}".format(" + ".join(confounders))
        # print("formula ", formula)
        model = smf.ols(
            formula=formula, data=subset)

    # Use that model to predict E[Y] -> fit model with data and use it to predict given confounder
        res = model.fit()
        #matching = df[["c", "d"]]
        data_dict = {}
        for c in confounders:
            data_dict[c] = df[c]
        matching = pd.DataFrame(data_dict)
        # will give you only the two columns in the df which are c and d
        num_rows = df.shape[0]
        temp_col = [a_] * num_rows
        matching.insert(0, intervention, temp_col)
        predictions = res.predict(matching)
        # print(predictions)
        avg = np.mean(predictions)
        results.append(avg)

        # N by 2 matrix params for c and d
        # multiplied by [c_param, d_param] -> output is Nx1

    return np.array(results)


def backdoorfeedback(df, confounders=["townid", "schlcode" "tipuach"], intervention="classize", outcome="avgmath"):
    """
    This backdoor estimator has been adjusted for only 2 class sizes as per the provided feedback. 
    """
    # pdb.set_trace()
    # expectation of y intervened on a

    # model = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
    # A backdoor adjustment  estimator for E[Y^a]
    # Use smf.ols to train an outcome model E(Y | A, confounders)

    results = []

    # Create new model that will train to predict y and fit the model
    a_dim = [25, 52]

    # print(a_dim)  # should be 0,1,2,3,4,5
    prev = 0
    for a_ in a_dim:
        # print("params are intercept, a_param, confounders  ", res.params)
        # =1/n sum(i=1) to n ;model average of sum from 1 thru n
        subset = df[(df[intervention] <= a_) & (df[intervention] > prev)]
        prev = a_
        #df[df[intervention] >= a_]
        # print("subset ", subset)
        std = np.std(subset, axis=0)
        #print("std", std)

        formula = outcome + " ~ + {}".format(" + ".join(confounders))
        # print("formula ", formula)
        model = smf.ols(
            formula=formula, data=subset)

    # Use that model to predict E[Y] -> fit model with data and use it to predict given confounder
        res = model.fit()
        #matching = df[["c", "d"]]
        data_dict = {}
        for c in confounders:
            data_dict[c] = df[c]
        matching = pd.DataFrame(data_dict)
        # will give you only the two columns in the df which are c and d
        num_rows = df.shape[0]
        temp_col = [a_] * num_rows
        matching.insert(0, intervention, temp_col)
        predictions = res.predict(matching)
        # print(predictions)
        avg = np.mean(predictions)
        results.append(avg)
    return np.array(results)


def ipw(df, confounders=["c", "d"]):
    """
    An inverse probability weighting estimator for E[Y^a]
    You may want to use smf.mnlogit to train a propensity model p(A | confounders)

    Arguments
      df: a data frame for which to estimate the causal effect
      confounders: the variables to treat as confounders.
          For the data we consider, if you include both c and d as confounders,
          this estimator should be unbiased. If you only include one,
          you would expect to see more bias.

    Returns
        results: an array of E[Y^a] estimates

    """

    results = []
    a_dim = np.unique(df["a"])

    model = smf.mnlogit(
        "a ~ {}".format(" + ".join(confounders)),  data=df)
    result = model.fit(method='powell')
    for a_ in a_dim:
        subset = df[df['a'] == a_]  # this takes care of whether a matches?
        predictions = result.predict(subset)
        predictions = np.array(predictions)
        predictions = predictions[:, a_]
        # print("predictions_subset ", predictions)
        # print("subsety ", subset['y'])
        y = np.array(subset['y'])
        expected = np.divide(y, predictions)
        # print("expected", expected)
        N = df.shape[0]
        results.append(np.sum(expected)/N)
    #print('results ', results)
    return np.array(results)


def frontdoor(df):
    """
    A front-door estimator for E[Y^a]
    Should only use a, m, and y -- not c or d.
    You may want to use smf.ols to model E[Y | M, A]

    Arguments
      df: a data frame for which to estimate the causal effect

    Returns
        results: an array of E[Y^a] estimates

    """
    results = []
    raise NotImplementedError

    return np.array(results)
