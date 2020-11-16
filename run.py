import pandas as pd
import numpy as np
from ols import OLS
import statsmodels.api as sm
import timeit

SAMPLE_SIZE = 100000
VARIANCE = 30

data = pd.DataFrame(
    np.array(
        [
            np.random.normal(0, 5, size=SAMPLE_SIZE),
            np.random.normal(0, 30, size=SAMPLE_SIZE),
            np.random.normal(0, 50, size=SAMPLE_SIZE),
            np.random.normal(0, 75, size=SAMPLE_SIZE),
            np.random.normal(0, 75, size=SAMPLE_SIZE),
        ]
    ).T,
    columns=["a", "b", "c", "d", "e"]
)

data["y"] = (
        np.random.normal(1, 10, SAMPLE_SIZE) * data["a"] +  # inject some noise on the variable's interactions
                                                            # with the response coefficient should be close to 1
                                                            # but with higher standard error than the others
        4 * data["b"] +  # coefficient should be close to 4
        2 * data["c"] +  # coefficient should be close to 2
        3 * data["d"] +  # coefficient should be close to 3
        # Don't include variable "e" -- should see a 0 for the coefficient
        np.random.normal(0, VARIANCE, SAMPLE_SIZE)
)

if __name__ == "__main__":
    ols = OLS(data)
    ols.fit()
    print(f"\n\nAd-Hoc Summary Stats from this package\n")
    print(f"R-Squared: {ols.r_squared():0.4f}")
    print(f"RSS: {ols.rss}")
    print("Parameter Estimates")
    for i in ols.parameter_fits():
        print(i)
    print("\n\n")

    for _ in range(3):
        print("-" * 50)

    print("\n\nStatsmodels Summary for QA")
    results = sm.OLS(data["y"], data[["a", "b", "c", "d", "e"]]).fit()
    print(results.summary())
    print("\n\n")

    for _ in range(3):
        print("-" * 50)

    print("\n")
    print(f"timeit results for this package: "
          f"{timeit.timeit(OLS(data).fit, number=100)} seconds")
    print(f"timeit results for Statsmodels package: "
          f"{timeit.timeit(sm.OLS(data['y'], data[['a', 'b', 'c', 'd', 'e']]).fit, number=100)} seconds")
