import numpy as np
import pandas as pd


class OLS:

    def __init__(self, data: pd.DataFrame):
        if "y" not in data.columns:
            raise ValueError("Must supply a column y with the target for the regression.")

        y = np.array(data["y"])
        feats = data.loc[:, data.columns != "y"]

        self.columns = feats.columns
        self.n = feats.shape[0]
        self.p = feats.shape[1]
        self._ext_mat = OLS._make_ext_mat(np.array(feats), y)
        self.tss = ((y - y.mean()) ** 2).sum()

        self.rss = None
        self.parameters = None
        self.parameters_se = None
        self._is_fit = False

    @staticmethod
    def _partial_inverse(mat: np.ndarray, k: int) -> np.ndarray:
        """
        Performs a partial inverse operation (also known as a SWEEP operation) on a
        matrix with the 'pivot block' being the diagonal element of the matrix at (k,k).

        https://en.wikipedia.org/wiki/Partial_inverse_of_a_matrix

        This is the same as adding the kth column to the regression, and acts as it's own
        inverse operation as well. So this can be used to efficiently add and remove
        variables from a regression by operating on the extended design matrix in place.

        TODO: Add stepwise regression functionality using this ability to add-remove variables

        :param mat: the extended design matrix to operate on
        :param k: the column to add-remove to the regression
        :return:
        """
        h = mat - np.outer(mat[:, k], mat[k, :]) / mat[k, k]
        h[:, k] = mat[:, k] / mat[k, k]
        h[k, :] = h[:, k]
        h[k, k] = -1 / mat[k, k]
        return h

    @staticmethod
    def _make_ext_mat(feats: np.ndarray, y: np.ndarray):
        """
        Builds a matrix that can be operated on by
        OLS._partial_inverse to add and remove variables from the regression.

         XtX | YtX                                           (XtX)-1 | b^
         ---------    --(partial_inverse operations)-->      ------------
         XtY | YtY                                             b^t   | RSS

         The coefficients of the regression and the RSS are directly available in the
         matrix, and the diagonal of (XtX)-1 gets filled with the se^2 of the parameters
         so these can be used to get the standard errors of the parameters and the t values.

        :param feats:
        :param y:
        :return:
        """
        xtx = feats.T.dot(feats)
        yty = y.dot(y)
        xty = feats.T.dot(y.T)

        return np.vstack(
            [
                np.append(xtx, np.array([xty]).T, axis=1),
                np.append(xty, yty),
            ]
        )

    def parameter_fits(self):
        if not self._is_fit:
            raise Exception("Must call OLS.fit() before accessing parameter fits")
        ses = []
        for i, feat in enumerate(self.columns):
            ses.append(
                dict(
                    feature=feat,
                    value=self.parameters[i],
                    standard_error=self.parameters_se[i],
                    t=self.parameters[i]/self.parameters_se[i],
                )
            )
        return ses

    def r_squared(self) -> float:
        if not self._is_fit:
            raise Exception("Must call OLS.fit() before accessing R^2")
        return 1 - self.rss/self.tss

    def fit(self):
        for i in range(self.p):
            self._ext_mat = OLS._partial_inverse(self._ext_mat, i)

        self._is_fit = True
        self.parameters = list(self._ext_mat[-1:, :-1][0])
        self.rss = self._ext_mat[-1, -1]
        self.parameters_se = list(np.sqrt(np.abs(np.diagonal(self._ext_mat[:-1, : -1])*(self.rss/(self.n - self.p - 1)))))
