from sklearn.pipeline import Pipeline


# Custom pipeline to expose estimator feature importances to perform feature selection with scaling
class RfecvPipeline(Pipeline):

    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_
