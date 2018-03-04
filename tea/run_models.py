from time import time

from sklearn.model_selection import GridSearchCV

from tea.features import *


def run_grid_search(X, y, pipeline, parameters, scoring='accuracy'):
    """

    :param X:
    :param y:
    :param pipeline:
    :param parameters:
    :param scoring:
    :return:
    """

    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline,
                               parameters,
                               n_jobs=-1,
                               verbose=10,
                               refit=True,
                               return_train_score=True,
                               scoring=scoring)

    logger.info("Performing grid search...")
    logger.info("Pipeline: {}".format([name for name, _ in pipeline.steps]))
    logger.info("Parameters:")
    logger.info(parameters)

    t0 = time()
    grid_search.fit(X=X, y=y)

    logger.info("Completed in %0.3fs" % (time() - t0))
    logger.info("Best score: %0.3f" % grid_search.best_score_)
    logger.info("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))
