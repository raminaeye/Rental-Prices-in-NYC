"""Feature engineering helpers for the random forest training step.

Currently used (and tested) by ``src/train_random_forest/run.py``. The
duplicated ``delta_date_feature`` definition in ``run.py`` keeps that script
self-contained when run inside an MLflow-managed environment.
"""
import pandas as pd


def delta_date_feature(dates):
    """Compute days since the most recent date, column-wise.

    :param dates: 2D array-like of dates in any format recognized by
        ``pd.to_datetime``
    :return: numpy array with the delta in days between each date and the
        most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()
