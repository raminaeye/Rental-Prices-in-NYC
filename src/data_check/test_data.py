"""Data quality tests for the cleaned rental listings dataset.

These tests are run by pytest through the ``data_check`` MLflow step (see the
``MLproject`` file). The ``data``, ``ref_data``, ``kl_threshold``,
``min_price`` and ``max_price`` fixtures are provided by ``conftest.py``.
"""
import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """Check that the dataset has exactly the expected columns, in order."""
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_columns) == list(these_columns)


def test_neighborhood_names(data):
    """Check that the borough names are exactly the five NYC boroughs."""
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """Check that all listings fall within the NYC longitude/latitude bounds."""
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """Check that the borough distribution has not drifted vs. the reference data.

    The KL divergence between the new data and the reference dataset must stay
    below ``kl_threshold``.
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data: pd.DataFrame):
    """Check that the dataset size is reasonable (not too small, not too large)."""
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """Check that every price falls within the accepted [min_price, max_price] range."""
    assert np.sum(~data['price'].between(min_price, max_price)) == 0
