"""pytest fixtures.

Module containing pytest fixtures.
"""
from typing import Tuple

import numpy as np
import pytest
from flask.testing import FlaskClient
from decision_mining.app import app


@pytest.fixture
def client() -> FlaskClient:
    """Testing client for decision_mining backend. A new one is generated for each test.

    Yields:
        FlaskClient[Response]: Flask test_client
    """
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


@pytest.fixture
def small_set() -> Tuple[np.ndarray, np.ndarray]:
    """Small generated test set. Only categorical.

    Yields:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    data = np.array([
        [3, 0, 1],
        [3, 1, 1],
        [0, 1, 0],
        [1, 2, 1],
        [1, 2, 0]], dtype=np.float64)

    X = data[:, :-1]
    y = data[:, -1]
    yield X, y


@pytest.fixture
def med_set() -> Tuple[np.ndarray, np.ndarray]:
    """Medium generated test set. Only categorical.

    Yields:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    X = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
    Xn = np.zeros(X.shape, dtype=object)
    Xn[X == 1] = "Ja"
    Xn[X == 0] = "Nee"
    y = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0])

    yield Xn, y


@pytest.fixture
def lin_small() -> Tuple[np.ndarray, np.ndarray]:
    """Small generated linearly separable test set. Only categorical.

    Yields:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    data = np.array([
        [3, 0, 1],
        [3, 1, 1],
        [0, 1, 0],
        [1, 2, 1],
        [1, 2, 1]], dtype=np.float64)

    X = data[:, :-1]
    y = data[:, -1]
    yield X, y


@pytest.fixture
def mixed_small() -> Tuple[np.ndarray, np.ndarray]:
    """Small generated separable test set. categorical and Continuous.

    Yields:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    data = np.array([
        [3, "True", 1],
        [3, "True", 1],
        [0, "False", 0],
        [1, "True", 1],
        [1, "False", 0]], dtype=object)

    X = data[:, :-1]
    y = data[:, -1]

    yield X, y


@pytest.fixture
def lin_med() -> Tuple[np.ndarray, np.ndarray]:
    """Linearly separable generated test set, only categorical.

    Yields:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    X = np.array([1, 1, 1, 0, 1, 2, 3, 4])
    y = np.array([0, 0, 0, 1, 0, 3, 4, 5])

    yield X, y


@pytest.fixture
def large_set() -> Tuple[np.ndarray, np.ndarray]:
    """Large data set for testing purposes.

    Yields:
        Iterator[Tuple[np.ndarray, np.ndarray]]: X, y
    """
    X = np.array([[15.326346], [12.861116], [3.818399], [9.966465],
                  [12.485344], [7.119658], [11.062046], [10.434194],
                  [10.017166], [9.301599], [11.732105], [14.812149],
                  [6.139737], [14.113096], [10.914521], [11.780550],
                  [5.453591], [10.540548], [15.938148], [5.680780],
                  [2.089087], [3.026511], [11.064281], [19.539869],
                  [14.494765], [16.690489], [10.396597], [15.591986],
                  [8.915008], [12.452817], [8.930731], [7.802764],
                  [10.530833], [8.095432], [15.233892], [10.780053],
                  [11.600840], [8.649471], [15.025889], [7.072122],
                  [12.640926], [8.596512], [6.242267], [8.042651],
                  [6.781635], [9.149209], [8.643439], [11.248680],
                  [12.260611], [9.410319], [24.896379], [26.156377],
                  [22.840484], [27.832640], [28.368899], [25.814323],
                  [34.578815], [28.669836], [24.550910], [23.551278],
                  [24.071271], [22.993084], [29.515141], [22.208760],
                  [24.675511], [22.882816], [29.184731], [19.325776],
                  [23.550003], [24.512377], [26.277426], [26.843612],
                  [24.136840], [28.956290], [26.259015], [34.870604],
                  [18.966714], [27.482403], [20.819470], [21.807965],
                  [32.940338], [31.979257], [17.575258], [24.108905],
                  [24.736609], [16.473152], [24.804678], [26.573365],
                  [25.869061], [17.022425], [29.430833], [25.978176],
                  [24.752352], [21.984428], [27.847836], [28.673077],
                  [23.071627], [25.358350], [28.307994], [17.181952],
                  [35.469903], [27.370172], [31.308363], [36.879006],
                  [34.422533], [33.399447], [33.816065], [38.392834],
                  [37.827322], [31.850924], [36.171763], [33.116771],
                  [44.617302], [32.042573], [33.748685], [33.604472],
                  [33.243895], [35.564417], [36.092197], [28.525717],
                  [32.707547], [29.718210], [39.944821], [44.861300],
                  [40.532929], [36.384932], [39.090064], [35.667241],
                  [41.626866], [37.671558], [34.080213], [30.481795],
                  [32.440150], [36.255322], [30.096656], [34.112827],
                  [40.359705], [35.117239], [42.941543], [40.788662],
                  [33.849482], [29.562758], [34.807835], [33.076851],
                  [36.511012], [41.457632], [30.507584], [33.445088],
                  [36.329400], [39.539893], [47.042858], [46.657191],
                  [50.378198], [46.974061], [44.051983], [45.229141],
                  [42.171714], [47.186659], [48.770002], [33.081613],
                  [49.872595], [44.773917], [46.843554], [47.649496],
                  [35.819587], [40.216283], [43.667595], [41.834437],
                  [46.096691], [42.940360], [38.155715], [47.449189],
                  [49.400517], [47.257410], [42.148802], [43.956562],
                  [47.193712], [47.412796], [49.027445], [43.822296],
                  [39.316478], [42.284213], [47.135539], [47.975898],
                  [53.900199], [45.468726], [45.978458], [44.290805],
                  [43.377082], [48.127101], [46.413910], [44.170882],
                  [40.681210], [44.507721], [43.436071], [50.020695],
                  [48.788504], [40.910757], [49.668673], [42.712093]])

    y = np.array([*["cold"] * 50,
                  *["warm"] * 50,
                  *["hot"] * 50,
                  *["hothot"] * 50])

    yield X, y
