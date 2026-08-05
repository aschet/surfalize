import warnings

import numpy as np
import pytest
import scipy.ndimage as ndimage

from surfalize import Surface
from surfalize.exceptions import CalculationError


@pytest.fixture
def grating():
    # A 1D grating (e.g. a 2-beam DLIP structure): stays correlated along its lamellae across the whole field, so Str
    # is undefined while Sal (the across-lamellae decay) is valid.
    x, _ = np.meshgrid(np.arange(200), np.arange(200))
    return Surface(np.sin(2 * np.pi * x / 8.0), 1.0, 1.0)


def test_str_within_unit_interval_for_uncorrelated_surface():
    # White noise decorrelates within ~1 pixel, so the region above the threshold is tiny and dominated by pixel
    # discretisation. This is the case that previously allowed the shortest decay length to exceed the longest and
    # produce Str > 1. Str must always stay within (0, 1].
    rng = np.random.RandomState(42)
    surface = Surface(rng.normal(size=(128, 128)), 1.0, 1.0)
    sal = surface.Sal()
    str_ = surface.Str()
    assert np.isfinite(sal) and sal > 0
    assert 0 < str_ <= 1


def test_str_matches_anisotropy_of_correlated_field():
    # A Gaussian-correlated field with different correlation lengths in x and y decays below the threshold in both
    # directions within the field, so Str is well-defined and approximates the ratio of the correlation lengths.
    rng = np.random.RandomState(7)
    data = ndimage.gaussian_filter(rng.normal(size=(300, 300)), sigma=(3.0, 9.0))  # sigma order is (y, x)
    surface = Surface(data, 1.0, 1.0)
    sal = surface.Sal()
    str_ = surface.Str()
    assert np.isfinite(sal) and sal > 0
    assert 0 < str_ <= 1
    assert str_ == pytest.approx(3.0 / 9.0, abs=0.1)


def test_str_undefined_but_sal_defined_for_1d_grating(grating):
    # A 1D grating (e.g. a 2-beam DLIP structure) stays correlated along its lamellae across the whole field, so the
    # slowest decay -- and therefore Str -- cannot be determined within the evaluation area. Sal, the across-lamellae
    # (fastest) decay, remains valid and must not warn.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sal = grating.Sal()
    assert not any("Sal is undefined" in str(w.message) for w in record)
    assert np.isfinite(sal) and sal > 0
    with pytest.warns(RuntimeWarning, match="Str is undefined"):
        assert np.isnan(grating.Str())


def test_str_on_undefined_bound_returns_upper_bound(grating):
    # 'bound' returns the shortest length over the measured longest length as an upper bound on Str (still within
    # (0, 1]) together with a warning, instead of NaN.
    with pytest.warns(RuntimeWarning, match="Str is undefined"):
        bound = grating.Str(on_undefined='bound')
    assert np.isfinite(bound)
    assert 0 < bound <= 1


def test_str_on_undefined_exception_raises(grating):
    with pytest.raises(CalculationError):
        grating.Str(on_undefined='exception')


def test_sal_on_undefined_exception_for_flat_field():
    surface = Surface(np.full((64, 64), 2.0), 1.0, 1.0)
    with pytest.raises(CalculationError):
        surface.Sal(on_undefined='exception')
    with pytest.raises(CalculationError):
        surface.Str(on_undefined='exception')


def test_on_undefined_ignored_when_defined():
    # When Str is well-defined, on_undefined has no effect and never warns or raises.
    rng = np.random.RandomState(7)
    data = ndimage.gaussian_filter(rng.normal(size=(300, 300)), sigma=(3.0, 9.0))
    surface = Surface(data, 1.0, 1.0)
    reference = surface.Str()
    for mode in ('nan', 'bound', 'exception'):
        assert surface.Str(on_undefined=mode) == reference


def test_invalid_on_undefined_raises_valueerror(grating):
    with pytest.raises(ValueError):
        grating.Str(on_undefined='nonsense')
    with pytest.raises(ValueError):
        grating.Sal(on_undefined='nonsense')


def test_sal_and_str_undefined_for_flat_field():
    # A perfectly flat field has no texture: the autocorrelation never decays below the threshold anywhere, so both
    # Sal and Str are undefined. This must be reported as NaN with a warning rather than raising.
    surface = Surface(np.full((64, 64), 2.0), 1.0, 1.0)
    with pytest.warns(RuntimeWarning, match="Sal is undefined"):
        assert np.isnan(surface.Sal())
    with pytest.warns(RuntimeWarning, match="Str is undefined"):
        assert np.isnan(surface.Str())
