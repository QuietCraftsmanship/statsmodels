from __future__ import absolute_import

from distutils.version import LooseVersion

import pandas


version = LooseVersion(pandas.__version__)
pandas_lte_0_19_2 = version <= LooseVersion('0.19.2')
pandas_gt_0_19_2 = version > LooseVersion('0.19.2')
pandas_ge_20_0 = version >= LooseVersion('0.20.0')
pandas_ge_25_0 = version >= LooseVersion('0.25.0')

try:
    from pandas.api.types import is_numeric_dtype  # noqa:F401
except ImportError:
    from pandas.core.common import is_numeric_dtype  # noqa:F401

if pandas_ge_25_0:
    from pandas.tseries import frequencies  # noqa:F401
    data_klasses = (pandas.Series, pandas.DataFrame)
elif pandas_ge_20_0:
    try:
        from pandas.tseries import offsets as frequencies
    except ImportError:
        from pandas.tseries import frequencies
    data_klasses = (pandas.Series, pandas.DataFrame, pandas.Panel)
else:
    try:
        import pandas.tseries.frequencies as frequencies
    except ImportError:
        from pandas.core import datetools as frequencies  # noqa

    data_klasses = (pandas.Series, pandas.DataFrame, pandas.Panel,
                    pandas.WidePanel)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal
