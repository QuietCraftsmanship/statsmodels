import pytest
import subprocess

from statsmodels.compat.platform import PLATFORM_WIN


def test_lazy_imports():
    # Check that when statsmodels.api is imported, matplotlib is _not_ imported
    cmd = ("import statsmodels.api as sm; "
           "import sys; "
           "mods = [x for x in sys.modules if 'matplotlib.pyplot' in x]; "
           "assert not mods, mods")

    # TODO: is there a cleaner way to do this import in an isolated environment
    pyexe = 'python3' if not PLATFORM_WIN else 'python'
    p = subprocess.Popen(pyexe + ' -c "' + cmd + '"',
                         shell=True, close_fds=True)
    p.wait()
    rc = p.returncode
    assert rc == 0


@pytest.mark.xfail(reason="scipy.stats raises at import-time with -OO")
def test_docstring_optimization_compat():
    # GH#5235 check that importing with stripped docstrings doesn't raise
    p = subprocess.Popen('python -OO -c "import statsmodels.api as sm"',
                         shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.communicate()
    rc = p.returncode
    assert rc == 0, out
