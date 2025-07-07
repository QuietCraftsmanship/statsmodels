"""
To build with coverage of Cython files
export SM_CYTHON_COVERAGE=1
python -m pip install -e .
pytest --cov=statsmodels statsmodels
coverage html
"""

from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution

from collections import defaultdict
import fnmatch
import inspect
import os
from os.path import dirname, join as pjoin, relpath
from pathlib import Path
import shutil
import sys

import numpy as np
from packaging.version import parse

try:
    # SM_FORCE_C is a testing shim to force setup to use C source files
    FORCE_C = int(os.environ.get("SM_FORCE_C", 0))
    if FORCE_C:
        raise ImportError("Force import error for testing")
    from Cython import Tempita, __version__ as cython_version
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

    HAS_CYTHON = True
    CYTHON_3 = parse(cython_version) >= parse("3.0")
except ImportError:
    from setuptools.command.build_ext import build_ext  # noqa: F401

    HAS_CYTHON = CYTHON_3 = False

SETUP_DIR = Path(__file__).parent.resolve()

###############################################################################
# Key Values that Change Each Release
###############################################################################
# These are strictly installation requirements. Builds requirements are
# managed in pyproject.toml
INSTALL_REQUIRES = []
with open("requirements.txt", encoding="utf-8") as req:
    for line in req.readlines():
        INSTALL_REQUIRES.append(line.split("#")[0].strip())

DEVELOP_REQUIRES = []
with open("requirements-dev.txt", encoding="utf-8") as req:
    for line in req.readlines():
        DEVELOP_REQUIRES.append(line.split("#")[0].strip())

CYTHON_MIN_VER = "3.0.10"  # released January 2023

EXTRAS_REQUIRE = {
    "build": ["cython>=" + CYTHON_MIN_VER],
    "develop": ["cython>=" + CYTHON_MIN_VER] + DEVELOP_REQUIRES,
    "docs": [
        "sphinx",
        "nbconvert",
        "jupyter_client",
        "ipykernel",
        "matplotlib",
        "nbformat",
        "numpydoc",
        "pandas-datareader",
    ],
}

###############################################################################
# Values that rarely change
###############################################################################
DISTNAME = "statsmodels"
DESCRIPTION = "Statistical computations and models for Python"
README = SETUP_DIR.joinpath("README.rst").read_text()
LONG_DESCRIPTION = README

MAINTAINER = 'Josef Perktold, Chad Fulton, Kerby Shedden'
MAINTAINER_EMAIL ='pystatsmodels@googlegroups.com'
URL = 'http://www.statsmodels.org/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

# These imports need to be here; setuptools needs to be imported first.
from distutils.extension import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    not numpy_incl in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          os.path.join(cwd, 'tools', 'cythonize.py'),
                          'statsmodels'],
                         cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def init_cython_exclusion(filename):
    with open(filename, 'w') as f:
        pass


def append_cython_exclusion(path, filename):
    with open(filename, 'a') as f:
        f.write(path + "\n")


def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)


def check_dependency_versions(min_versions):
    """
    Don't let pip/setuptools do this all by itself.  It's rude.

    For all dependencies, try to import them and check if the versions of
    installed dependencies match the minimum version requirements.  If
    installed but version too low, raise an error.  If not installed at all,
    return the correct ``setup_requires`` and ``install_requires`` arguments to
    be added to the setuptools kwargs.  This prevents upgrading installed
    dependencies like numpy (that should be an explicit choice by the user and
    never happen automatically), but make things work when installing into an
    empty virtualenv for example.

    """
    setup_requires = []
    install_requires = []

    try:
        from numpy.version import short_version as npversion
    except ImportError:
        setup_requires.append('numpy')
        install_requires.append('numpy')
    else:
        if not (LooseVersion(npversion) >= min_versions['numpy']):
            raise ImportError("Numpy version is %s. Requires >= %s" %
                              (npversion, min_versions['numpy']))

    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    else:
        try:
            from scipy.version import short_version as spversion
        except ImportError:
            from scipy.version import version as spversion  # scipy 0.7.0
        if not (LooseVersion(spversion) >= min_versions['scipy']):
            raise ImportError("Scipy version is %s. Requires >= %s" %
                              (spversion, min_versions['scipy']))

    try:
        from pandas import __version__ as pversion
    except ImportError:
        install_requires.append('pandas')
    else:
        if not (LooseVersion(pversion) >= min_versions['pandas']):
            ImportError("Pandas version is %s. Requires >= %s" %
                        (pversion, min_versions['pandas']))

    try:
        from patsy import __version__ as patsy_version
    except ImportError:
        install_requires.append('patsy')
    else:
        # patsy dev looks like 0.1.0+dev
        pversion = re.match("\d*\.\d*\.\d*", patsy_version).group()
        if not (LooseVersion(pversion) >= min_versions['patsy']):
            raise ImportError("Patsy version is %s. Requires >= %s" %
                              (pversion, min_versions["patsy"]))

    return setup_requires, install_requires


MAJ = 0
MIN = 9
REV = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJ,MIN,REV)

classifiers = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Operating System :: OS Independent',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: BSD License',
               'Topic :: Office/Business :: Financial',
               'Topic :: Scientific/Engineering']

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(" ".join(cmd), stdout = subprocess.PIPE, env=env,
                               shell=True).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def write_version_py(filename=pjoin(curdir, 'statsmodels/version.py')):
    cnt = "\n".join(["",
                    "# THIS FILE IS GENERATED FROM SETUP.PY",
                    "short_version = '%(version)s'",
                    "version = '%(version)s'",
                    "full_version = '%(full_version)s'",
                    "git_revision = '%(git_revision)s'",
                    "release = %(isrelease)s", "",
                    "if not release:",
                    "    version = full_version"])
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    dowrite = True
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        try:
            from statsmodels.version import git_revision as GIT_REVISION
        except ImportError:
            dowrite = False
            GIT_REVISION = "Unknown"
    else:
        GIT_REVISION = "Unknown"

MAINTAINER = "statsmodels Developers"
MAINTAINER_EMAIL = "pystatsmodels@googlegroups.com"
URL = "https://www.statsmodels.org/"
LICENSE = "BSD License"
DOWNLOAD_URL = ""
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/statsmodels/statsmodels/issues",
    "Documentation": "https://www.statsmodels.org/stable/index.html",
    "Source Code": "https://github.com/statsmodels/statsmodels",
}

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: OS Independent",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: BSD License",
    "Topic :: Office/Business :: Financial",
    "Topic :: Scientific/Engineering",
]

FILES_TO_INCLUDE_IN_PACKAGE = ["LICENSE.txt", "setup.cfg"]

FILES_COPIED_TO_PACKAGE = []
for filename in FILES_TO_INCLUDE_IN_PACKAGE:
    if os.path.exists(filename):
        dest = os.path.join("statsmodels", filename)
        shutil.copy2(filename, dest)
        FILES_COPIED_TO_PACKAGE.append(dest)

STATESPACE_RESULTS = "statsmodels.tsa.statespace.tests.results"

ADDITIONAL_PACKAGE_DATA = {
    "statsmodels": FILES_TO_INCLUDE_IN_PACKAGE,
    "statsmodels.datasets.tests": ["*.zip"],
    "statsmodels.iolib.tests.results": ["*.dta"],
    "statsmodels.stats.tests.results": ["*.json"],
    "statsmodels.tsa.stl.tests.results": ["*.csv"],
    "statsmodels.tsa.vector_ar.tests.results": ["*.npz", "*.dat"],
    "statsmodels.stats.tests": ["*.txt"],
    "statsmodels.stats.libqsturng": ["*.r", "*.txt", "*.dat"],
    "statsmodels.stats.libqsturng.tests": ["*.csv", "*.dat"],
    "statsmodels.sandbox.regression.tests": ["*.dta", "*.csv"],
    STATESPACE_RESULTS: ["*.pkl", "*.csv"],
    STATESPACE_RESULTS + ".frbny_nowcast": ["test*.mat"],
    STATESPACE_RESULTS + ".frbny_nowcast.Nowcasting.data.US": ["*.csv"],
}

##############################################################################
# Extension Building
##############################################################################
CYTHON_COVERAGE = os.environ.get("SM_CYTHON_COVERAGE", False)
CYTHON_COVERAGE = CYTHON_COVERAGE in ("1", "true", '"true"')
CYTHON_TRACE_NOGIL = str(int(CYTHON_COVERAGE))
if CYTHON_COVERAGE:
    print("Building with coverage for Cython code")
COMPILER_DIRECTIVES = {"linetrace": CYTHON_COVERAGE}
DEFINE_MACROS = [
    ("CYTHON_TRACE_NOGIL", CYTHON_TRACE_NOGIL),
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
]


exts = dict(
    _stl={"source": "statsmodels/tsa/stl/_stl.pyx"},
    _exponential_smoothers={
        "source": "statsmodels/tsa/holtwinters/_exponential_smoothers.pyx"
    },  # noqa: E501
    _ets_smooth={
        "source": "statsmodels/tsa/exponential_smoothing/_ets_smooth.pyx"
    },  # noqa: E501
    _innovations={"source": "statsmodels/tsa/_innovations.pyx"},
    _hamilton_filter={
        "source": "statsmodels/tsa/regime_switching/_hamilton_filter.pyx.in"
    },  # noqa: E501
    _kim_smoother={
        "source": "statsmodels/tsa/regime_switching/_kim_smoother.pyx.in"
    },  # noqa: E501
    _arma_innovations={
        "source": "statsmodels/tsa/innovations/_arma_innovations.pyx.in"
    },  # noqa: E501
    linbin={"source": "statsmodels/nonparametric/linbin.pyx"},
    _qn={"source": "statsmodels/robust/_qn.pyx"},
    _smoothers_lowess={
        "source": "statsmodels/nonparametric/_smoothers_lowess.pyx"
    },  # noqa: E501
)


statespace_exts = [
    "statsmodels/tsa/statespace/_initialization.pyx.in",
    "statsmodels/tsa/statespace/_representation.pyx.in",
    "statsmodels/tsa/statespace/_kalman_filter.pyx.in",
    "statsmodels/tsa/statespace/_filters/_conventional.pyx.in",
    "statsmodels/tsa/statespace/_filters/_inversions.pyx.in",
    "statsmodels/tsa/statespace/_filters/_univariate.pyx.in",
    "statsmodels/tsa/statespace/_filters/_univariate_diffuse.pyx.in",
    "statsmodels/tsa/statespace/_kalman_smoother.pyx.in",
    "statsmodels/tsa/statespace/_smoothers/_alternative.pyx.in",
    "statsmodels/tsa/statespace/_smoothers/_classical.pyx.in",
    "statsmodels/tsa/statespace/_smoothers/_conventional.pyx.in",
    "statsmodels/tsa/statespace/_smoothers/_univariate.pyx.in",
    "statsmodels/tsa/statespace/_smoothers/_univariate_diffuse.pyx.in",
    "statsmodels/tsa/statespace/_simulation_smoother.pyx.in",
    "statsmodels/tsa/statespace/_cfa_simulation_smoother.pyx.in",
    "statsmodels/tsa/statespace/_tools.pyx.in",
]


class CleanCommand(Command):
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        msg = """

python setup.py clean is not supported.

Use one of:

* `git clean -xdf` to clean all untracked files
* `git clean -Xdf` to clean untracked files ignored by .gitignore
"""
        print(msg)
        sys.exit(1)


cmdclass = {"clean": CleanCommand}


def check_source(source_name):
    """Chooses C or pyx source files, and raises if C is needed but missing"""
    source_ext = ".pyx"
    if not HAS_CYTHON:
        source_name = source_name.replace(".pyx.in", ".c")
        source_name = source_name.replace(".pyx", ".c")
        source_ext = ".c"
        if not os.path.exists(source_name):
            msg = (
                "C source not found.  You must have Cython installed to "
                "build if the C source files have not been generated."
            )
            raise OSError(msg)
    return source_name, source_ext


def process_tempita(source_name):
    """Runs pyx.in files through tempita is needed"""
    if source_name.endswith("pyx.in"):
        with open(source_name, encoding="utf-8") as templated:
            pyx_template = templated.read()
        pyx = Tempita.sub(pyx_template)
        pyx_filename = source_name[:-3]
        with open(pyx_filename, "w", encoding="utf-8") as pyx_file:
            pyx_file.write(pyx)
        file_stats = os.stat(source_name)
        try:
            os.utime(
                pyx_filename,
                ns=(file_stats.st_atime_ns, file_stats.st_mtime_ns),
            )
        except AttributeError:
            os.utime(pyx_filename, (file_stats.st_atime, file_stats.st_mtime))
        source_name = pyx_filename
    return source_name


NUMPY_INCLUDES = sorted(
    {np.get_include(), pjoin(dirname(inspect.getfile(np.core)), "include")}
)
NUMPY_MATH_LIBS = {
    "include_dirs": [np.get_include()],
    "library_dirs": [os.path.join(np.get_include(), "..", "lib")],
    "libraries": ["npymath"],
}


extensions = []
for config in exts.values():
    source, ext = check_source(config["source"])
    source = process_tempita(source)
    name = source.replace("/", ".").replace(ext, "")
    include_dirs = config.get("include_dirs", [])
    depends = config.get("depends", [])
    libraries = config.get("libraries", [])
    library_dirs = config.get("library_dirs", [])
    uses_numpy_libraries = config.get("numpy_libraries", False)

    include_dirs = sorted(set(include_dirs + NUMPY_MATH_LIBS["include_dirs"]))
    libraries = sorted(set(libraries + NUMPY_MATH_LIBS["libraries"]))
    library_dirs = sorted(set(library_dirs + NUMPY_MATH_LIBS["library_dirs"]))

    ext = Extension(
        name,
        [source],
        include_dirs=include_dirs,
        depends=depends,
        libraries=libraries,
        library_dirs=library_dirs,
        define_macros=DEFINE_MACROS,
    )
    extensions.append(ext)

for source in statespace_exts:
    source, ext = check_source(source)
    source = process_tempita(source)
    name = source.replace("/", ".").replace(ext, "")

    ext = Extension(
        name,
        [source],
        include_dirs=["statsmodels/src"] + NUMPY_MATH_LIBS["include_dirs"],
        depends=[],
        libraries=NUMPY_MATH_LIBS["libraries"],
        library_dirs=NUMPY_MATH_LIBS["library_dirs"],
        define_macros=DEFINE_MACROS,
    )
    extensions.append(ext)

COMPILER_DIRECTIVES["cpow"] = True
extensions = cythonize(
    extensions,
    compiler_directives=COMPILER_DIRECTIVES,
    language_level=3,
    force=CYTHON_COVERAGE,
)

##############################################################################
# Construct package data
##############################################################################
package_data = defaultdict(list)
filetypes = ["*.csv", "*.txt", "*.dta"]
for root, _, filenames in os.walk(
    pjoin(os.getcwd(), "statsmodels", "datasets")
):  # noqa: E501
    matches = []
    for filetype in filetypes:
        for filename in fnmatch.filter(filenames, filetype):
            matches.append(filename)
    if matches:
        package_data[".".join(relpath(root).split(os.path.sep))] = filetypes
for root, _, _ in os.walk(pjoin(os.getcwd(), "statsmodels")):
    if root.endswith("results"):
        package_data[".".join(relpath(root).split(os.path.sep))] = filetypes

for path, filetypes in ADDITIONAL_PACKAGE_DATA.items():
    package_data[path].extend(filetypes)

if os.path.exists("MANIFEST"):
    os.unlink("MANIFEST")


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(name=DISTNAME,
      version=versioneer.get_version(),
      maintainer=MAINTAINER,
      ext_modules=extensions,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      platforms='any',
      cmdclass=cmdclass,
      packages=find_packages(),
      package_data=package_data,
      distclass=BinaryDistribution,
      include_package_data=False,  # True will install all files in repo
      setup_requires=SETUP_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      zip_safe=False,
      data_files=[('', ['LICENSE.txt', 'setup.cfg'])]
      )

# Clean-up copied files
for copy in FILES_COPIED_TO_PACKAGE:
    os.unlink(copy)
