"""Package setup file."""
import os
from setuptools import setup, find_packages


def get_version(setup_fpath):
    """Retrieves the version number."""

    os.chdir(os.path.join(setup_fpath, "uncertaintyAwareDeepLearn"))
    with open("__init__.py", "r") as fhandle:
        version_line = [l for l in fhandle.readlines() if
                    l.startswith("__version__")]
        version = version_line[0].split("=")[1].strip().replace('"', "")
    os.chdir(setup_fpath)
    return version




def main():
    """Builds the package, including all currently used extensions."""
    setup_fpath = os.path.dirname(os.path.abspath(__file__))
    read_me = os.path.join(setup_fpath, "README.md")
    with open(read_me, "r") as fhandle:
        long_description = "".join(fhandle.readlines())

    setup(
        name="uncertaintyAwareDeepLearn",
        version=get_version(setup_fpath),
        packages=find_packages(),
        #cmdclass = {"build_ext": build_ext},
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        description="Plug-in uncertainty quantitation for neural networks",
        long_description = long_description,
        long_description_content_type="text/markdown",
        #ext_modules = ext_modules,
        #package_data={"": ["*.h", "*.c", "*.cu", "*.cpp",
        #                    "*.pyx", "*.sh"]}
        )



if __name__ == "__main__":
    main()
