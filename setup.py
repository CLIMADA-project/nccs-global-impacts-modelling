from setuptools import setup, find_packages

setup(
    name="nccs",
    version="0.1",
    description="NCCS Global impacts of climate change on Switzerland Modelling Pipeline",
    url="https://github.com/CLIMADA-project/nccs-global-impacts-modelling",
    author="NCCS",
    author_email="sjuhel@ethz.ch",
    license="OSI Approved :: GNU Lesser General Public License v3 (GPLv3)",
    python_requires=">=3.9,<3.12",
    install_requires=[
        "bokeh",
        "boto3",
        "python-dotenv",
        "climada",
        "climada_petals",
    ],
    packages=find_packages(),
    include_package_data=True,
)
