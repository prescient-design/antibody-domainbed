from setuptools import setup, find_packages

setup(
    name='antibody_domainbed',
    version='v0.10',
    packages=find_packages(),
    description='Antibody DomainBed',
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose'],
)
