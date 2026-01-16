from setuptools import find_packages, setup

setup(
    name='regdiff',
    packages=find_packages(include=['regdiff','regdiff.*']),
    version='3.5.0',
    description='TV-regularized differentiation of numerical time-series data',
    author='Luis Viornery',
    author_email='lviornery@cmu.edu',
    url='https://github.com/lviornery/RegDiff',
    license='BSD',
    python_requires='>3.5.2',
    install_requires=['numpy>=2.2','scipy>=1.11'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)