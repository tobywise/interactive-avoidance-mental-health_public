from setuptools import setup, find_packages

setup(
    name="interactive_avoidance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pingouin",
        "seaborn",
        "statsmodels",
        "scipy",
        "requests",
        "tqdm",
        "networkx",
        "factor_analyzer",
        "stats_utils @ git+https://github.com/the-wise-lab/stats-utils.git",
        "maMDP @ git+https://github.com/tobywise/multi_agent_mdp.git",
    ],
    url="https://github.com/tobywise/interactive_avoidance_mental_health",
    license="MIT",
)
