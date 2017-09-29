from setuptools import setup

def readme():
    with open("README.md") as f:
        return f.read()

setup(name="addm_toolbox",
      version="0.1",
      description="A toolbox for data analysis using the attentional "
      "drift-diffusion model.",
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      url="http://github.com/goptavares/aDDM-Toolbox",
      download_url = "https://github.com/goptavares/aDDM-Toolbox/archive/0.1.tar.gz",
      author="Gabriela Tavares",
      author_email="gtavares@caltech.edu",
      license="GPLv3",
      packages=["addm_toolbox"],
      entry_points = {
        "console_scripts": [
            "addm_demo = addm_toolbox.demo:main",
            "ddm_pta_test = addm_toolbox.ddm_pta_test:main",
            "addm_pta_test = addm_toolbox.addm_pta_test:main",
            "addm_run_tests = addm_toolbox.run_all_tests:main",
            "addm_pta_mle = addm_toolbox.addm_pta_mle:main",
            "addm_pta_map = addm_toolbox.addm_pta_map:main",
            "addm_simulate_true_distributions = addm_toolbox.simulate_addm_true_distributions:main",
            "addm_basinhopping = addm_toolbox.basinhopping_optimize:main",
            "addm_genetic_algorithm = addm_toolbox.genetic_algorithm_optimize:main",
        ],
      },
      install_requires=[
          "deap",
          "matplotlib",
          "numpy",
          "pandas",
          "scipy",
      ],
      zip_safe=False)
