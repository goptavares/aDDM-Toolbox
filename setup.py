#!/usr/bin/python

from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst', 'md')
except:
    print("Warning: pypandoc module not found, could not convert Markdown to "
          "RST.")
    read_md = lambda f: open(f, 'r').read()


setup(name="addm_toolbox",
      version="0.1.7",
      description="A toolbox for data analysis using the attentional "
      "drift-diffusion model.",
      long_description=read_md("README.md"),
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Development Status :: 3 - Alpha",
          "Topic :: Scientific/Engineering",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      url="http://github.com/goptavares/aDDM-Toolbox",
      download_url = "https://github.com/goptavares/aDDM-Toolbox/archive/" \
                     "0.1.7.tar.gz",
      author="Gabriela Tavares",
      author_email="gtavares@caltech.edu",
      license="GPLv3",
      packages=["addm_toolbox"],
      entry_points = {
        "console_scripts": [
            "addm_demo = addm_toolbox.demo:main",
            "addm_util_test = addm_toolbox.util_test:main",
            "ddm_pta_test = addm_toolbox.ddm_pta_test:main",
            "addm_pta_test = addm_toolbox.addm_pta_test:main",
            "addm_run_tests = addm_toolbox.run_all_tests:main",
            "addm_pta_mle = addm_toolbox.addm_pta_mle:main",
            "addm_pta_map = addm_toolbox.addm_pta_map:main",
            "addm_simulate_true_distributions = " \
            "addm_toolbox.simulate_addm_true_distributions:main",
            "addm_basinhopping = addm_toolbox.basinhopping_optimize:main",
            "addm_genetic_algorithm = " \
            "addm_toolbox.genetic_algorithm_optimize:main",
            "addm_cis_trans_fit = addm_toolbox.cis_trans_fitting:main",
            "ddm_mla = addm_toolbox.ddm_mla:main",
            "addm_mla = addm_toolbox.addm_mla:main",
        ],
      },
      test_suite="nose.collector",
      tests_require=["nose"],
      install_requires=[
          "deap",
          "matplotlib",
          "numpy",
          "pandas",
          "scipy",
      ],
      include_package_data=True,
      zip_safe=False)
