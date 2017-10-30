#!/usr/bin/env python

from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst', 'md')
except:
    print("Warning: pypandoc module not found, could not convert Markdown to "
          "RST.")
    read_md = lambda f: open(f, 'r').read()


setup(name="addm_toolbox",
      version="0.1.11",
      description="A toolbox for data analysis using the attentional "
      "drift-diffusion model.",
      long_description=read_md("README.md"),
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.6",
          "Development Status :: 3 - Alpha",
          "Topic :: Scientific/Engineering",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      url="http://github.com/goptavares/aDDM-Toolbox",
      download_url = "https://github.com/goptavares/aDDM-Toolbox/archive/" \
                     "0.1.11.tar.gz",
      author="Gabriela Tavares",
      author_email="gtavares@caltech.edu",
      license="GPLv3",
      packages=["addm_toolbox"],
      scripts=[
          "bin/addm_toolbox_tests",
          "bin/addm_demo",
          "bin/ddm_pta_test",
          "bin/addm_pta_test",
          "bin/addm_pta_mle",
          "bin/addm_pta_map",
          "bin/ddm_mla_test",
          "bin/addm_mla_test",
          "bin/addm_basinhopping",
          "bin/addm_genetic_algorithm",
          "bin/addm_simulate_true_distributions",
          "bin/addm_cis_trans_fit",
      ],
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
