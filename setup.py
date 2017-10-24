#!/usr/bin/python

from __future__ import absolute_import

from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst', 'md')
except:
    print(u"Warning: pypandoc module not found, could not convert Markdown to "
          "RST.")
    read_md = lambda f: open(f, 'r').read()


setup(name=u"addm_toolbox",
      version=u"0.1.9",
      description=u"A toolbox for data analysis using the attentional "
      "drift-diffusion model.",
      long_description=read_md(u"README.md"),
      classifiers=[
          u"Programming Language :: Python :: 2.7",
          u"Programming Language :: Python :: 3.6",
          u"Development Status :: 3 - Alpha",
          u"Topic :: Scientific/Engineering",
          u"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      url=u"http://github.com/goptavares/aDDM-Toolbox",
      download_url = u"https://github.com/goptavares/aDDM-Toolbox/archive/" \
                     "0.1.9.tar.gz",
      author=u"Gabriela Tavares",
      author_email=u"gtavares@caltech.edu",
      license=u"GPLv3",
      packages=[u"addm_toolbox"],
      entry_points = {
        u"console_scripts": [
            u"addm_util_test = util_test:main",
            u"addm_run_tests = run_all_tests:main",
            u"addm_demo = demo:main",
            u"ddm_mla_test = ddm_mla_test:main",
            u"addm_mla_test = addm_mla_test:main",
            u"ddm_pta_test = ddm_pta_test:main",
            u"addm_pta_test = addm_pta_test:main",
            u"addm_pta_mle = addm_pta_mle:main",
            u"addm_pta_map = addm_pta_map:main",
            u"addm_simulate_true_distributions = " \
            "simulate_addm_true_distributions:main",
            u"addm_basinhopping = basinhopping_optimize:main",
            u"addm_genetic_algorithm = genetic_algorithm_optimize:main",
            u"addm_cis_trans_fit = cis_trans_fitting:main",  
        ],
      },
      test_suite=u"nose.collector",
      tests_require=[u"nose"],
      install_requires=[
          u"deap",
          u"matplotlib",
          u"numpy",
          u"pandas",
          u"scipy",
      ],
      include_package_data=True,
      zip_safe=False)
