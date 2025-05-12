******************************
Electric Emission & Cost (EEC)
******************************

.. image::
   https://github.com/we3lab/electric-emission-cost/workflows/Build%20Main/badge.svg
   :height: 30
   :target: https://github.com/we3lab/electric-emission-cost/actions
   :alt: Build Status

.. image::
   https://github.com/we3lab/electric-emission-cost/workflows/Documentation/badge.svg
   :height: 30
   :target: https://we3lab.github.io/electric-emission-cost
   :alt: Documentation

.. image::
   https://codecov.io/gh/we3lab/electric-emission-cost/branch/main/graph/badge.svg
   :height: 30
   :target: https://codecov.io/gh/we3lab/electric-emission-cost
   :alt: Code Coverage

A package for calculating electricity-related emissions and costs for optimization problem formulation and other computational analyses.

Useful Commands
===============

1. ``pip install -e .``

  This will install your package in editable mode.

2. ``pytest electric_emission_cost/tests --cov=electric_emission_cost --cov-report=html``

  Produces an HTML test coverage report for the entire project which can
  be found at ``htmlcov/index.html``.

3. ``docs/make html``

  This will generate an HTML version of the documentation which can be found
  at ``_build/html/index.html``.

4. ``flake8 electric_emission_cost --count --verbose --show-source --statistics``

  This will lint the code and share all the style errors it finds.

5. ``black electric_emission_cost``

  This will reformat the code according to strict style guidelines.

Legal Documents
===============

This work was funded XX.

- `LICENSE <https://github.com/we3lab/electric-emission-cost/blob/main/LICENSE/>`_
- `CONTRIBUTING <https://github.com/we3lab/electric-emission-cost/blob/main/CONTRIBUTING.rst/>`_