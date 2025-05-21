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

Documentation
==============
The documentation for this package is hosted on `GitHub Pages <https://we3lab.github.io/electric-emission-cost>`_.

Legal Documents
===============

This work was funded by the National Alliance for Water Innovation (NAWI, grant number UBJQH - MSM), the Office of Energy Efficiency and Renewable Energy (EERE, grant number 0009499 - MSM), and the Office of Energy Efficiency and Renewable Energy, Advanced Manufacturing Office (grant number DE-EE0009499).
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.

- `LICENSE <https://github.com/we3lab/electric-emission-cost/blob/main/LICENSE/>`_
- `CONTRIBUTING <https://github.com/we3lab/electric-emission-cost/blob/main/CONTRIBUTING.rst/>`_