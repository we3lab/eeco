******************************************
Electric Emissions & Cost Optimizer (EECO)
******************************************

.. image::
   https://github.com/we3lab/eeco/workflows/Build%20Main/badge.svg
   :height: 30
   :target: https://github.com/we3lab/eeco/actions
   :alt: Build Status

.. image::
   https://github.com/we3lab/eeco/workflows/Documentation/badge.svg
   :height: 30
   :target: https://we3lab.github.io/eeco
   :alt: Documentation

.. image::
   https://codecov.io/gh/we3lab/eeco/branch/main/graph/badge.svg
   :height: 30
   :target: https://codecov.io/gh/we3lab/eeco
   :alt: Code Coverage

.. image:: 
  https://zenodo.org/badge/979642377.svg
  :height: 30
  :target: https://doi.org/10.5281/zenodo.17102024
  :alt: Zenodo DOI

A package for calculating electricity-related emissions and costs for optimization problem formulation and other computational analyses.

Useful Commands
===============

1. ``pip install -e .``

  This will install your package in editable mode.

2. ``pytest eeco/tests --cov=eeco --cov-report=html``

  Produces an HTML test coverage report for the entire project which can
  be found at ``htmlcov/index.html``.

3. ``docs/make html``

  This will generate an HTML version of the documentation which can be found
  at ``_build/html/index.html``.

4. ``flake8 eeco --count --verbose --show-source --statistics``

  This will lint the code and share all the style errors it finds.

5. ``black eeco``

  This will reformat the code according to strict style guidelines.

Documentation
==============

The documentation for this package is hosted on `GitHub Pages <https://we3lab.github.io/eeco>`_.

Legal Documents
===============

This work was supported by the following grants and programs:

- `National Alliance for Water Innovation (NAWI) <https://www.nawihub.org/>`_ (grant number UBJQH - MSM)
- `Department of Energy, the Office of Energy Efficiency and Renewable Energy, Advanced Manufacturing Office <https://www.energy.gov/eere/ammto/advanced-materials-and-manufacturing-technologies-office>`_ (grant number DE-EE0009499)
- `California Energy Commission (CEC) <https://www.energy.ca.gov/>`_ (grant number GFO-23-316)
- `Equitable, Affordable & Resilient Nationwide Energy System Transition (EARNEST) Consortium <https://earnest.stanford.edu/>`_
- `Stanford University Bits & Watts Initiative <https://bitsandwatts.stanford.edu/>`_
- `Stanford Woods Institute Realizing Environmental Innovation Program (REIP) <https://woods.stanford.edu/research/funding-opportunities/realizing-environmental-innovation-program>`_
- `Stanford Woods Institute Mentoring Undergraduate in Interdisciplinary Research (MUIR) Program <https://woods.stanford.edu/educating-leaders/education-leadership-programs/mentoring-undergraduates-interdisciplinary-research>`_
- `Stanford University Sustainability Undergraduate Research in Geoscience and Engineering (SURGE) Program <https://sustainability.stanford.edu/our-community/access-belonging-community/surge>`_

The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.

- `LICENSE <https://github.com/we3lab/eeco/blob/main/LICENSE/>`_
- `CONTRIBUTING <https://github.com/we3lab/eeco/blob/main/CONTRIBUTING.rst/>`_

Attribution
===========

If you found this package useful, we encourage you to cite the papers below depending on which portion of the code you use.

See the metadata in `CITATION.cff <https://github.com/we3lab/eeco/blob/main/CITATION.cff>`_, on `Zenodo <https://doi.org/10.5281/zenodo.17102024>`_, 
or the following `BibTeX` format to cite the Python package in its entirety:

.. code-block:: 

  @software{chapin_2025_17102024,
    author={Chapin, Fletcher T. and
            Rao, Akshay K. and
            Sakthivelu, Adhithyan and
            Wettermark, Daly and
            Musabandesu, Erin and
            Jaminet, Anne and
            Dudchenko, Alexander V. and
            Mauter, Meagan S.},
    title={Electric Emissions \& Cost Optimizer (EECO)},
    month=sep,
    year=2025,
    publisher={Zenodo},
    version={v0.1.0},
    doi={10.5281/zenodo.17102025},
    url={https://doi.org/10.5281/zenodo.17102025}
  }

Citing `costs.py`
*****************

The development of `costs.py` was the culmination of two papers from the WE3Lab.

The convex formulation of tariff costs for optimizing flexible loads was originally developed for a case study of flexible wastewater treatment plant operation published in Environmental Science & Technology:

    Bolorinos, J., Mauter, M. S., & Rajagopal, R. Integrated energy flexibility management at wastewater treatment facilities. *Environ. Sci. Technol.* **57**, 18362-18371. (2023). DOI: `10.1021/acs.est.3c00365 <https://doi.org/10.1021/acs.est.3c00365>`_

In `BibTeX` format:

.. code-block:: 

  @article{bolorinos2023integrated,
    title={Integrated energy flexibility management at wastewater treatment facilities},
    author={Bolorinos, Jose and Mauter, Meagan S and Rajagopal, Ram},
    journal={Environmental Science \& Technology},
    volume={57},
    number={46},
    pages={18362--18371},
    year={2023},
    publisher={ACS Publications},
    url={https://doi.org/10.1021/acs.est.3c00365}
  }


The tariff data format was published in the following data descriptor in Nature Scientific Data:

    Chapin, F.T., Bolorinos, J. & Mauter, M.S. Electricity and natural gas tariffs at United States wastewater treatment plants. *Sci Data* **11**, 113 (2024). DOI: `10.1038/s41597-023-02886-6 <https://doi.org/10.1038/s41597-023-02886-6>`_

In `BibTeX` format:

.. code-block:: 
  
  @Article{Chapin2024,
  author={Chapin, Fletcher T and Bolorinos, Jose and Mauter, Meagan S.},
  title={Electricity and natural gas tariffs at United States wastewater treatment plants},
  journal={Scientific Data},
  year={2024},
  month={Jan},
  day={23},
  volume={11},
  number={1},
  pages={113},
  issn={2052-4463},
  doi={10.1038/s41597-023-02886-6},
  url={https://doi.org/10.1038/s41597-023-02886-6}
  }

Citing `emissions.py`
*********************

The emissions optimization code was originally developed for co-optimizing costs and emissions at a wastewater treatment plant and published in Environmental Science & Technology:

    Chapin, F.T., Wettermark, D., Bolorinos, J. & Mauter, M.S. Load-shifting strategies for cost-effective emission reductions at wastewater facilities *Environ. Sci. Technol.* **59**, 2285-2294 (2025). DOI: `10.1021/acs.est.4c09773 <https://doi.org/10.1021/acs.est.4c09773>`_

In `BibTeX` format:

.. code-block:: 
  
  @article{chapin2025load,
    title={Load-Shifting Strategies for Cost-Effective Emission Reductions at Wastewater Facilities},
    author={Chapin, Fletcher T and Wettermark, Daly and Bolorinos, Jose and Mauter, Meagan S},
    journal={Environmental Science \& Technology},
    volume={59},
    number={4},
    pages={2285--2294},
    year={2025},
    publisher={ACS Publications},
    url={https://pubs.acs.org/doi/10.1021/acs.est.4c09773}
  }

Citing `metrics.py`
*******************

The flexibility metrics come from the following Nature Water paper:

    Rao, A. K., Bolorinos, J., Musabandesu, E., Chapin, F. T., & Mauter, M. S. Valuing energy flexibility from water systems. *Nat. Water* **2**, 1028-1037 (2024). DOI: `10.1038/s44221-024-00316-4 <https://doi.org/10.1038/s44221-024-00316-4>`_

In `BibTeX` format:

.. code-block:: 
  
  @article{rao2024valuing,
    title={Valuing energy flexibility from water systems},
    author={Rao, Akshay K and Bolorinos, Jose and Musabandesu, Erin and Chapin, Fletcher T and Mauter, Meagan S},
    journal={Nature Water},
    volume={2},
    number={10},
    pages={1028--1037},
    year={2024},
    publisher={Nature Publishing Group UK London},
    url={https://doi.org/10.1038/s44221-024-00316-4}
  }
