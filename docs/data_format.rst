.. contents::

.. _data-format:

******************
How to Format Data
******************

.. _data-format-tariff:

Electricity Tariffs
===================

The electricity tariff data in tariff.csv represents the industrial tariff structure for one facility.

The columns are:

- **utility**: Type of utility, i.e., "electric" or "gas".
- **type**: Type of charge. Options are "customer", "demand", and "energy".
- **period**: Name for the charge period. Only relevant for demand charges, since there can be multiple concurrent demand charges. E.g., a charge named "maximum" that is in effect 24 hours a day vs. a charge named "on-peak" that is only in effect during afternoon hours.
- **basic_charge_limit (imperial)**: Consumption limit above which the charge takes effect in imperial units (i.e., kWh of electricity and therms of natural gas). Default is 0. A limit is in effect until another limit supersedes it.
- **basic_charge_limit (metric)**: Consumption limit above which the charge takes effect in metric units (i.e., kWh of electricity and m3 of natural gas). Default is 0. A limit is in effect until another limit supersedes it.
- **month_start**: First month during which this charge occurs (1–12).
- **month_end**: Last month during which this charge occurs (1–12).
- **hour_start**: Hour at which this charge starts (0–24).
- **hour_end**: Hour at which this charge ends (0–24).
- **weekday_start**: First weekday on which this charge occurs (0 = Monday to 6 = Sunday).
- **weekday_end**: Last weekday on which this charge occurs (0 = Monday to 6 = Sunday).
- **charge (imperial)**: Cost represented as a float in imperial units. I.e., “$/month”, “$/kWh”, “$/kW”, “$/therm”, and “$/therm/hr” for customer charges, electricity energy charges, electric demand charges, natural gas energy charges, and natural gas demand charges, respectively.
- **charge (metric)**: Cost represented as a float in metric units. I.e., “$/month”, “$/kWh”, “$/kW”, “$/m3”, and “$/m3/hr” for customer charges, electricity energy charges, electricity demand charges, natural gas energy charges, and natural gas demand charges, respectively. A conversion factor of 2.83168 cubic meters to 1 therm was used.
- **units**: Units of the charge, e.g., “$/kWh”. If units are different between imperial and metric, then imperial is listed followed by metric. E.g., “$/therm or $/m3”.
- **Notes**: Any comments the authors felt would help explain unintuitive decisions in data collection or formatting.

Each row in the tariff data corresponds to a different charge. 
For example, a municipality with a flat electricity tariff would have only one charge and therefore one row. A municipality with a complex tariff would have many rows corresponding to many charges.

The electricity and natural gas tariff data format used in this project is based on the following reference.
Additional industrial tariff examples are available in the same reference:

   Chapin, F.T., Bolorinos, J. & Mauter, M.S. Electricity and natural gas tariffs at United States wastewater treatment plants. *Sci Data* **11**, 113 (2024). DOI: `10.1038/s41597-023-02886-6 <https://doi.org/10.1038/s41597-023-02886-6/>`_

The raw data can also be cited directly from Figshare:

   Chapin, F.T., Bolorinos, J., & Mauter, M. S. Electricity and natural gas tariffs at United States wastewater treatment plants. *figshare* https://doi.org/10.6084/m9.figshare.c.6435578.v1 (2024).

In `bibtex` format:

.. code-block:: bibtex

   @Article{Chapin2024,
   author={Chapin, Fletcher T.
   and Bolorinos, Jose
   and Mauter, Meagan S.},
   title={Electricity and natural gas tariffs at United States wastewater treatment plants},
   journal={Scientific Data},
   year={2024},
   month={Jan},
   day={23},
   volume={11},
   number={1},
   pages={113},
   abstract={Wastewater treatment plants (WWTPs) are large electricity and natural gas consumers with untapped potential to recover carbon-neutral biogas and provide energy services for the grid. Techno-economic analysis of emerging energy recovery and management technologies is critical to understanding their commercial viability, but quantifying their energy cost savings potential is stymied by a lack of well curated, nationally representative electricity and natural gas tariff data. We present a dataset of electricity tariffs for the 100 largest WWTPs in the Clean Watershed Needs Survey (CWNS) and natural gas tariffs for the 54 of 100 WWTPs with on-site cogeneration. We manually collected tariffs from each utility's website and implemented data checks to ensure their validity. The dataset includes facility metadata, electricity tariffs, and natural gas tariffs (where cogeneration is present). Tariffs are current as of November 2021. We provide code for technical validation along with a sample simulation.},
   issn={2052-4463},
   doi={10.1038/s41597-023-02886-6},
   url={https://doi.org/10.1038/s41597-023-02886-6}
   }

   @misc{chapin_bolorinos_mauter_2024, 
   author={Chapin, Fletcher T. 
   and Bolorinos, Jose 
   and Mauter, Meagan S.}, 
   title={Electricity and natural gas rate schedules at U.S. wastewater treatment plants}, 
   url={https://springernature.figshare.com/collections/Electricity_and_natural_gas_rate_schedules_at_U_S_wastewater_treatment_plants/6435578/1}, 
   DOI={10.6084/m9.figshare.c.6435578.v1}, 
   abstractNote={Electricity and natural gas tariffs of the 100 largest wastewater treatment plants in the United States}, 
   publisher={figshare}, 
   year={2024}, 
   month={Jan}
   }

.. _data-format-emissions:

Scope 2 Emissions
=================

The emissions dataset includes hourly average CO\ :sub:`2`\ e emissions intensity values for each calendar month. 
The hourly CO\ :sub:`2`\ e emissions are calculated by aggregating values for electric-generating units in the continental U.S. that are metered by balancing authorities and used to serve required demand. 
The data may represent Scope 2 emissions at the level of balancing authority, U.S. region, or state.


The columns are:

- **month**: The calendar month for which this grid emissions value is relevant (1–12).
- **hour**: The hour for which this grid emissions value is relevant (0–23).
- **co2_eq_kg_per_MWh**: The average grid emissions intensity for a given hour during a given calendar month, given in kg CO\ :sub:`2`\ e per MWh.

Each row in the emissions data corresponds to a different hourly average emissions intensity. 
For example, there are 24 rows for January. The row with month "1" and hour "0" represents the average emissions intensity from 00:00 - 00:59 across all days in January.

The Scope 2 emissions data format used in this project is based on the following reference:
   
   de Chalendar, J.A., Taggart, J. & Benson, S.M. Tracking emissions in the US electricity system. *Proc Natl Acad Sci USA* **116**, 25497-25502 (2019). DOI: `10.1073/pnas.1912950116 <https://doi.org/10.1073/pnas.1912950116/>`_

In `bibtex` format:

.. code-block:: bibtex

   @Article{deChalendar2019,
   author={de Chalendar, Jacques A.
   and Taggart, John
   and Benson, Sally M.},
   title={Tracking emissions in the US electricity system},
   journal={Proceedings of the National Academy of Sciences},
   year={2019},
   month={Dec},
   volume={116},
   number={51},
   pages={25497-25502},
   doi={10.1073/pnas.1912950116},
   url={https://doi.org/10.1073/pnas.1912950116}
   }

Additional historical Scope 2 emissions data examples are available from de Chalendar 2019. Ongoing updates to the grid emissions data, as well as region and balancing authority definitions, are made available from the `EIA Hourly Electric Grid Monitor <https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48>`_.