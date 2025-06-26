.. contents::

.. _dataformat:

************
Data Formats
************

.. _dataformattariff:

Electricity Tariffs
===================

Each row in the tariff data corresponds to a different charge. 
E.g. a municipality with a flat electricity tariff would have only one charge and therefore one row
E.g. a municipality with a complex tariff would have many rows corresponding to many charges

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
- **units**: Units of the charge, e.g. “$/kWh”. If units are different between imperial and metric then imperial is listed followed by metric. E.g., “$/therm or $/m3”.
- **Notes**: Any comments the authors felt would help explain unintuitive decisions in data collection or formatting.

The electricity and natural gas tariff data format used in this project is based on the following reference.

&nbsp; Chapin, F.T., Bolorinos, J. & Mauter, M.S. Electricity and natural gas tariffs at United States wastewater treatment plants. *Sci Data* **11**, 113 (2024). DOI: [10.1038/s41597-023-02886-6](https://doi.org/10.1038/s41597-023-02886-6)

The raw data can also be cited directly from Figshare:

&nbsp; Chapin, F.T., Bolorinos, J., & Mauter, M. S. Electricity and natural gas tariffs at United States wastewater treatment plants. *figshare* https://doi.org/10.6084/m9.figshare.c.6435578.v1 (2024).

In `bibtex` format:

```
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
```

.. _dataformatemissions:

Scope 2 Emissions
=================
