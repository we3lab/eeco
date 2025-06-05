.. contents::

.. _dataformat:

************
Data Formats
************

.. _dataformattariff:

Electricity Tariffs
===================

The electricity and natural gas tariff data format used in this project is based on the structure described in:

Chapin, F.T., Bolorinos, J. & Mauter, M.S. Electricity and natural gas tariffs at United States wastewater treatment plants. Sci Data 11, 113 (2024). https://doi.org/10.1038/s41597-023-02886-6

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

.. _dataformatemissions:

Scope 2 Emissions
=================
