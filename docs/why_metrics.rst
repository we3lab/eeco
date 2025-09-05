.. contents::

.. _why-metrics:

****************************
Why Use Flexibility Metrics?
****************************

Flexibility metrics refer to the quantification of flexible industrial electricity loads from our previous research [`Rao 𝑁𝑎𝑡. 𝑊𝑎𝑡𝑒𝑟 (2024) <https://doi.org/10.1038/s44221-024-00316-4>`_].
These metrics include round-trip efficiency (RTE), energy capacity, and power capacity.

  - **Round-trip Efficiency (RTE)**: the fraction of energy returned after storage
  - **Energy Capacity**: the amount of energy that may be stored or shifted
  - **Power Capacity**: the speed at which energy can be charged or discharged

While these flexibility metrics are certainly not required, we think that using a unified method to quantify flexibility is a good idea for a few reasons:

  1. It enables apples-to-apples comparisons across diverse energy assets, from Li-ion batteries to wastewater storage tanks to biogas holders.
  2. It directs future research by highlighting where technological improvement is most valuable. 
     E.g., would improving a battery's RTE or discharge rate be more beneficial in terms of costs and emissions?
  3. It provides a standard method to constrain the system when formulating an optimization problem rather than creating custom constraints for every model.

Hopefully, we have convinced you that using flexibility metrics is indeed worthwhile. 
If so, find out more at :ref:`how-to-metrics`.
If you'd like to read more about how we apply these flexibility metrics in our research, check out `lvof.we3lab.tech <https://lvof.we3lab.tech>`_.