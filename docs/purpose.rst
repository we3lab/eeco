.. contents::

.. _purpose:

************************************
What is the Purpose of this Package?
************************************

This package was created by researchers from the `WE3Lab <https://we3lab.stanford.edu/>`_ at Stanford University to assist in their energy-water nexus research. 
Our goal is to address the complexity of computing the electricity bill of an industrial consumer.
We originally developed these methods while evaluating the cost-minimizing energy management strategies for water systems, 
such as wastewater treatment plants [`Musabendesu 𝐽. 𝐶𝑙𝑒𝑎𝑛. 𝑃𝑟𝑜𝑑. (2021) <https://doi.org/10.1016/j.jclepro.2020.124454>`_; `Bolorinos 𝐸𝑛𝑣𝑖𝑟𝑜𝑛. 𝑆𝑐𝑖. 𝑇𝑒𝑐ℎ𝑛𝑜𝑙. (2023) <https://doi.org/10.1021/acs.est.3c00365>`_; `Chapin 𝐸𝑛𝑣𝑖𝑟𝑜𝑛. 𝑆𝑐𝑖. 𝑇𝑒𝑐ℎ𝑛𝑜𝑙. (2025) <https://doi.org/10.1021/acs.est.4c09773>`_],
desalination facilities [`Rao 𝐴𝐶𝑆 𝑆𝑢𝑠𝑡𝑎𝑖𝑛. 𝐶ℎ𝑒𝑚. 𝐸𝑛𝑔. (2024) <https://doi.org/10.1021/acssuschemeng.4c06353>`_],
and water distribution [`Rao 𝑁𝑎𝑡. 𝑊𝑎𝑡𝑒𝑟 (2024) <https://doi.org/10.1038/s44221-024-00316-4>`_].

Unlike wholesale market pricing mechanisms, such as locational marginal prices (LMPs), which are straightforward to calculate directly as a timeseries in units of '$/MWh',
retail electricity prices are broken down into a set of discrete charges. In particular, the **demand charge** is the maximum consumption during a specified time period,
so computing a demand charge is a nonlinear operation (:ref:`complexities`).

In addition to the mathematical complexity behind computing the electricity bill and optimizing bill savings, the wide variation in tariff structure is a challenge.
Some regions offer flat prices, whereas others have up to four different time-of-use (TOU) pricing periods
(i.e., "super-off-peak", "off-peak", "mid-peak", and "on-peak" periods).
We designed a machine-readable format for encoding all relevant tariff data, 
including TOU periods, tiered charges, daily demand charges, and unit conversions [`Chapin 𝑁𝑎𝑡. 𝑆𝑐𝑖. 𝐷𝑎𝑡𝑎 (2024) <https://doi.org/10.1038/s41597-023-02886-6>`_].

Our research focuses on investigating the benefits of flexible operation of water systems, but we hope that this package 
will be useful to the wider community for seamlessly optimizing industrial consumers' electricity-related costs and emissions!