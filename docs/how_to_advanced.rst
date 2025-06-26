.. contents::

.. _how-to-advanced:

****************************
How to Use Advanced Features
****************************

There are a few advanced features that can be used via flags in `calculate_cost`.
These features are particularly useful for moving horizon optimization. 
Check out :ref:`why-advanced` for more background as to why we recommend moving horizon optimization.

.. _prev-consumption:

How to Use `prev_demand_dict` and `prev_consumption_dict`
=========================================================

By default, `prev_demand_dict=None` and `prev_consumption_dict=None`. 
However, a user may want to ...

TODO: :ref:`why-prev-consumption`

.. code-block:: python

    from electric_emission_cost


.. _consumption-est:

How to Use `consumption_estimate`
=================================


.. _scale-demand:

How to Use `demand_scale_factor`
================================
