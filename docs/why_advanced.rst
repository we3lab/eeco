.. contents::

.. _why-advanced:

***********************************
Why Do the Advanced Features Exist?
***********************************

.. _why-prev-consumption:

What Is the Purpose of `prev_demand_dict` and `prev_demand_dict`?
=================================================================


.. _why-consumption-est:

What Is the Purpose of `consumption_estimate`?
==============================================


.. _why-scale-demand:

What Is the Purpose of `demand_scale_factor`?
=============================================

[cite Bolorinos, Chapin]: gist is that it is necessary for moving horizon optimization 
because the certainty around demand charges and therefore relative weight between demand and energy charges
varies throughout the month.