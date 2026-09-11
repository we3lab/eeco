.. contents::

.. _demandresponse:

**************
demandresponse
**************

This module computes demand response revenue from electricity consumption
data, modeled along two composable axes -- :class:`BaselineMethod` for how
the counterfactual is measured and :class:`PaymentStructure` for how it is
paid -- with three entry points: ``calculate_dr_revenue``,
``build_dr_revenue`` (pyomo only, nets into ``model.objective``), and
``calculate_itemized_dr_revenue`` (ex-post, per-event breakdown).

.. automodule:: eeco.demandresponse
   :members:
