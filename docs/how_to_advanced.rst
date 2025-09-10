.. contents::

.. WARNING::

  Site under construction!    
  Documentation incomplete :( 

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

The `demand_scale_factor` parameter allows you to scale demand charges to reflect shorter optimization horizons or to prioritize demand differently across sequential optimization horizons.

By default, `demand_scale_factor=1.0`. Use values less than 1.0 when solving for a subset of the billing period, or to adjust demand charge weighting in sequential optimization.

When `demand_scale_factor < 1.0`, demand charges are proportionally reduced to reflect the shorter optimization horizon. This is useful for:
- Moving horizon optimization where you solve for sub-periods of the billing cycle
- Sequential optimization where you want to reduce demand charge weighting as time goes on in the month

.. code-block:: python

    from electric_emission_cost import costs
    
    # E.g. solving for 3 days out of a 30-day billing period
    demand_scale_factor = 3 / 30
    
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data,
        demand_scale_factor=demand_scale_factor
        # ...
    )

For more details on applying the sequential optimization strategy, see:

&nbsp; Bolorinos, J., Mauter, M.S. & Rajagopal, R. Integrated Energy Flexibility Management at Wastewater Treatment Facilities. *Environ. Sci. Technol.* **57**, 46, 18362–18371 (2023). DOI: [10.1021/acs.est.3c00365](https://doi.org/10.1021/acs.est.3c00365)

In `bibtex` format:

.. code-block:: bibtex

   @Article{Bolorinos2023,
   author={Bolorinos, Jose
   and Mauter, Meagan S.
   and Rajagopal, Ram},
   title={Integrated Energy Flexibility Management at Wastewater Treatment Facilities},
   journal={Environmental Science & Technology},
   year={2023},
   month={Jun},
   day={16},
   volume={57},
   number={46},
   pages={18362--18371},
   doi={10.1021/acs.est.3c00365},
   url={https://doi.org/10.1021/acs.est.3c00365}
   }


.. _decompose-exports:

How to Use `decomposition_type`
==============================

The `decomposition_type` parameter allows you to decompose consumption data into positive (imports) and negative (exports) components. This is useful when you have export charges or credits in your rate structure.

Options:
- Default `None`
- `"binary_variable"`: To be implemented
- `"absolute_value"`

.. code-block:: python

    from electric_emission_cost import costs
    
    # Example with export charges
    charge_dict = {
        "electric_export_0_2024-07-10_2024-07-10_0": np.ones(96) * 0.025,
    }
    
    consumption_data = {
        "electric": np.concatenate([np.ones(48) * 10, -np.ones(48) * 5]),
        "gas": np.ones(96),
    }
    
    # Decompose consumption into imports and exports
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data,
        decomposition_type="absolute_value"
    )

When decomposition_type is not None the function creates separate variables for positive consumption (imports) and negative consumption (exports)
and applies export charges only to the export component.
For Pyomo models, decomposition_type adds a constraint total_consumption = imports - exports


.. _varstr-alias:

How to Use `varstr_alias_func`
==============================
