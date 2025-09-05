.. contents::

.. WARNING::

  Site under construction!    
  Documentation incomplete :( 

.. _why-neutral:

*****************************************
Why Design a Pyomo/CVXPY Neutral Library?
*****************************************

Bullets to formulate into paragraphs:

  - Both libraries are widely used

    - Pros and cons of each

      - cvx: native support for matrices, computation time
      - pyo: nonconvex support, object-oriented (could be considered a con as well - not as lightweight)
      - etc.
  - Do not want the energy research community siloed
  - Reproducability and code quality

    - Using the same functions allows for results between researchers using different methods to be consistent
    - We know that any discrepancies are due to their model itself, not the cost calculation
    - Bugs are more likely to be discovered the more widely used the package is,
      and when they are fixed, there will not be two versions out of sync with one another
  - Distributed optimization with multiple libraries

    - Maybe this is a niche use case, 
      but we use this package to link together cost calculations from a joint opt. problem with both CVXPY and Pyomo subproblems
    - It's nice to have the same dependency for every subproblem
