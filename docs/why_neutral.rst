.. contents::

.. _why-neutral:

*****************************************
Why Design a Pyomo/CVXPY Neutral Library?
*****************************************

`Pyomo <https://www.pyomo.org/>`_ and `CVXPY <https://www.cvxpy.org/>`_ are the two most widely used optimization modelling languages in Python.
As a result, both are widely used in scientific research across applications such as computer hardware design, electric grid modeling, and assembly line optimization.

Without getting too into the weeds, each library has its own pros and cons. CVXPY has native support for matrices and is generally lighter weight, 
leading to shorter model intialization and solve times. 
Pyomo supports nonconvex optimization, enabling it to solve a much larger set of problems than CVXPY. 
Additionally, Pyomo's object-oriented nature can be helpful in organizing complex models, but could be considered a con as well as it can lead to unnecessary overhead.

Since there are good reasons to use both CVXPY and Pyomo depending on the application, we do not want to make that decision for our users.
With that in mind, we could have created two separate packages, one for CVXPY and one for Pyomo, and in a lot of ways that would have been a more straightforward approach.
However, we feel that maintaining a Pyomo/CVXPY neutral library has three key benefits:

#. *Generalizability*: A useful model or other software tool can reach a larger audience of both Pyomo and CVXPY experts and avoid siloing research.
#. *Reproducibility*: A researcher can validate results from a CVXPY model in Pyomo using the same cost calculation code. In other words,
   Using the same functions allows for researchers to be certain that any discrepancies are due to the models themselves, not the cost calculation code.
#. *Quality*: Maintaining correct code becomes easier because bug fixes are done for both Pyomo and CVXPY simultaneously rather than having to keep
   two separate repositories in sync. Additionally, bugs are more likely to be discovered by widening the user base to both CVXPY and Pyomo.

As an example of the utility of a Pyomo/CVXPY neutral package, the WE3Lab has been working on distributed optimization to coordinate operation between multiple water systems.
We use this package to link together cost calculations from each water system subproblem when there are both CVXPY and Pyomo subproblems.
When orchestrating the solver, it is extremely helpful to use the same cost calculation code across subproblems regardless of the underlying model architecture.