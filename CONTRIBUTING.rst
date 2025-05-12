**********************
Contribution Agreement
**********************

Contribution Status
===================
We use this for internal purpose and are currently not accepting pull requests. We are releasing it to the community AS IS.

.. _CONTRIBUTING_Deploying:

Deploying
=========

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

.. code-block:: bash

  git checkout main
  git pull
  bumpversion [part]
  git push
  git branch -D stable
  git checkout -b stable
  git push --set-upstream origin stable -f
  git push --tags

Where ``[part]`` is ``major``, ``minor``, or ``patch``.
This will release a new package version on Git + GitHub and publish to PyPI.
