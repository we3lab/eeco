************
Contributing
************

Contribution Agreement
======================
This section describes the terms under which you may make “Contributions” — 
which may include without limitation, software additions, revisions, bug fixes, configuration changes, documentation, or any other materials —
to any of the projects owned or managed by the WE3Lab. 
If you have questions about these terms, please contact Professor Mauter at mauter@stanford.edu.

You certify that:

- Your Contributions are either:

  1. Created in whole or in part by you and you have the right to submit them under the designated license (described below); or
  2. Based upon previous work that, to the best of your knowledge, 
     is covered under an appropriate open source license and you have the right under that license to submit that work with modifications, 
     whether created in whole or in part by you, under the designated license; or
  3.	Provided directly to you by some other person who certified (1) or (2) and you have not modified them.
- You are granting your Contributions to the WE3Lab under the terms of the MIT license (the “designated license”).
- You understand and agree that the WE3Lab projects and your Contributions are public and that a record of the Contributions 
  (including all metadata and personal information you submit with them) is maintained indefinitely 
  and may be redistributed consistent with the WE3Lab's mission and the MIT license.

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
