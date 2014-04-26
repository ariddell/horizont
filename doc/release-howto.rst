This documentation is intended for use by developers.

Release checklist
=================

- Update doc/whats_new.rst
- ``git tag``
- Push tag to github ``git push --tags``
- Git branch ``master`` should be fast-forwarded to ``develop``
- Assemble source distribution, sign it, upload to PyPI:

::

    python setup.py sdist
    gpg --detach-sign -a dist/horizont-x.x.x.tar.gz
    twine upload dist/*

After release
=============

- Add placeholder for changes in doc/whats_new.rst
- Bump version in horizont/__init__.py

See also
========
- http://docs.astropy.org/en/v0.2/development/building_packaging.html
