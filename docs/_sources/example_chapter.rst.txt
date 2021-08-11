.. _a_chapter:

=========
A Chapter
=========

Here is a paragraph

Here are:
    - Bullet lists in reStructuredText are fairly straightforward.
	- A bunch of bullet points.

	  - Nested bullets should be indented by **exactly** two spaces.
	  - They also need a blank line before and after them, otherwise it doesn't
		work right.

	- It is also possible to have multiple paragraphs in a bullet.

	  Make sure to have a blank line between the paragraphs, and to indent the
	  paragraphs to the correct level.
	  
	  
	  
1. Numbered lists are not complicated.
2. They do exactly what you think they do.

.. figure:: images/digital_twin_component_basic.svg
    :width: 400px
    :align: center
    :figclass: align-center
    :alt: Text description of the image

    This is the caption of an svg file
	
.. image:: https://source.unsplash.com/200x200/daily?cute+animals
   :align: center


.. _a_section:

A Section
=========

A link in rst is like
`this <https://www.octue.com>`_.

You cna create numbered bullets too. Here are some things that were important in the library which I made this template from:

#. Openness
#. Federation
#. Security
#. Public Good (cross ref other parts of the docs like this... :ref:`using_inline_tabs`)

---

Tables
======

Simple table:

=====  =====  ======
   Inputs     Output
------------  ------
  A      B    A or B
=====  =====  ======
False  False  False
True   False  True
False  True   True
True   True   True
=====  =====  ======

Definition
==========

Definition Lists
   An extremely useful tool for describing stuff.

Complicated Term
   Definition


.. _using_inline_tabs:

Inline Tabs
===========


.. tab:: Good Looks

      This actually looks better than group tabs

.. tab:: Even better looks
	  
	  This also suits the original look and feel of theme
	  
	.. code:: python

		print("And with the directive syntax, you can have syntax highlighting.")

.. tab:: Some disabled code
	.. code:: none

		print("Or disable all syntax highlighting.")




