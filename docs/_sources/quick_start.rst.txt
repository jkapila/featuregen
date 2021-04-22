.. _quick_start:

============
Quick Start
============

.. _create_a_twine:

Just to quick start things
==========================

To use library

.. code:: py


   import pandas as pd
   from featuregen import GroupedVariableTransformation

   df = pd.DataFrame({'attribute':['A','A','A','A','A','A','B','B','B','B','B','B'],
                      'value':[1,2,4,5,3,6,100,33,44,77,77,99]})
   gvt = GroupedVariableTransformation(key='attribute',target='value')
   gvt.fit(df)
   print(gvt)
   gvt.transform(df)
