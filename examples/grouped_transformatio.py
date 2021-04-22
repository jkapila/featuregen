"""
Grouped Varaible Transformation Example
=======================================

A group wise behaviour is very common in panel data. Now making transofmration on
whole from one prespectives would not be ideal and may cause more harm than solving the
scaling problem.

To keep the consistency of the transformation within groups GroupedVariableTransformation
can be used. Following is very minimilatic example of the same.

"""


import pandas as pd

from featuregen import GroupedVariableTransformation


#%%
# Creating a grouped data set with range differnece in values
df = pd.DataFrame(
    {
        "attribute": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        "value": [1, 2, 4, 5, 3, 6, 100, 33, 44, 77, 77, 99],
    }
)
print(df)
#%%
# Instantiating model calss with relevtant attributes
gvt = GroupedVariableTransformation(key="attribute", target="value")
print(gvt)

#%%
# Fitting our grouped data in transformer
gvt.fit(df)

#%%
# Transforming data toe scale with zscore strategy
df_tr = gvt.transform(df)
print(df_tr)

#%%
# Inverse transforming the  data back based on groupwise learned scale.
df_inv = gvt.inverse_transform(df_tr)
print(df_inv)
