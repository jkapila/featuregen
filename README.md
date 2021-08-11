# FeatureGen


**featuregen** is a package that holds many common feature generating facilities for Predictive Modelling purposes. Please read docs to for example usage.

### **Modules:**
* Aggregation
* Transfromation
* Time based utilities
* Summary Generation
* Utility Funcitons

A simple quickstart

```python
import pandas as pd
from featuregen import GroupedVariableTransformation

df = pd.DataFrame(
    {'attribute':['A','A','A','A','A','A','B','B','B','B','B','B'],
     'value':[1,2,4,5,3,6,100,33,44,77,77,99]})
gvt = GroupedVariableTransformation(key='attribute',target='value')
gvt.fit(df)
print(gvt)
gvt.transform(df)

```

The Documentation can be found at [here](https://jkapila.github.io/featuregen/)
