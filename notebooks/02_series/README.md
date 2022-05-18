## Notebook 02_series section
Set of Notebooks on higher dimensional input. Typically, 3 dimensional.
- [01_stacked](https://github.com/t0kk35/d373c7/tree/master/notebooks/02_series/01_stacked). The stacked series examples have a list of transactions and input. In the single examples the Neural Nets were asked to score one transaction at a time. In these examples they will use 3 dimensional input. The shape of the training input will be `(Batch x List-size x Features)`
- [02_frequency](https://github.com/t0kk35/d373c7/tree/master/notebooks/02_series/01_frequency). Frequency series also have 3 dimensional output, but the second dimension contains aggregates in a time bin. For instance a 5-day sum frequency, would contain 5 entries in the second dimension containing the sum 4 days ago, 3 days ago, 2 days ago, 1 day ago, today. The shape of the training input will be `(Batch x Frequency x Features)`

