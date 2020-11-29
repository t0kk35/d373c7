# d373c7
## Some Deep Learning 4 Financial Crime Experiments
Deep Learning is probably not used enough in financial crime fraud detection. This repo hosts some code and experiments which should hopefully help one to get running experiments quickly on financial data such as credit card transactions or payments.

The examples use PyTorch as DeepLearning framework. PyTorch is a fantastic framework, but it does have somewhat of a learning curve. In-depth knowledge of PyTorch, Pandas and Numpy is not needed to run the examples. They will start high level, try to abstract some of the details, but allow someone who is interested to dig into the nitty gritty network definitions.

## Why would you?
At present (late 2020), Financial Crime detection is mainly done by taking an input source, for instance a transaction, enriching that source data with hand crafted aggregated features and running the enriched data through classifiers. Examples of aggregated features could be the 6 month average per customer. The accumulated 24h withdrawal per card. The count of cash deposits per week per account etc....

The aggregated enriched features provide context to the models, they are vital to getting good results. Finding the correct features is a manual and fairly intensive human job. In the overall process of setting up and training a classifier fraud model around 70~80% of time is spent in the feature engineering prior to actually classifying.

Neural Nets have the nice property that the feature engineering is no longer a human job, but that it becomes the task of the machine. The models no longer just classify the data, but also need to come up with all sorts of latent representations (i.e. features). Domains like NLP, Vision style problems used to hand craft features, but they all changed to Deep Neural Nets around 2012, and made massive progress since. 

It's probably unrealistic to think Deep Neural Nets will be able to tackle all Financial Crime detection problems, especially when very long running data, say 15-month avergages are needed. But there should be use cases where they thrash the hand-crafting approach much like in NLP and Vision. Should be fun figuring out which ...

## Why would you not?
Deep Neural Nets are data hungry beasts; they require a lot of data to train. In some Financial Crime domains there is simply not enough labelled data available. Add to that the fact that the data is often tremendously unbalanced. It can feel a bit like a needle in a haystack sort of problem. If one were to try and use a classifier to find 100 fraud records in 10.000.000 payments, then using neural nets should not be the first choice.

But no stress; some creative use of data augmentation, auto-encoders, transfer learning, semi-supervised learning should be able to alleviate some of these challenges. Should be fun trying to figure out how ...

## Exmples
This repo contains a set of examples in the [Notebooks](https://github.com/t0kk35/d373c7/tree/master/notebooks) section. They are a good place to start and get a feel for what can be achieved with Neural Networks. The sub-directories and notebooks follow a numbering scheme and are like the chapters in a book, ff followed they hopefully convey a story. (The notebooks can be opened directly in GitHub, but you sometimes need to click 'Reload' a couple of times ...)

Most examples use the [banksim1](https://www.kaggle.com/ntnu-testimon/banksim1) data-set. It is a fairly limited and simple data-set but has the advantage it is not PCA'ed. A lot of the public sample data for Fraud and Credit Card Fraud are PCA'ed, which makes them less usefull to show-off Neural Nets. Finding the fraud in this data-set is not complicated at all, if anything it should show the potential architectures. In order to actually compare performance a more complex and bigger data-set would be needed.

---
Under __Heavy__ Construction
---
License: Under Apache 2.0 See [License](./LICENSE)
