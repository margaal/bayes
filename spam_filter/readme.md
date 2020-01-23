# Spam filter using Naive Bayes (python)

This is a simple sms spam filter based on Naïve Bayes algorithm. Naïve Bayes is a supervised learning algorithm used for classification. It is particularly useful for text classification issues. It is based on the bayes theorem which is used to calculate the conditional probability. It assumes that a strong independence exist between the features, hence the name naïve Bayes.

## Prerequisites

- Understand [Naïve Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier algorithm
- See also [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) model
- Know how to use certain module classes of [Sickit Learn](https://scikit-learn.org)
- Handle csv with [pandas](https://pandas.pydata.org)
- Know the value of [Sickit Learn Pipeline](https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976)
- (Optional) [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) allows to assign non-zero probabilities to words that do not appear in the sample. It's used by MultinomialNB (attribute alpha=1)

## Reasons to choose Multinomial Naïve Bayes classifier
We have many sort of Naïve Bayes algorithm (Gaussian NB, Bernoulli NB) but the best for this problem is Multinomial Naïve Bayes because:
- Naive Bayes multinomial classifier is suitable for classification with discrete features
- We want to make a filter taking into account the frequency of the words in the document
For more [information](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf)

## Steps
The code is fully documented, check it to see how you can implement the following steps. 

1. Load data
2. Explore and visualize data
[image 1](../resources/screenshots/dataset_head.jpg)
[image 2](../resources/screenshots/length_repartition.jpg)
0 : Ham
1 : Spam
3. Prepare data for Model
4. Create Pipeline to build Bag of Words, normalize features and apply classification algorithm (In this case MultinomialNB)
5. Model evaluation
[image 3](../resources/screenshots/model_evaluation_1.jpg)

## References
- [Kaggle](https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier/notebook)
- [Sickit Learn Pipeline](https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976)
