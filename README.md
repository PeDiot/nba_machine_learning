# Machine Learning applied to NBA games data

## Purpose of the project

In the field of sports, predictive analysis has played an important role for the past two decades. Some say the starting point has been the team’s analytical, evidence-based approach of the baseball manager Billy Beane whose goal was to assemble a competitive baseball team despite a small budget. At the same time, machine learning has become a powerful tool to predict various outcomes. That is why we choose the [kaggle](https://www.kaggle.com/ionaskel/nba-games-stats-from-2014-to-2018) data set related to the results of all NBA games from 2014 to 2018.

The objective of our project is to predict the odds of winning for each team given the games’ features. The target variable used in the models is `Win`.

## Workflow

The method we use to split the data is to randomly select 70% of the dates we have in our data. Then we build the train data set using these 70% dates and the test data set is made of the remaining dates.

Once the data set split, the following ML algorithms are trained, then tested:

- LDA & QDA,
- Logistic regression,
- k-NN,
- Decision trees,
- Bagging & Random forests,
- Boosted decision trees.

The final step consists of comparing the different models in terms of accuracy in order to pick the one that best predicts the odds of winning an NBA game.
