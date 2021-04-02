# Decision trees using the complete data set  "nba.data.cleaned.csv"

# http://www.sthda.com/english/articles/35-statistical-machine-learning-essentials/141-cart-model-decision-tree-essentials/
# https://koalaverse.github.io/homlr/notebooks/09-decision-trees.nb.html

# Packages
library(rpart)
library(dplyr)
library(rsample)
library(rpart.plot)
library(rattle)  # Fancy tree plot
library(kableExtra)
library(caret)
library(ggplot2)
library(ggpubr)
library(vip)

theme_set(theme_minimal())

# Dataset

setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/trees")
source("auxiliaire.R")
train.data <- read.csv("train.data.reduced.csv")
test.data <- read.csv("test.data.reduced.csv")

#
train.data <- train.data[, -c(1:4)]
test.data <- test.data[, -c(1:4)]

#
train.data <- get_cat_var(
  train.data,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)

test.data <- get_cat_var(
  test.data,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)

str(train.data)
str(test.data)


# Using the 'caret' package

#
rep <- 5
cv <- 10
ctrlCv <- trainControl(method = "repeatedcv", repeats = rep, number = cv)

min_cp <- 0.01
max_cp <- 0.1
tune.gridcp <- expand.grid(cp = seq(min_cp, max_cp, 0.01))

system.time(
  rpartFit1 <- train(
    Win~., 
    data = train.data,
    method = "rpart",
    trControl = ctrlCv,
    tuneGrid = tune.gridcp,
    metric = "Accuracy"
  )
)

rpartFit1
rpartFit1$bestTune
rpartFit1$results
rpartFit1$finalModel

rpartFit1 %>%
  plot()
prp(rpartFit1$finalModel, type=1, extra=1, split.box.col="lightblue",cex=0.6)

#
depth_min <- 2
depth_max <- 20
tune.gridcart <- expand.grid(maxdepth = depth_min:depth_max)
system.time(
  rpartFit2 <- train(
    Win~., 
    data = train.data,
    method = "rpart2",
    tuneGrid =tune.gridcart,
    trControl = ctrlCv,
    metric = "Accuracy"
  )
)

rpartFit2
rpartFit2 %>%
  plot()

rpartFit2$finalModel
prp(rpartFit2$finalModel, type=1, extra=1, split.box.col="lightblue",cex=0.6)

# importance plots

imp.var(rpartFit2$finalModel)

#
best.control <- rpartFit2$finalModel$control
best.tree.reduced <- rpart(
  Win~. , 
  data = train.data,
  control = best.control, 
  parms = list(split = "Gini")
) 

save(best.tree.reduced, file = "best.tree.reduced.RData")

prp(best.tree, type=1, extra=1, split.box.col="lightblue",cex=0.6)

# apply best.tree on test.data

best.tree.pred <- predict(best.tree, newdata = test.data, type = "class")
err.tree.reduced <- mean(best.tree.pred != test.data$Win)
save(err.tree.reduced, file = "err.tree.reduced.RData")

cm <- confusionMatrix(data = best.tree.pred, reference = test.data$Win)
draw_confusion_matrix(
  cm, 
  class1 = "Loss",
  class2 = "Win",
  title = "Confusion matrix - Best decision tree"
)

