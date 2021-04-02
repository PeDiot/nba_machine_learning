# Bagging and random forests using caret

# Packages
library(rpart)
library(dplyr)
library(rsample)
library(rpart.plot)
library(rattle)  # Fancy tree plot
library(kableExtra)
library(caret)
library(randomForest)
library(ggplot2)
library(ggpubr)
library(parallel)
library(doParallel)

theme_set(theme_minimal())

#
setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/bagging_rf")
source("auxiliaire.R")

################################################################################################

# using the complete data set 

train.data <- read.csv("train.data.cleaned.csv")
test.data <- read.csv("test.data.cleaned.csv")

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


# number of trees
set.seed(1)
rf <- randomForest(
  Win ~ .,
  data = train.data, 
  method = "class", 
  ntree = 1000,
  parms = list(split = "gini"),
  keep.forest = TRUE, 
  importance = TRUE
)

oob.rf.min <- rf$err.rate[, 1] %>%
  min()

err.rf.df <- c(
  seq(1:1000),
  rf$err.rate[, 1]
  ) %>%
  matrix(nrow = nrow(rf$err.rate), ncol = 2) %>%
  as.data.frame()

colnames(err.rf.df) <- c("ntrees", "OOB.err")

minerr <- min(rf$err.rate[, "OOB"])
best.ntrees <- which(abs(rf$err.rate[, "OOB"] - minerr) < 0.001) %>%
  min()
best.ntrees

err.rf.df %>%
  ggplot(
    aes(x = ntrees, y = OOB.err)
  ) +
    geom_line(size = 1, col = "#C6B3C5") +
    geom_vline(
      xintercept = best.ntrees, 
      linetype = "dashed",
      size =.5
      ) +
    annotate(
      "text",
      label = "Best number of trees",
      x = 310,
      y = 0.24,
      size = 4
      ) +
    xlab("Number of trees") +
    ylab("OOB error") +
    ggtitle("OOB error depending on number of trees") +
    theme(plot.title = element_text(face="bold", size = 12))
  
rf %>%
  imp.var()

# random forest with one predictor (forest of stumps)
rf.q1 <- randomForest(
  Win ~ .,
  data = train.data, 
  method = "class", 
  ntree = 1000,
  mtry = 1,
  parms = list(split = "gini"),
  keep.forest = TRUE, 
  importance = TRUE
)

rf.q1
rf.q1$importance
rf.q1 %>%
  imp.var()
rf.q1$votes

rf.q1 %>%
  getTree() 

oob.rf.q1.min <- rf.q1$err.rate[, 1] %>%
  min()

# bagging
nvar <- ncol(train.data) - 1
bag <- randomForest(
  Win ~ .,
  data = train.data, 
  method = "class", 
  ntree = 1000,
  mtry = nvar,
  parms = list(split = "gini"),
  keep.forest = TRUE, 
  importance = TRUE
)

oob.bag.min <- bag$err.rate[, 1] %>%
  min()

#
ctrlCv <- trainControl(method = "repeatedcv", repeats = 2, number = 5)
rangerGrid <- expand.grid(
  mtry = c(1, 2, 4, 8, 16, nvar),
  splitrule = "gini",
  min.node.size = c(5, 10, 50, 100)
  )
registerDoParallel(cores = 3)
system.time(
  rf.caret <- train(
    Win ~ .,
    data = train.data, 
    method = "ranger",
    num.trees = 250,
    trControl = ctrlCv, 
    tuneGrid = rangerGrid
  )
)
stopImplicitCluster()

save(rf.caret, file = "rf.caret.RData")
rf.caret
rf.caret$bestTune

oob.rf.caret.min <- rf.caret$finalModel$prediction.error
  
#################################################################################################

# using the reduced data set 

#
train.data.reduced <- read.csv("train.data.reduced.csv")
test.data.reduced  <- read.csv("test.data.reduced.csv")

#
train.data.reduced <- train.data.reduced [, -c(1:4)]
test.data.reduced  <- test.data.reduced [, -c(1:4)]

#
train.data.reduced <- transform_cat_var(
  train.data.reduced,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)

test.data.reduced <- transform_cat_var(
  test.data.reduced,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)

# random forest

set.seed(1)
rf.reduced <- randomForest(
  Win ~ .,
  data = train.data.reduced, 
  method = "class", 
  ntree = 1000,
  parms = list(split = "gini"),
  keep.forest = TRUE, 
  importance = TRUE
)

rf.reduced

save(rf.reduced, file = "rf.reduced.RData")
load("rf.reduced.RData")

oob.rf.red.min <- rf.reduced$err.rate[, 1] %>%
  min()

err.rf.reduced.df <- c(
  seq(1:1000),
  rf.reduced$err.rate
) %>%
  matrix(nrow = nrow(rf.reduced$err.rate), ncol = 4) %>%
  as.data.frame()

colnames(err.rf.reduced.df) <- c("ntrees", "OOB.err", "Loss.err", "Win.err")

minerr <- min(rf.reduced$err.rate[, "OOB"])
best.ntrees <- which(abs(rf.reduced$err.rate[, "OOB"] - minerr) < 0.0005) %>%
  min()
best.ntrees

best.ntrees.plot <- err.rf.reduced.df %>%
  ggplot() +
  geom_line(
    aes(
      x = ntrees,
      y = OOB.err,
      colour = "OOB"
      ),
    size = 1
    ) +
  geom_line(
    aes(
      x = ntrees,
      y = Loss.err,
      colour = "Loss"
    ),
    size = .5
  ) +
  geom_line(
    aes(
      x = ntrees,
      y = Win.err,
      colour = "Win"
    ),
    size = .5
  ) +
  geom_vline(
    xintercept = best.ntrees, 
    linetype = "dashed",
    size =.5
  ) +
  annotate(
    "text", 
    label =  paste("Optimal number of trees = ", best.ntrees),
    x = best.ntrees + 250,
    y = .125,
    size = 3
  ) +
  scale_y_continuous(limits = c(0.09, .13)) +
  xlab("Number of trees") +
  ylab("error") +
  labs(
    title = "OOB error and error by group depending on number of trees",
    subtitle = "Training reduced set"
    ) +
  theme(
    plot.title = element_text(face="bold", size = 10),
    plot.subtitle = element_text(size = 9),
    axis.title = element_text(size = 8),
    legend.title = element_blank()
    )

save(best.ntrees.plot, file = "best.ntrees.plot.RData")

# bagging
nvar <- ncol(train.data.reduced) - 1

bag.reduced <- randomForest(
  Win ~ .,
  data = train.data.reduced, 
  method = "class", 
  ntree = 1000,
  mtry = nvar,
  parms = list(split = "gini"),
  keep.forest = TRUE, 
  importance = TRUE
)

bag.reduced
bag.reduced %>%
  imp.var()

save(bag.reduced, file = "bag.reduced.RData")
load("bag.reduced.RData")
oob.bag.red.min <- bag.reduced$err.rate[, 1] %>%
  min()

# stumps
rf.q1.reduced <- randomForest(
  Win ~ .,
  data = train.data.reduced, 
  method = "class", 
  ntree = 1000,
  mtry = nvar,
  parms = list(split = "gini"),
  keep.forest = TRUE, 
  importance = TRUE
)

oob.rf.q1.red.min <- rf.q1.reduced$err.rate[, 1] %>%
  min()

# summary of error rates for different models

tab.err.rf <- c(
  oob.rf.min,
  oob.bag.min,
  oob.rf.q1.min,
  oob.rf.red.min,
  oob.bag.red.min,
  oob.rf.q1.red.min
) %>%
  matrix(ncol = 2)

colnames(tab.err.rf) <- c("Complete data set", "Reduced data set")
rownames(tab.err.rf) <- c("Random forest", "Bagging", "Forest of stumps")
tab.err.rf <- tab.err.rf * 100
save(tab.err.rf, file = "tab.err.rf.RData")

#
ctrlCv <- trainControl(method = "repeatedcv", repeats = 2, number = 5)
rangerGrid <- expand.grid(
  mtry = 1:nvar,
  splitrule = "gini",
  min.node.size = c(5, 10, 50, 100)
)
registerDoParallel(cores = 3)
system.time(
  rf.reduced.caret <- train(
    Win ~ .,
    data = train.data.reduced, 
    method = "ranger",
    num.trees = best.ntrees,
    trControl = ctrlCv, 
    tuneGrid = rangerGrid,
    importance = "permutation"
  )
)
stopImplicitCluster()

rf.reduced.caret
rf.reduced.caret$finalModel
varImp(rf.reduced.caret)

save(rf.reduced.caret, file = "rf.reduced.caret.RData")
load("rf.reduced.caret.RData")

err.rf <- mean(
  predict(rf.reduced.caret, newdata = test.data.reduced[, -2]) != test.data.reduced[, 2]
)
save(err.rf, file = "err.rf.RData")

#
rf.reduced.caret$finalModel %>%
  imp.var()
