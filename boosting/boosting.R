# Boosting using caret

# Packages
library(rpart)
library(dplyr)
library(tidyr)
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
library(ada)

theme_set(theme_minimal())

##############################################################################################

# Using the complete data set

setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/boosting")
source("auxiliaire.R")
train.data <- read.csv("train.data.cleaned.csv")
test.data <- read.csv("test.data.cleaned.csv")

#
train.data <- train.data[, -c(1:4)]
test.data <- test.data[, -c(1:4)]

#
train.data <- transform_cat_var(
  train.data,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)

test.data <- transform_cat_var(
  test.data,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)


# Boosting
boost <- ada(
  Win ~ .,
  data = train.data,
  type = "discrete", 
  loss = "exponential", # exp(model coef)
  control = rpart.control(cp = 0),
  iter = 500, # nb of models
  nu = 1 # shrinkage parameter or learning rate (lambda * alpha = 1 * alpha= alpha)
)

save(boost, file = "boost.RData")

load("boost.RData")
boost
summary(boost)
plot(boost)

boost$model$errs
  
# stumps
boostump <- ada(
  Win ~ .,
  data = train.data,
  type = "discrete", 
  loss = "exponential", 
  control = rpart.control(maxdepth = 1, cp = -1, minsplit = 0, xval = 0),  # xval = nb of cv
  iter = 500, 
  nu = 1 
)

load("boostump.RData")
boostump
summary(boostump)
plot(boostump)

boostump$model$errs

save(boostump, file = "boostump.RData")

# penalisation

boostpen01 <- ada(
  Win ~ .,
  data = train.data,
  type = "discrete", 
  loss = "exponential",
  control = rpart.control(maxdepth = 1, cp = -1, minsplit = 0, xval = 0), 
  iter = 500, 
  nu = .01 
)

load("boostpen01.RData")

save(boostpen01, file = "boostpen01.RData")

boostpen001 <- ada(
  Win ~ .,
  data = train.data,
  type = "discrete", 
  loss = "exponential", 
  control = rpart.control(maxdepth = 1, cp = -1, minsplit = 0, xval = 0),  
  iter = 500, 
  nu = .001
)

load("boostpen001.RData")

save(boostpen001, file = "boostpen001.RData")

#
niter <- 500
boos.err.plt <- 1:niter %>% as_tibble() %>% 
  rename("iter" = value) %>% 
  mutate(boost1 = boost$model$errs[1:niter, c("train.err")],
         stump = boostump$model$errs[1:niter, c("train.err")],
         pen01 = boostpen01$model$errs[1:niter, c("train.err")],
         pen001 = boostpen001$model$errs[1:niter, c("train.err")]) %>%
  pivot_longer(cols = 2:5, names_to = "model", values_to = "error") %>%
  ggplot() + aes(x = iter, y = error, color = model) + geom_line() +
  labs(
    title = "Different boosting models", 
    subtitle = "Train error",
    x = "Iterations", 
    y = "Error") +
  theme(plot.title = element_text(size = 12, face = "bold"))
  lims(y = c(0,0.4))

save(boos.err.plt, file = "boos.err.plt.RData")

# optimisation of boosting params
registerDoParallel(cores = 3)
ctrlCv <- trainControl(method = "repeatedcv", repeats = 2, number = 5)
adaGrid <- expand.grid(
  maxdepth = 1:5,
  iter = c(50, 100, 200),
  nu = c(1, 0.1, 0.01)
)
system.time(
  caretada <- train(
    Win ~ .,
    data = train.data,
    method = "ada",
    trControl = ctrlCv, 
    tuneGrid = adaGrid
  )
) 
stopImplicitCluster()

caretada
caretada$results
caretada$results

best.ada.control <- caretada$bestTune

save(caretada, file = "caretada.RData")

load("caretada.RData")

caretada$finalModel
caretada$bestTune
caretada$results


# apply the tuned results to the data 

best_boost_iter <- caretada$bestTune$iter
best_boost_pen <- caretada$bestTune$nu
best_boost_depth <- caretada$bestTune$maxdepth

best_boost <- ada(
  Win ~ .,
  data = train.data, 
  type = "discrete", 
  loss = "exponential",
  control = rpart.control(cp = -1, maxdepth = best_boost_depth), 
  iter = best_boost_iter, 
  nu = best_boost_pen,
  test.y = test.data[, 2], 
  test.x = test.data[, -2]
)

best_boost$model$errs

niter <- best_boost$iter
best_boost.plt <- 1:niter %>% as_tibble() %>% 
  rename("iter" = value) %>% 
  mutate(
    train = best_boost$model$errs[1:niter, "train.err"],
    test = best_boost$model$errs[1:niter, "test.err"]
    ) %>%
  pivot_longer(cols = 2:3, names_to = "model", values_to = "error") %>%
  ggplot() +
  aes(x = iter, y = error, color = model) + 
  geom_line() +
  labs(
    title = "Perfomance of the best boosting model",
    x = "Iterations", 
    y = "Error"
    ) +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    legend.title = element_blank()
    ) +
  lims(y = c(0.05,0.25))

save(best_boost.plt, file = "best_boost.plt.RData")

err.boost <- mean(
  predict(best_boost, newdata = test.data[, -2]) != test.data[, 2]
  )

save(err.boost, file = "err.boost.RData")

cm.boost <- confusionMatrix(
  data = predict(best_boost, newdata = test.data[, -2]),
  reference = test.data[, 2],
  positive = "1"
)

save(cm.boost, file = "cm.boost.RData")

######################################################################################################################

# Using the reduced data set 

train.data.reduced <- read.csv("train.data.reduced.csv")
test.data.reduced <- read.csv("test.data.reduced.csv")

#
train.data.reduced <- train.data.reduced[, -c(1:4)]
test.data.reduced <- test.data.reduced[, -c(1:4)]

#
train.data.reduced <- get_cat_var(
  train.data.reduced,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)

test.data.reduced <- get_cat_var(
  test.data.reduced,
  selected_var = c("Home", "Win", "TeamGroup", "OppGroup")
)


registerDoParallel(cores = 3)
ctrlCv <- trainControl(method = "repeatedcv", repeats = 2, number = 5)
adaGrid.reduced <- expand.grid(
  maxdepth = 1:5,
  iter = c(50, 100, 200), 
  nu = c(1, 0.1, 0.01)
)
system.time(
  caretada.reduced <- train(
    Win ~ .,
    data = train.data.reduced,
    method = "ada",
    trControl = ctrlCv, 
    tuneGrid = adaGrid.reduced
  )
) 
stopImplicitCluster()

caretada.reduced
caretada.reduced$bestTune
caretada.reduced$finalModel

save(caretada.reduced, file = "caretada.reduced.RData")

load("caretada.reduced.RData")
caretada.reduced$finalModel
