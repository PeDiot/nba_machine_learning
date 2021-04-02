# k-Nearest Neighbor & Ensemble methods 


# Packages

library(dplyr)
library(reshape2)
library(rsample)
library(class)
library(ggplot2)
library(e1071)
library(caret)
library(readr)
library(fastDummies)

theme_set(theme_minimal())

#
setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/ensemble_methods")

#
source("auxiliaire.R")
train.data <- read.csv("train.data.pca.csv", sep = ",", header = T)


#
train.data <- train.data %>%
  transform_cat_var(
    c("Game", "Team", "Opponent", "Home", "Win", "TeamGroup", "OppGroup")
  )

train.data <- train.data[, c("Win", get_quanti_variables(train.data))] 

# Train / test split

set.seed(1)
data_split <- train.data %>% initial_split(prop = 2/3)
test_data <- data_split %>% testing()
train_data <- data_split %>% training()

str(train_data)

# Tuning knn

# With cross-validation

cv <- 5
nrep <- 2
K <- 20

start_time <- Sys.time()

knn.tune.cv <- tune.knn(
  x = train_data[, get_quanti_variables(train_data)],
  y = train_data$Win,
  k = 1:K,
  tunecontrol = tune.control(
    sampling = "cross", 
    cross = cv,
    nrepeat = nrep
    )
)

end_time <- Sys.time()

end_time - start_time

summary(knn.tune.cv)
plot(knn.tune.cv)

k.best.cv <- knn.tune.cv$best.parameters$k
k.best.cv

# With bootstrap

nboot <- 10

start_time <- Sys.time()

knn.tune.boot <- tune.knn(
  x = train_data[, get_quanti_variables(train_data)],
  y = train_data$Win,
  k = 1:K,
  tunecontrol = tune.control(sampling = "boot", nboot = nboot)
)

end_time <- Sys.time()

end_time - start_time

summary(knn.tune.boot)
plot(knn.tune.boot)

k.best.boot <- knn.tune.boot$best.parameters$k
k.best.boot

#
knn.err.tab <- rbind(
  knn.tune.cv$performances[k.best.cv,],
  knn.tune.boot$performances[k.best.boot,]
)

rownames(knn.err.tab) <- c("5-fold cv", "Bootstrap (n=10)")
save(knn.err.tab, file = "knn.err.tab.RData")

#  visualizations

knn.df.perf <- cbind(
  knn.tune.cv$performances[, 1:2],
  knn.tune.boot$performances[, 2]
  ) %>%
  as.data.frame()

colnames(knn.df.perf)[2:3] <- c("5-fold cv", "Bootstrap (n=10)")

knn.df.perf <- knn.df.perf %>%
  melt(id = "k")

knn.plot <- knn.df.perf %>%
  as.data.frame() %>%
  ggplot(
    aes(x = k, y = value, colour = variable)
  ) +
  geom_point(shape = 4) +
  geom_line() +
  labs(
    title = "k-nearest neighbors",
    subtitle ="Training error"
    ) +
  ylab("error") +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    legend.title = element_blank()
    )

save(knn.plot, file = "knn.plot.RData") 


# Prediction on test data

#
test.data <- read.csv("test.data.pca.csv", sep = ",", header = T)
test.data <- test.data %>%
  transform_cat_var(
    c("Game", "Team", "Opponent", "Home", "Win", "TeamGroup", "OppGroup")
  )
test.data <- test.data[, c("Win", get_quanti_variables(test.data))]

#
knn.pred <- knn(
  train = train.data[, get_quanti_variables(train.data)],
  test = test.data[, get_quanti_variables(test.data)], 
  cl = train.data[, get_quali_variables(test.data)],
  k = k.best.cv
)

err.knn <- sum(knn.pred != test.data$Win) / nrow(test.data)
err.knn

save(err.knn, file = "err.knn.RData")
