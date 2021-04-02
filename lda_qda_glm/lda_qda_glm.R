# Supervised classification on the new data set reduced by PCA 

#
library(dplyr)
library(ISLR)
library(ggplot2)
library(pROC)
library(class)
library(caret)
library(rsample)
library(MASS)

#
theme_set(theme_minimal())

#
setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/lda_qda_glm")

#
source("auxiliaire.R")
train.data <- read.csv("train.data.pca.csv", sep = ",", header = T)

#
train.data <- train.data %>%
  transform_cat_var(
    c("Game", "Team", "Opponent", "Home", "Win", "TeamGroup", "OppGroup")
  )

#
set.seed(1)
data_split <- train.data[, c("Win", get_quanti_variables(train.data))] %>% 
  initial_split(prop = 2 / 3)
test_data <- data_split %>% testing()
train_data <- data_split %>% training()


# First model: LDA

lda.mod <- lda(Win ~., data = train_data)
lda.mod

res.lda <- predict(lda.mod, newdata = test_data)

win.pred.lda <- res.lda$class

err.lda <- sum(test_data$Win != win.pred.lda) / nrow(test_data)
err.lda

save(err.lda, file = "err.lda.RData")

# Second model: QDA

qda.mod <- qda(Win ~., data = train_data)

res.qda <- predict(qda.mod, newdata = test_data)

win.pred.qda <- res.qda$class

err.qda <- sum(test_data$Win != win.pred.qda) / nrow(test_data)
err.qda


# Third model: Logistic regression 

glm.mod <- glm(
  Win ~., 
  data = train_data,
  family = binomial
  )

glm.probs <- predict(
  glm.mod,
  newdata = test_data,
  type = "response"
  )

win.pred.glm <- ifelse(glm.probs > 0.5, "1", "0") %>%
  as.factor()

err.glm <- sum(test_data$Win != win.pred.glm) / nrow(test_data)
err.glm

# prediction error on the test part of the training set 

glob.err <- c(
    err.lda,
    err.qda,
    err.glm
    )

# ROC curves

tab.lda <- roc(test_data$Win, res.lda$posterior[,2])
tab.qda <- roc(test_data$Win, res.qda$posterior[,2])
tab.glm <- roc(test_data$Win, glm.probs)

roc.lda <- ggroc(
  list(
    lda = tab.lda,
    qda = tab.qda,
    glm = tab.glm
    ),
  size = .7
) +
  geom_abline(
    intercept=1, 
    slope=1, 
    color="red",
    linetype="dashed") +
  labs(
    title = "ROC curves",
    subtitle = "Without cross-validation") +
  theme(
    plot.title = element_text(size = 12, face="bold"),
    legend.title = element_blank()
    )

save(roc.lda, file = "roc.lda.RData")

# Using `caret`

set.seed(1)
ctrlCv <- trainControl(method = "repeatedcv", repeats = 2, number = 7)
performance_metric <- "Accuracy"

ldaCv <- train(
  Win~ .,
  data = train.data[, c("Win", get_quanti_variables(train.data))],
  method = "lda",
  metric = performance_metric,
  trControl = ctrlCv 
  )


qdaCv <- train(
  Win~ .,
  data = train.data[, c("Win", get_quanti_variables(train.data))],
  method = "qda",
  metric = performance_metric,
  trControl = ctrlCv 
)

glmCv <- train(
  Win~ .,
  data = train.data[, c("Win", get_quanti_variables(train.data))],
  method = "glm",
  metric = performance_metric,
  trControl = ctrlCv 
)

results <- resamples(
  list(LDA = ldaCv,  QDA = qdaCv, GLM = glmCv)
  )

accuracy_plot <- ggplot(results) +
  ylab("Accuracy") +
  labs(
    title = "Dispersion of accuracy for the 3 models",
    subtitle = "With 5-fold cross-validation") +
  theme(plot.title = element_text(face="bold", size = 12))

save(accuracy_plot, file = "accuracy_plot.RData")


err.ldaCv <- 1 - ldaCv$results$Accuracy
err.qdaCv <- 1 - qdaCv$results$Accuracy
err.glmCv <- 1 - glmCv$results$Accuracy

glob.err.cv <-  c(
    err.ldaCv,
    err.qdaCv,
    err.glmCv
  )

glob.err.tab <- round(rbind(glob.err, glob.err.cv) * 100, 2)
rownames(glob.err.tab) <- c("Without cv", "With 5-fold cv")
colnames(glob.err.tab) <- c("LDA", "QDA", "GLM")
save(glob.err.tab, file = "glob.err.tab.RData")


# GLM seems to be the best fit !

#
glm.best <- glmCv$finalModel
save(glm.best, file = "glm.best.RData")


# Apply the cross-validated GLM on the test data

# 
test.data <- read.csv("test.data.pca.csv", sep = ",", header = T)

#
test.data <- test.data %>%
  transform_cat_var(
    c("Game", "Team", "Opponent", "Home", "Win", "TeamGroup", "OppGroup")
  )

#
glm.probs <- predict(
  glm.best,
  test.data[, c("Win", get_quanti_variables(test.data))],
  type = "response"
)

win.pred.glm <- ifelse(glm.probs > .5, "1", "0") %>%
  as.factor()

# global error
err.glm <- sum(win.pred.glm != test.data$Win) / nrow(test.data)
err.glm

save(err.glm, file = "err.glm.RData")

# confusion matrix
cm.glm <- confusionMatrix(data = win.pred.glm, reference = test.data$Win)
draw_confusion_matrix(
  cm.glm, 
  class1 = "Loss",
  class2 = "Win",
  title = "Confusion matrix - Best logistic regression after 5-fold cross-validation"
  )

save(cm.glm, file = "cm.glm.RData")
