# Decision trees using the complete data set "nba.data.cleaned.csv"

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

str(train.data)
str(test.data)


# Using the 'caret' package

#
rep <- 5
cv <- 10
ctrlCv <- trainControl(method = "repeatedcv", repeats = rep, number = cv)

system.time(
  rpartFit1 <- train(
    Win~., 
    data = train.data,
    method = "rpart",
    trControl = ctrlCv,
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

rpart_df <- rpartFit2$results[, 1:2] %>%
  as.data.frame()

depth_plot <- rpart_df %>%
  ggplot(
    aes(x = maxdepth, y = Accuracy)
  ) +
  geom_point(shape = 4) + 
  geom_line(col = "#BCB2BC", size = 1) +
  labs(
    title = "Accuracy depending on depth of tree",
    subtitle = "Training set"
  ) +
  theme(plot.title = element_text(size = 12, face = "bold"))

save(depth_plot, file = "depth_plot.RData")

#
rpartFit2$finalModel
prp(rpartFit2$finalModel, type=1, extra=1, split.box.col="lightblue",cex=0.6)

# importance plots

imp.var.tree <- imp.var(rpartFit2$finalModel)
save(imp.var.tree, file = "imp.var.tree.RData")

#
best.control <- rpartFit2$finalModel$control
best_tree <- rpart(
  Win~. , 
  data = train.data,
  control = best.control, 
  parms = list(split = "Gini")
) 

save(best_tree, file = "best_tree.RData")

prp(best_tree, type=1, extra=1, split.box.col="lightblue",cex=0.6)

# apply best.tree on test.data

best.tree.pred <- predict(best_tree, newdata = test.data, type = "class")
err.tree <- mean(best.tree.pred != test.data$Win) 
save(err.tree, file = "err.tree.RData")

confusion.matrix(test.data$Win, best.tree.pred, threshold=0.5)

cm <- confusionMatrix(data = best.tree.pred, reference = test.data$Win)
draw_confusion_matrix(
  cm, 
  class1 = "Loss",
  class2 = "Win",
  title = "Confusion matrix - Best decision tree"
)


# Train the models on the train.data data set 

set.seed(1)
data_split <- train.data[, -c(1:4)] %>% initial_split(prop = 2/3)
test_data <- data_split %>% testing()
train_data <- data_split %>% training()
train_data$Win <- train_data$Win %>%
  as.factor()
test_data$Win <- test_data$Win %>%
  as.factor()

# Initial decision tree

control.max <- rpart.control(cp = 0, max.depth = 0, minbucket = 1, minsplit = 1)
tree.max <- rpart(
  Win~. , 
  data = train_data,
  control = control.max, 
  parms = list(split = "information")
  ) 


# prp(tree.max,type=1,extra=1,split.box.col="lightblue",cex=0.6)

pred.tree.max <- predict(tree.max, newdata = test_data, type = "class")

# tree.max$splits

#
plotcp(tree.max)

tree.max$cp
# xerror = error on the observations from cross validation data

visu_tree_res(tree.max, xlim = c(0, 50), ylim = c(0, 100), title = "Maximal tree")

beta <- complexity(tree.max$cp)
beta

best.cp <- beta[11]

# pruned trees

#
tree.prune <- prune(tree.max, cp = best.cp)

prp(tree.prune, type=1, extra=1, split.box.col="lightblue", cex=0.6)
fancyRpartPlot(tree.prune) 


pred.tree.prune <- predict(tree.prune, newdata = test_data, type = "class")

plotcp(tree.prune)
tree.prune$cp
beta.prune <- complexity(tree.prune$cp)
beta.prune

best.cp.prune <- beta.prune[6]

visu_tree_res(tree.prune, xlim = c(0, 25), ylim = c(0, 40), title = "Pruned tree")

#
tree.prune.bis <- prune(tree.prune, cp = best.cp.prune)
prp(tree.prune.bis, type=1, extra=1, split.box.col="lightblue", cex=0.6)
tree.prune.bis %>%
  plotcp()

pred.tree.prune.bis <- predict(tree.prune.bis, newdata = test_data, type = "class")


# Prediction errors

err.tree.max <- mean(pred.tree.max != test_data$Win)
err.tree.prune <- mean(pred.tree.prune != test_data$Win) 
err.tree.prune.bis <- mean(pred.tree.prune.bis != test_data$Win) 

glob.err.tab(
  c(err.tree.max, err.tree.prune, err.tree.prune.bis),
  colnames = c("Maximal", "Pruned (13 splits)", "Pruned (11 splits)"),
  title = "Test global error - Decision trees"
)


# importance plots

df <- data.frame(imp = tree.prune$variable.importance) %>% 
  tibble::rownames_to_column() %>% 
  rename("variable" = rowname) %>% 
  arrange(imp) %>%
  mutate(variable = forcats::fct_inorder(variable))

df %>%
  ggplot() +
    geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
                 size = 1.5, alpha = 0.7) +
    geom_point(aes(x = variable, y = imp, col = variable), 
               size = 4, show.legend = F) +
    coord_flip() +
    ylab("Mean Decrease Gini") +
    ggtitle("Variable importance")


# calculate the confusion matrices
cm.max <- confusionMatrix(data = pred.tree.max, reference = test_data$Win)
draw_confusion_matrix(
  cm.max, 
  class1 = "Loss", class2 = "Win",
  title = "Confusion matrix - Maximal tree")

cm.prune <- confusionMatrix(data = pred.tree.prune, reference = test_data$Win)
draw_confusion_matrix(
  cm.prune, 
  class1 = "Loss", class2 = "Win",
  title = "Confusion matrix - Pruned tree 1")

# https://www.standardwisdom.com/2011/12/29/confusion-matrix-another-single-value-metric-kappa-statistic/

# Sensitivity = True Positive rate
# Specificity = True Negative Rate
