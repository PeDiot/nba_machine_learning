# Create a data set using pca results for lda, qda, logistic regression

#
library(FactoMineR)
library(factoextra)
library(dplyr)
library(ggplot2)

#
setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/data_cleaning")

#
source("auxiliaire.R")

#
train.data <- read.csv("train.data.cleaned.csv")
test.data <- read.csv("test.data.cleaned.csv")


#
train.data <- train.data %>%
  mutate_each(
    funs(factor),
    c("Game", "Home", "Win", "TeamGroup", "OppGroup")
  )

test.data <- test.data %>%
  mutate_each(
    funs(factor),
    c("Game", "Home", "Win", "TeamGroup", "OppGroup")
  )

#
train.data$Type <- rep("Train", nrow(train.data))
test.data$Type <- rep("Test", nrow(test.data))

#
data <- rbind(train.data, test.data)
data$Type <- as.factor(data$Type)

data.pca <- data[, c(get_quanti_variables(data), "Type")]

# pca
ind.sup <- which(data$Type == "Test")

res.pca <- PCA(
  data.pca[, -ncol(data.pca)], 
  ind.sup = ind.sup,
  scale.unit = TRUE, 
  ncp = Inf, 
  graph = TRUE
  )

res.pca$var$coord %>%
  dim()

eig_val <- get_eigenvalue(res.pca)

ncp <- eig_val[which(eig_val[,1] > 1),] %>% 
  nrow()

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 20), ncp = ncp)

pca.coord.train <- res.pca$ind$coord[, 1:ncp] %>% 
  as.data.frame()

pca.coord.test<- res.pca$ind.sup$coord[, 1:ncp] %>% 
  as.data.frame()

# new database with pca coordinates
quali.train <- get_quali_variables(train.data)
train.data.pca <- cbind(
  train.data[, quali.train[-length(quali.train)]],
  pca.coord.train
)

quali.test <- get_quali_variables(test.data)
test.data.pca <- cbind(
  test.data[, quali.test[-length(quali.test)]],
  pca.coord.test
)

# Export de la base de données au format .csv
write.csv(train.data.pca, "train.data.pca.csv", row.names = F)
write.csv(test.data.pca, "test.data.pca.csv", row.names = F)



