# Linear Discriminant Analysis (LDA) - geometric approach

#
library(dplyr)
library(ggplot2)

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
X.train <- train.data[, get_quanti_variables(train.data)] %>%
  as.matrix()
win.train <- train.data[, 6]


# mean points of each group
gL <- colMeans(X.train[win.train == "0",])
gW <- colMeans(X.train[win.train == "1",])


# covariance matrix of the predictors (uncorrected)
n <- nrow(X.train)
V <- (n-1)/n*cov(X.train)

# intra-group covariance matrix (uncorrected)

cov_intra <- function(X, N){
  
  # Inputs: matrix of predictors (X), nb of obs (N)
  # Output: intra-group covariance matrix (W)
  
  # list of intra-group cov matrix for each group weighted by frequency
  Vw <- list()
  for (i in 1:length(X)){
    n <- nrow(X[[i]])
    Vw[[i]] <- (n-1)*cov(X[[i]])
  }
  
  W <- matrix(0, nrow(Vw[[1]]), ncol(Vw[[1]]))
  for (v in Vw){
    W <- (W + v)
  }
  W <- W/N
  
  return(W)
}

W <- cov_intra(
  X = list(X.train[win.train == "0",], X.train[win.train == "1",]), 
  N = n
  )
W


# inter-group cov matrix (uncorrected)

# with variance decomposition
B <- V - W

# with the mean points
nL <- nrow(X.train[win.train == "0",])
nW <- nrow(X.train[win.train == "1",])
B <- (nL*gL%*%t(gL) + nW*gW%*%t(gW))/n

# check
B + W - V

# get the linear discriminant factors

# get the corrected cov matrices, dividing by n - 2 (2 groups)

W_corr <- n/(n-2)*W
B_corr <- n/(n-2)*B

# get the metrics
M <- solve(W)

# eigen vectors

get_eig_vect <- function(M, W_corr, B_corr, K, P){
  
  # retrieves the coordinates of the eigenvectors of the matrix M^(-1)*B
  # Inputs: metrics, corr cov matrix, nb of groups K, nb of predictors P
  # Ouputs: coordinates of the eigen vectors
  
  eig_vect <- c(NULL)
  
  for (k in 1:K-1){
    v <- eigen(M%*%B_corr)$vectors[,k] 
    u <- c(NULL)
    for (p in 1:P){
      u <- c(u, 
             v[p]/sqrt(t(v)%*%W_corr%*%v)
      )
    }
    eig_vect <- c(eig_vect, u)
  }
  return(
    matrix(eig_vect, nrow = P, ncol = K-1) 
  )
}

eig_vect_list <- get_eig_vect(
  M, 
  W_corr, B_corr, 
  K = nlevels(win.train),
  P = ncol(X.train)
  )

# individuals' coordinates on the two linear discriminant axes

get_factor_coord <- function(eig_vect, X){
  
  fact_coord <- c(NULL)
  
  for (col in ncol(eig_vect)){
    fact_coord <- c(
      fact_coord,
      X%*%eig_vect[,col]
    )
  }
  
  return(
    matrix(as.numeric(fact_coord), nrow = nrow(X), ncol = ncol(eig_vect))
  )
  
}

factor_coord <- get_factor_coord(eig_vect_list, X.train)

#
df.lda <- cbind(factor_coord, win.train) %>%
  as.data.frame()

df.lda$win.train <- ifelse(
  df.lda$win.train == 1,
  "0",
  "1"
) %>% 
  as.factor()

colnames(df.lda) <- c("F1", "Win")

df.lda %>%
  head()

#
plot_lda_geom <- df.lda %>%
  ggplot(
    aes(x = F1, fill = Win)
    ) +
    geom_density(alpha=.5) +
    scale_fill_manual(values=c("#69b3a2", "#404080")) +
    labs(
      title = "The two groups projected onto the first linear discriminant axis",
      subtitle = "Training set"
      ) +
    xlab("Axis 1") +
    ylab("Density") +
    theme(plot.title = element_text(size = 12, face="bold"))

save(plot_lda_geom, file = "plot_lda_geom.RData")

# Predict: Geometric assignment rule 

# 
test.data <- read.csv("test.data.pca.csv", sep = ",", header = T)

#
test.data <- test.data %>%
  transform_cat_var(
    c("Game", "Team", "Opponent", "Home", "Win", "TeamGroup", "OppGroup")
  )

str(test.data)

# 
X.test <- test.data[, get_quanti_variables(test.data)] %>%
  as.matrix()
win.test <- test.data[, 6]

CoefL <- M%*%gL
CoefW <- M%*%gW

CteL <- -t(gL)%*%M%*%gL/2
CteW <- -t(gW)%*%M%*%gW/2

# sorting function for the two groups

FL <- function(x)x%*%CoefL+CteL
FW <- function(x)x%*%CoefW+CteW

FCL <- apply(X.test, MARGIN=1, FUN=FL)
FCW <- apply(X.test, MARGIN=1, FUN=FW)

win.pred.geom <- rep("0", nrow(X.test))
win.pred.geom <- ifelse(
  FCW > FCL, 
  "1", 
  "0"
) %>%
  as.factor()

err.lda.geom <- sum(win.test != win.pred.geom) / nrow(test.data)
err.lda.geom

save(err.lda.geom, file = "err.lda.geom.RData")

cm.lda <- confusionMatrix(data = win.pred.geom, reference = win.test)
draw_confusion_matrix(
  cm.lda, 
  class1 = "Loss", class2 = "Win",
  title = "Confusion matrix - LDA geometric approach")
