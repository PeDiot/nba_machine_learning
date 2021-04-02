# Exploratory Data Analysis: The NBA data set


# https://www.kaggle.com/ionaskel/nba-games-stats-from-2014-to-2018


# packages
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(ggpubr)
library(lattice)
library(caret)
library(FactoMineR)
library(factoextra)


# settings

theme_set(theme_minimal())

#
setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/nettoyage")
source("auxiliaire.R")


# import the data 

nba.data <- read.csv("nba.games.stats.csv", sep = ",", header = T)


# recoding variables

nba.data <- nba.data[, -1] # remove the 'X' column

which(is.na(nba.data))

str(nba.data)
nba.data$Date <- as.Date(nba.data$Date)

nba.data <- nba.data %>%
  mutate_each(
    funs(factor),
  c("Team", "Game", "Home", "Opponent", "WINorLOSS")
  )

str(nba.data)
summary(nba.data)

# quali variables

nba.quali.var <- get_quali_variables(nba.data)

nba.quali <- nba.data[, nba.quali.var]

# relationships between quali variables

ggcorrplot(cramer.matrix(nba.quali[, -3]),
           hc.order = TRUE, type = "lower",
           outline.col = "white",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"),
           lab = F, tl.cex = 10,
           title = "Relationships beteween qualitative variables")


# Win VS Loss

ggplot(
  nba.data, aes(
    x = WINorLOSS,
    y = prop.table(stat(count)),
    label = scales::percent(
      prop.table(stat(count))
      )
    )
  ) + 
  geom_bar(
    position = "dodge", 
    fill = c("#69b3a2", "#404080")
    ) + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 4) +
  scale_y_continuous(labels = scales::percent, limits = c(0, .6)) + 
  ylab("% count")


# Game

nba.data$Game.cut <- ifelse(
  as.numeric(nba.data$Game) < 22,
  "[1:21]",
  ifelse(
    as.numeric(nba.data$Game) < 42,
    "[22:41]",
    ifelse(
      as.numeric(nba.data$Game) < 62,
      "[42:61]",
      "[62:82]"
    )
  )
) %>%
  as.factor()


plot_list <- list()
for (i in 1:nlevels(nba.data$Game.cut)){
  p <- ggplot(
    nba.data[
      nba.data$Game.cut == levels(nba.data$Game.cut)[i],
      ], 
    aes(
      x = Game,
      fill = WINorLOSS
    )
  ) + 
    geom_bar(
      position = "dodge"
    ) + 
    geom_hline(yintercept=60, 
               linetype='dotted', 
               col = 'red') +
    scale_fill_manual(values=c("#69b3a2", "#404080")) +
    ylab("Count")
  plot_list[[i]] <- p
}

ggarrange(plotlist = plot_list) %>%
  annotate_figure(
    top = text_grob("Game VS WINorLOSS",
                    color = "black",
                    size = 12)
  )

# Away VS Home

ggplot(
  nba.data, aes(
    x = Home,
    y = prop.table(stat(count)),
    fill = WINorLOSS,
    label = scales::percent(
      prop.table(stat(count))
    )
  )
) + 
  geom_bar(
    position = "dodge"
  ) + 
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 4) +
  scale_y_continuous(labels = scales::percent, limits = c(0, .4)) + 
  ylab("% count")


# WINorLOSS vs Team

p1 <- ggplot(
  nba.data, aes(
    x = Team,
    fill = WINorLOSS
  )
) + 
  geom_bar(
    position = "dodge"
  ) + 
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  ylab("Count")

p1 +
  ggtitle("Win or Loss per team")

p1 + 
  ggtitle("Win or Loss per team for home and away games") +
  facet_grid(Home~.)


# Correlations between quantitative variables

nba.quanti.var <- get_quanti_variables(nba.data)

nba.quanti <- nba.data[, nba.quanti.var]

corr <- round(cor(nba.quanti), 3)

ggcorrplot(corr, hc.order = TRUE, type = "lower",
           outline.col = "white",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"),
           lab = F, tl.cex = 10,
           title = "Correlation matix") 


# histograms of numerical features depending on win or loss

make_hists_in_loop(
  nba.data, 
  c(11, 14, 17, 6), 
  "Histograms of continuous variables depending on WINorLOSS"
  )

make_hists_in_loop(
  nba.data, 
  c(7:10, 6), 
  "Histograms of discrete variables depending on WINorLOSS"
)

make_hists_in_loop(
  nba.data, 
  c(12:13, 15:16, 6), 
  "Histograms of discrete variables depending on WINorLOSS"
)

make_hists_in_loop(
  nba.data, 
  c(18:26, 6), 
  "Histograms of discrete variables depending on WINorLOSS"
)

make_hists_in_loop(
  nba.data, 
  c(28:29, 31:32, 6), 
  "Histograms of discrete variables depending on WINorLOSS"
)

make_hists_in_loop(
  nba.data, 
  c(34:40, 6), 
  "Histograms of discrete variables depending on WINorLOSS"
)


# pca on quanti variables

res.pca <- PCA(nba.quanti, scale.unit = TRUE, ncp = Inf, graph = TRUE)

eig_val <- get_eigenvalue(res.pca)

ncp <- eig_val[which(eig_val[,1] > 1),] %>% nrow()

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 20), ncp = ncp)

pca.coord <- get_pca_ind(res.pca)$coord[, 1:ncp] %>% 
  as.data.frame()

nba.data.cleaned <- cbind(
  nba.data[, nba.quali.var],
  pca.coord
  )

str(nba.data.cleaned)

write.csv(nba.data.cleaned, "nba.data.cleaned.csv", row.names = F)

