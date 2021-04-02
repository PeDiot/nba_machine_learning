# Data cleaning

#
library(dplyr)
library(ggplot2)
library(ggpubr)
library(ggcorrplot)

#
theme_set(theme_minimal())

#
source("auxiliaire.R")

# import the data 
setwd("C:/Users/pemma/OneDrive - Université de Tours/Mécen/M1/S2/Data Mining/Projet/nba/data_cleaning")
nba.data <- read.csv("nba.games.stats.csv", sep = ",", header = T)


# recoding variables

nba.data <- nba.data[, -1] # remove the 'X' column

which(is.na(nba.data))

str(nba.data)

nba.data$Date <- as.Date(nba.data$Date)

nba.data <- nba.data %>%
  transform_cat_var(
    c("Game", "Team", "Opponent", "Home", "WINorLOSS")
  )
str(nba.data)

#
nba.data$Home <- ifelse(
  nba.data$Home == "Home",
  1,
  0
) %>% as.factor()

nba.data$Win <- ifelse(
  nba.data$WINorLOSS == "W",
  1,
  0
) 


# PIR (Performance Index Rating)

# PIR = (Points + Rebounds + Assists + Steals + Blocks + Fouls Drawn) - (Missed Field Goals + Missed Free Throws + Turnovers + Shots Rejected + Fouls Committed)

nba.data$PIR <- nba.data$TeamPoints + nba.data$TotalRebounds + nba.data$Assists + nba.data$Steals + nba.data$Blocks + nba.data$Opp.TotalFouls -
  ((nba.data$FieldGoalsAttempted - nba.data$FieldGoals) + (nba.data$FreeThrowsAttempted - nba.data$FreeThrows) +
  (nba.data$Turnovers) + nba.data$Opp.Blocks + nba.data$TotalFouls) 

nba.data$OppPIR <- nba.data$OpponentPoints + nba.data$Opp.TotalRebounds + nba.data$Opp.Assists + nba.data$Opp.Steals + nba.data$Opp.Blocks + nba.data$TotalFouls -
  ((nba.data$Opp.FieldGoalsAttempted - nba.data$Opp.FieldGoals) + (nba.data$Opp.FreeThrowsAttempted - nba.data$Opp.FreeThrows) +
     (nba.data$Opp.Turnovers) + nba.data$Blocks + nba.data$Opp.TotalFouls) 

nba.data %>%
  ggplot(aes(x = WINorLOSS, y = PIR, fill = WINorLOSS)) +
  geom_boxplot(fill = c("#69b3a2", "#404080"))


# PIR per team

nba.data %>%
  group_by(Team) %>%
  summarise(PIR = mean(PIR)) %>%
  ggplot(aes(x = Team, y = PIR, fill = PIR)) +
  geom_bar(stat = "identity") +
  scale_fill_continuous(low="lightblue", high="darkblue") +
  ggtitle("Average PIR per Team") +
  ylab(" ")

# group team depending on PIR

PIR.avg <- nba.data %>%
  group_by(Team) %>%
  summarise(PIR = mean(PIR))

PIR.q <- quantile(PIR.avg$PIR)

for (team in levels(as.factor(PIR.avg$Team))){
  if (PIR.avg[PIR.avg$Team == team, "PIR"] < PIR.q[2]){
    nba.data[nba.data$Team == team, "TeamGroup"] <- 1
  }
  else{
    if (PIR.avg[PIR.avg$Team == team, "PIR"] < PIR.q[3]){
      nba.data[nba.data$Team == team, "TeamGroup"] <- 2
    }
    else{
      if (PIR.avg[PIR.avg$Team == team, "PIR"] < PIR.q[4]){
        nba.data[nba.data$Team == team, "TeamGroup"] <- 3
      }
      else{
        nba.data[nba.data$Team == team, "TeamGroup"] <- 4
      }
    }
  }
}

# opponent group depending on PIR

OppPIR.avg <- nba.data %>%
  group_by(Opponent) %>%
  summarise(OppPIR = mean(OppPIR))

OppPIR.q <- quantile(OppPIR.avg$OppPIR)

for (opp in levels(as.factor(OppPIR.avg$Opponent))){
  if (OppPIR.avg[OppPIR.avg$Opponent == opp, "OppPIR"] < OppPIR.q[2]){
    nba.data[nba.data$Opponent == opp, "OppGroup"] <- 1
  }
  else{
    if (OppPIR.avg[OppPIR.avg$Opponent == opp, "OppPIR"] < OppPIR.q[3]){
      nba.data[nba.data$Opponent == opp, "OppGroup"] <- 2
    }
    else{
      if (OppPIR.avg[OppPIR.avg$Opponent == opp, "OppPIR"] < OppPIR.q[4]){
        nba.data[nba.data$Opponent == opp, "OppGroup"] <- 3
      }
      else{
        nba.data[nba.data$Opponent == opp, "OppGroup"] <- 4
      }
    }
  }
}


nba.data$Win <- nba.data$Win %>% as.factor()


# with variables like ratio achieved / attempted
nba.data.cleaned <- nba.data[
  , 
  c(3, 2, 1, 5, 4, 41, 11, 14, 17:24, 27, 30, 33:40, 44:45)
]

# correlations between quantitative variables

nba.quanti <- nba.data.cleaned[, get_quanti_variables(nba.data.cleaned)]

corr <- round(cor(nba.quanti), 3)
p.mat <- cor_pmat(nba.quanti)

ggcorrplot(corr, hc.order = TRUE, type = "lower",
           outline.col = "white",
           colors = c("#6D9EC1", "white", "#E46726"),
           lab = F, tl.cex = 10, p.mat = p.mat,
           title = "Correlation matix") 

# with PIR and OppPIR
nba.data.reduced <- nba.data[
  , 
  c(3, 2, 1, 5, 4, 41:45)
  ]


# save the modified data sets
write.csv(nba.data.reduced, "nba.data.reduced.csv", row.names = F)
write.csv(nba.data.cleaned, "nba.data.cleaned.csv", row.names = F)

# train / test split

dates <- nba.data$Date %>%
  unique()

'%!in%' <- function(x,y)!('%in%'(x,y))

set.seed(1)
train.date <- sample(dates, 0.7 * length(dates))

train.data.cleaned <- nba.data.cleaned[nba.data.cleaned$Date %in% train.date, ]
test.data.cleaned <- nba.data.cleaned[nba.data.cleaned$Date %!in% train.date, ]

train.data.reduced <- nba.data.reduced[nba.data.reduced$Date %in% train.date, ]
test.data.reduced <- nba.data.reduced[nba.data.reduced$Date %!in% train.date, ]

# save the train and test data sets
data_names <- c(
  "train.data.cleaned", 
  "test.data.cleaned",
  "train.data.reduced",
  "test.data.reduced"
  )


for(i in 1:length(data_names)) { 
  write.csv(
    get(data_names[i]),  
    paste0(data_names[i], ".csv"),
    row.names = FALSE
    )
}

