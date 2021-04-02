# Description.
# Fonctions utilisées dans le rapport "Machine Learning on NBA games results"

# packages
library(knitr)
library(kableExtra)


# tables' layout

if (knitr::is_latex_output()) {
  mykable <- function(tab, transp = FALSE, digits =2, titre=NULL, font_size = NULL,...){
    if( transp ){
      if(ncol(tab)<=6){
        tab %>% t() %>% kable(caption=titre, digits = digits, booktabs=TRUE,...) %>%
          kable_styling(full_width = F, position = "center", 
                        latex_options = c("striped", "condensed", "HOLD_position"),
                        font_size =  font_size)
      } else {
        tab %>% t() %>% kable(caption=titre, digits = digits, booktabs=TRUE,...) %>%
          kable_styling(full_width = F, position = "center", 
                        latex_options = c("striped", "condensed", "HOLD_position","scale_down"),
                        font_size =  font_size)
      }
      
    } else {
      if(ncol(tab)<=6){
        tab %>% kable(caption=titre, digits = digits, booktabs=TRUE,...) %>%
          kable_styling(full_width = F, position = "center", 
                        latex_options = c("striped", "condensed", "HOLD_position"),
                        font_size =  font_size)
      } else {
        tab %>% kable(caption=titre, digits = digits, booktabs=TRUE,...) %>%
          kable_styling(full_width = F, position = "center", 
                        latex_options = c("striped", "condensed", "HOLD_position","scale_down"),
                        font_size =  font_size)
      }
    }
  }
} else {
  mykable <- function(tab, transp = FALSE, digits = 2, titre=NULL, font_size = NULL, ...){
    if(transp){
      tab %>% t() %>% kable(caption=titre, digits = digits,...) %>%
        kable_styling(full_width = F, position = "center",
                      bootstrap_options = c("striped", "condensed"))  
    } else {
      tab %>% kable(caption=titre, digits = digits, ...) %>%
        kable_styling(full_width = F, position = "center",
                      bootstrap_options = c("striped", "condensed"))
    }
  }
}

# 
get_quali_variables <- function(df){
  
  quali <- c(NULL)
  
  nb_var <- length(names(df))
  
  for (var in 1:nb_var){
    if (class(df[, var]) != "integer" & class(df[, var]) != "numeric"){
      quali <- c(quali, names(df)[var])
    }
  }
  
  return(quali)
  
}

#
get_quanti_variables <- function(df){
  
  quanti <- c(NULL)
  
  nb_var <- length(names(df))
  
  for (var in 1:nb_var){
    if (class(df[, var]) == "integer" | class(df[, var]) == "numeric"){
      quanti <- c(quanti, names(df)[var])
    }
  }
  
  return(quanti)
  
}

# 
get_cat_var <- function(data, selected_var){
  
  data <- data %>%
    mutate_each(
      funs(factor),
      selected_var
    )
  
  return(data)
}

#
density_in_loop <- function(df, selected_col, title){
  
  var_list <- names(df)[selected_col]
  plot_list <- list()
  
  for (i in 2:length(var_list)) {
    
    p <- ggplot(
      df, 
      aes_string(
        x = var_list[[i-1]], fill = var_list[[length(var_list)]]
      )
    ) +
      geom_density(alpha=0.5, adjust=2) +
      scale_fill_manual(values=c("#69b3a2", "#404080")) +
      labs(fill=var_list[[length(var_list)]])
    plot_list[[i-1]] <- p
    
  }
  
  visu <- ggarrange(plotlist = plot_list) %>%
    annotate_figure(
      top = text_grob(title,
                      color = "black",
                      size = 12,
                      face = "bold")
    )
  
  return(visu)
  
}

# table for global error

glob.err.tab <- function(list.err, colnames, title){
  
  res <- NULL
  
  for (err in list.err){
    res <- c(
      res,
      err * 100
    )
  }
  
  tab <- as.matrix(
    res, 
    nrow = 1
    ) %>%
    t() %>%
    as.table()
  
  colnames(tab) <- colnames
  rownames(tab) <- "% error"
  
  visu <- tab %>%
    mykable(
      titre = title,
      digits = 3
    )
  
  return(visu)
}


# confusion matrix
draw_confusion_matrix <- function(cm, class1, class2, title) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(title, cex.main=1.5)
  # create the matrix 
  rect(150, 430, 240, 370, col="#69b3a2")
  text(195, 435, class1, cex=1.2)
  rect(250, 430, 340, 370, col="#404080")
  text(295, 435, class2, cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col="#404080")
  rect(250, 305, 340, 365, col="#69b3a2")
  text(140, 400, class1, cex=1.2, srt=90)
  text(140, 335, class2, cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "Details", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(50, 35, names(cm$overall[1]), cex=1.4, font=2)
  text(50, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
}  



# variable importance plots

imp.var <- function(obj){
  
  if (class(obj) == "rpart"){
    
    df.imp <- data.frame(imp = obj$variable.importance) %>% 
      tibble::rownames_to_column() %>% 
      rename("Variable" = rowname) %>% 
      arrange(imp) %>%
      mutate(Variable = forcats::fct_inorder(Variable))
    
    visu <- df.imp %>%
      ggplot() +
      geom_segment(aes(x = Variable, y = 0, xend = Variable, yend = imp), 
                   size = 1, alpha = 0.7, linetype = "dashed") +
      geom_point(aes(x = Variable, y = imp, col = Variable), 
                 size = 4, show.legend = F) +
      coord_flip() +
      xlab("") +
      ylab("Mean Decrease Gini") +
      ggtitle("Variable importance") +
      theme(plot.title = element_text(size=12, face = "bold"))
  }
  
  else {
    
    df.imp <- data.frame(obj$importance[,3:4]) %>% 
      tibble::rownames_to_column() %>% 
      rename("Variable" = rowname) %>% 
      arrange(MeanDecreaseAccuracy) %>%
      arrange(MeanDecreaseGini) %>%
      mutate(Variable = forcats::fct_inorder(Variable))
    
    p.mdg <- df.imp %>%
      ggplot() +
      geom_segment(aes(x = Variable, y = 0, xend = Variable, yend = MeanDecreaseGini), 
                   size = 1, alpha = 0.7, linetype = "dashed") +
      geom_point(aes(x = Variable, y = MeanDecreaseGini, col = Variable), 
                 size = 4, show.legend = F) +
      coord_flip() +
      xlab("") +
      ylab("") +
      ggtitle("Mean Decrease Gini") +
      theme(plot.title = element_text(size=10))
    
    p.mda <- df.imp %>%
      ggplot() +
      geom_segment(aes(x = Variable, y = 0, xend = Variable, yend = MeanDecreaseAccuracy), 
                   size = 1, alpha = 0.7, linetype = "dashed") +
      geom_point(aes(x = Variable, y = MeanDecreaseAccuracy, col = Variable), 
                 size = 4, show.legend = F) +
      coord_flip() +
      xlab("") +
      ylab("") +
      ggtitle("Mean Decrease Accuracy") +
      theme(plot.title = element_text(size=10))
    
    visu <- ggarrange(p.mdg, p.mda) %>%
      annotate_figure(
        text_grob("Variable importance plots", face = "bold")
      )
  }
  
  return(visu)
  
}
  

