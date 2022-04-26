# ---
#   title: "Individual Machine Learning Project"
# author: "Russell Matthew Bayes"
# date: "28/03/2022"
# output: html_document
# editor_options:
#   markdown:
#   wrap: 90
# ---
#
#
###### 1. INTRODUCTION #####
#
# This Rmarkdown is part of the final assignment of the Professional Data Science Certificate program, run by HarvardX on the EDx platform.
#
# This assignment demonstrates the various statistical methods and Machine Learning techniques that have been taught, through the modules of the professional certificate series, to an independent data set. In meeting the assigned project requirements, a topic has been selected that is both data-rich and has few publicly available write ups.
#
# The Machine Failure Prediction Data set from the Kaggle platform was chosen several reasons: it can be accessed and understood by a wide audience, and it facilitates the development of advanced data visualization techniques, and machine learning models.
#
# This report analysis in detail the Machine Failure data set in five steps:
#
#   Data preparation: Download and cleaning the data
#
# Data exploration & analysis: Basic data exploration and advanced data visualization
#
# Model development: Multiple distinct approaches for model engineering
#
# Model selection: Based on model performance KPIs: $AUC$, and $Kappa$
#
#   Conclusion: Final discussion/comments, including future work
#
# Please note that the project comprises of a PDF file, containing the report, and the respective R and RMD files. All omitted code in this report can be found in detail in the R and RMD files.
#
# 1.1 Goals & Challenges
#
# This capstone project has two main goals, described below:
#
#   Evidence the data science concepts, knowledge, and skills acquired throughout the Professional Certificate Program (HarvardX) through their application to the Machine Failures Prediction Data set.
#
# Train and validate a models that predicts Machine Failures and their classification, using two methods.
#
# Demonstrate competence in several data science methods by training and validating a model that predicts Machine Failures, and their classification(s).
#
# Two KPIs will be used to assess model performance, the $Kappa$ and the $AUC$. The goal of this project is to set a base line the for each of the two modelling algorithms for each of the two target column in the data set, and then test 4 different techniques for treating outliers in a data set, up-sampling, and the use of class weights on one of the two modelling algorithms $AUC$ will be used as the primary matrix used to measure the performance of two class modelling engine and $Kappa$ for the mutli-class classification engine. Once baselines have been reset, treatment of outliers will be applied, up-sampling and using class-weight will be applied to the data set and modelling algorithms, in an attempt to improve upon the original baselines, the models will be down selected based on being achieving the best score. Once the models have been down selected and tuned the models that achieved the highest scores for the respective primary metric will trained and tuned using and training & validation data set. Conclusions drawn regarding the success of these models will be based on the models used make prediction against the validation data set.
#
# 1.2 The Machine Failure Data set
#
# The Machine Failure data set was obtained in the Kaggle data base. The Machine Failure data set consists of a synthetic data set that reflects real predictive maintenance encountered in the industry to the best the publishers knowledge.
#
# The data represents several key process variables that have been generated during the manufacturing process. The data set also includes information regarding which products were manufactured, whether any process-related failures occurred, and how these failures were classified.
#
# 2. DATA ANALYSIS
#
# The following section details the data preparation and analysis process. Initially, a review of the available data sets was conducted, and an appropriate data set was selected. Once acquired, the data set was examined to inform an understanding of the information contained therein, and a hypothesis regarding which model would be the best fit for modelling was generated.
#
# Additionally, the data visualization of the Machine Failure data set is going to create insists into the data set which will act a foundation or the decision-making process.


#
##### INSTALL AND LOAD NECESSARY PACKAGES ######
#
# Note: this process could take a couple of minutes
#The list of packages to be loaded. I already have ggplot2 package in my R environment and I need to install rgl package. This code should work fine for both cases.
#
list.of.packages <- c("plyr","parallel","doParallel","foreach","xgboost","ranger","rpart","caret","MLmetrics","tidyverse","data.table","gridExtra","magrittr","tidyr","dplyr","mltools","klaR","ggplot2","pROC","class","caTools","tidyverse","rstatix","fastknn","zoo","splitTools","kit","kableExtra","dlookr","DT","ggpubr","patchwork","summarytools","png","imager")
#
#You should be able to simply reuse the following lines of code as is
#
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#
if(length(new.packages)) install.packages(new.packages,dependencies = T)
#
# By now we have installed the requisite packages. Time to load them .
#
lapply(list.of.packages,function(x){library(x,character.only = TRUE)})



##### 2.1 Acquire Data #####
#
# As explained previously, data for this project was downloaded in the Kaggle website - link to the Machine Failure dataset. The data set was replicated on GitHub, to allow all the project files to be saved in the same hosting server.
#
# The first section of the R script downloads the Machine Failure data set in .csv format from GitHub and loads it into a data frame of raw data. Data will only be downloaded if not present in the current R working directory, to save the script´s running time.



# Original Kaggle link with the Machine Failure dataset
# Kaggle_data_link <-
#     "https://www.kaggle.com/shivamb/machine-predictive-maintenance-classification"# Mirror Machine Failure dataset on gitHub
# gitHub_data_link <- "https://raw.githubusercontent.com/RMbayes/Edx_Individual_Capstone/main/predictive_maintenance.csv"
#  Download File Direct from Github #
MM <- read.csv("https://raw.githubusercontent.com/RMbayes/Edx_Individual_Capstone/main/predictive_maintenance.csv")

MM <- as.data.frame(MM)


####### Project Functions ########
#### User Difined Functions  ####
#** Modify_Data_for_target **
#
# Function removes one of the two target columns and copies data into new data frame
#
Modify_Data_for_target <- function(df, target){
  if (target == "target") { #copies the binary response column and removes failure Category column
    dft <- df %>% dplyr::select(.,-Product_ID,-Failure_Type) #ProductID is not used during this project, Categorical column Failure_Type removed
    dfm <<- dft
    return(dfm)

  }
  if (target == "Failure_Type"){#copies the failure Category   column and removes  binary response column
    dft <- df %>% dplyr::select(.,-Product_ID,-target)
    dft$target <- df$Failure_Type # Renames failure_type to target so processes can be standardised for the twp target columns
    dft <- dft %>% dplyr::select(.,-Failure_Type) # removes Failure_Type column as it is now named target
    dfm <<- dft
    return(dfm)

  }
}

f0.5 <- function(data, lev = NULL, model = NULL) {
  #Function is used to calculate the F0.5 Score the two-class classification binary target
  f0.5_val <- MLmetrics::FBeta_Score(y_pred = data$pred,
                                     y_true = data$obs,
                                     positive = lev[1])
  c(F0.5 = f0.5_val)
}

#
# Function for Calculating RSME
#
RMSE_XX <- function(true_ratings, predicted_ratings) {
  trueratings <-  as.numeric(true_ratings)
  predictedratings <-  as.numeric(predicted_ratings)
  sqrt(mean((trueratings - predictedratings) ^ 2))}

#
#Function takes a column from data frame calcuated the  upper & lower quantile ranges and remove the values from the data frame
#
RM_Outliers_Oringal <- function(x) {

  na.rm = TRUE
  qnt <- stats::quantile(x, probs=c(.25, .75), na.rm = na.rm)
  H <- 1.5 * stats::IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  return(y)
}
#
# functions takes a and removes modelling test cases that do have any impact or work against each other
#
VaildateGrid <- function(df){
  # Below are the cases that need to be removed for the grids with all the various modelling conditions
  df <- df %>% mutate(remove_C = case_when((Sampling =="up" & Class_Weights == "Applied") ~ "NA",
                                           (Cart_Methd  =="xgbTree" & Class_Weights == "Applied") ~ "NA",
                                           (Cart_Methd  =="treebag" & Class_Weights == "Applied") ~ "NA",
                                           TRUE ~ "Keep"))
  df <- df %>% filter(remove_C == "Keep") %>% dplyr::select(.,-remove_C) # Filters out case that are not need & remove the additional column that has been added
  return(df) # returns df with only the desired test cases
}

#
#Creates case weights
#
Create_Weights <- function(df){ ## Function to help create list of class weights
  dx<- as.data.frame(df) # takes original list of factors and copies it
  colnames(dx)<-c("join") # renames column for ease later on
  waits <- NULL # place holder for data
  wx <- 1/nrow(unique(dx)) # calculates the proportion of any single data point
  tmp <- NULL # place holder for data



  for (i in 1:nrow(unique(dx))){ # loops for the number of factors in supplied df
    w <- 1/table(dx)[i]*wx # calculates factors weight in the df
    lvl <- levels(df)[i] # captures the names of the factor level
    tmp <- cbind(lvl,w) # binds the bname and factor level weight together
    waits <- rbind(waits,tmp) #stored values for later
  }

  colnames(waits) <- c("join","Class Weights") # changes name for ease of join
  waits<- as.data.frame(waits)


  Out <- merge(x = dx, y = waits, by = "join", all = TRUE,sort = FALSE)
  Out <- dplyr::left_join(dx, waits, by="join")# does a left join of the supplied df and the calculated weight

  Out <- Out %>% dplyr::select(.,-join) %>% as.data.frame #drops the original factor level names

  Cl_Weights <- as.numeric(Out$`Class Weights`)
  return(Cl_Weights) # return list of weights
}

#
#Function below used to generate the results table for 2 class classifcation models
#
Binary_Results <- function(Target_Column,Method,Sampling,OutLiear_Treatment,test_data,Conf,pred,pred_Prob,Class_Weights,Models_Results_Binary){
  ## Function used to wrtie results data data.frame -- function created to reduce the need to repeated the same code over and over again.

  Context_data <- tibble(target = Target_Column, Method = Method,Sampling_Method = Sampling,OutLiear_Treatment = OutLiear_Treatment,Class_Weights = Class_Weights,
                         F0.5 = MLmetrics::FBeta_Score(test_data$target, pred, positive = "Fail", beta = 0.5), # F-Beta Calculations
                         F2 = MLmetrics::FBeta_Score(test_data$target, pred, positive = "Fail", beta = 2), # F-Beta Calculations
                         Prep_Fail_is_Fail = Conf$table[1:1] , #confustion Matrix - stored in two class results
                         Prep_Pass_is_Fail_False = Conf$table[2:2], #confustion Matrix - stored in two class results
                         Pred_Fail_is_Pass_False = Conf$table[1,2], #confustion Matrix - stored in two class results
                         Pred_Pass_as_Pass = Conf$table[2:2,2:2], #confustion Matrix - stored in two class results
                         RSME =  RMSE_XX(pred,test_data$target), # Calcualtes RMSE and add to data frame
                         Overall_Error = # Calculates model overall_error and stores in df
                           fastknn::classLoss(
                             actual = test_data$target,
                             predicted = pred,
                             eval.metric = "overall_error"),
                         AUC = # Calculates model Area Under Cureve and stores in df
                           fastknn::classLoss(
                             actual = test_data$target,
                             predicted = pred
                             , prob = pred_Prob,
                             eval.metric = "auc"
                           ),
                         Mean_Error =
                           fastknn::classLoss( # Calculates model mean_error and stores in df
                             actual = test_data$target,
                             predicted = pred,
                             eval.metric = "mean_error"
                           ),
                         LogLoss = fastknn::classLoss(actual = test_data$target,predicted = pred,prob = pred_Prob, eval.metric = "logloss"))
  # Calculates model mean_error and stores in df

  # shapes and combines results data from the confusion matrix

  M_Stats <- cbind(as.data.frame(t(Conf$byClass)),(as.data.frame(t(Conf$overall))))

  # column binds the confusion matrix measures with model context data and additional measures to help drive decisions

  res <- cbind(Context_data,M_Stats)
  return(res)
}

#
#Function below used to generate the results table for mutliclass classifcation models
#
Class_Results <- function(Target_Column,Method,Sampling,OutLiear_Treatment,test_data,Conf,pred,pred_Prob,Class_Weights){
  Context_data <- tibble(target = Target_Column, Method = Method, Sampling_Method = Sampling,OutLiear_Treatment = OutLiear_Treatment,Class_Weights=Class_Weights,
                         Overall_Error = # Calculates model overall_error and stores in df
                           classLoss(
                             actual = test_data$target,
                             predicted = pred,
                             eval.metric = "overall_error"),
                         AUC = # Calculates model Area Under Cureve and stores in df
                           classLoss(
                             actual = test_data$target,
                             predicted = pred
                             , prob = pred_Prob,
                             eval.metric = "auc"
                           ),
                         Mean_Error =
                           classLoss( # Calculates model mean_error and stores in df
                             actual = test_data$target,
                             predicted = pred,
                             eval.metric = "mean_error"
                           ),
                         LogLoss = classLoss(actual = test_data$target,predicted = pred,prob = pred_Prob, eval.metric = "logloss"))
  # Calculates model mean_error and stores in df

  # shapes and combines results for each class in the classification models from the confusion matrix
  M_Stats <- cbind(as.data.frame(Conf$byClass),tidyr::pivot_wider(as.data.frame(t(Conf$table)),names_from = Prediction,values_from = Freq))

  # creates a df of confusion matrix measures with model context data and additional measures to help drive decisions
  M_Stats <- M_Stats %>% mutate(target = Target_Column,
                                Cart_Method = Method,
                                Sampling_Method = Sampling,
                                OutLiear_Treatment = OutLiear_Treatment)

  #labels the rows of dataframe with the classification Class
  rownames(M_Stats) <- colnames(Conf[["table"]])

  #Globally Store measures and context for each classification class of the models
  #Models_Results_Class_by_Class <<- rbind(Models_Results_Class_by_Class,M_Stats)

  #place holder for measures and context for the overall model / model interation
  Overal <- cbind(Context_data,t(as.data.frame(Conf$overall)))
  return(Overal)
  #remove row names that are not needed
  rownames(Overal) <- NULL

  #Globally measures and context for the overall model / model interation

  #Models_Results_Class_Overall <<- rbind(Models_Results_Class_Overall,Overal)
}

#
# Function creates plots of df, and metric(column), displays how sampling, class weights, and various outliears treatement method impacted the metric
#
Create_Plots <- function(df, ColumnName) {
  require(ggplot2)
  ## Method by Method
  Metric <- c(ColumnName)
  Metric_Name <- c(Metric)
  #df_type <- names(df)
  df_type <- deparse(substitute(df))
  if (df_type == "Models_Results_Class_Overall") {
    plt_title <-
      paste("Distribution of",
            Metric_Name,
            "for all Multi-Class Classification Models")
    fname = "MRCO"
  }

  if (df_type == "Models_Results_Binary") {
    plt_title <-
      paste("Distribution of",
            Metric_Name,
            "for all 2-Class Classification Models")
    fname = "MRB"
  }

  plt2 <- ggplot2::ggplot(df)+ggplot2::aes(x = "",y = get(Metric),
                                           fill = Method) +
    geom_boxplot(shape = "circle") +
    scale_fill_brewer(palette = "Set1", direction = 1) +
    labs(
      y = Metric_Name,
      x = " ",
      title = plt_title,
      subtitle = "By Modelling Method"
    ) +
    ggthemes::theme_calc(base_size = 16) +
    theme(
      axis.text = element_text(size = 16),
      plot.background = element_rect(
        color  = 'white',
        size = 2,
        linetype = 'solid',
        fill = "white"
      ),
      legend.position = "none",
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(face = "italic", hjust = 0.5)
    ) +
    facet_wrap(vars(Method))


  ## Method by Method + Sampling
  plt3 <- ggplot2::ggplot(df) +
    ggplot2::aes(x = "",
                 y = get(Metric),
                 fill = Method) +
    geom_boxplot(shape = "circle") +
    scale_fill_brewer(palette = "Set1", direction = 1) +
    labs(y = Metric_Name, x = " ", title = "Distribution grouped by \n Modeling Method & Sampling Technique") +
    ggthemes::theme_calc(base_size = 16)  +
    theme(
      axis.text = element_text(size = 16),
      plot.background = element_rect(
        color  = 'white',
        size = 2,
        linetype = 'solid',
        fill = "white"
      ),
      legend.direction = "horizontal",
      legend.background = element_rect(fill = alpha("white", 0.1)),
      legend.position = "none",
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(face = "italic", hjust = 0.5)
    ) +
    facet_wrap(vars(Sampling_Method))

  ## Method by Method + OutLiear_Treatment
  plt4 <- ggplot2::ggplot(df) +
    ggplot2::aes(x = "",
                 y = get(Metric),
                 fill = Method) +
    geom_boxplot(shape = "circle") +
    scale_fill_brewer(palette = "Set1", direction = 1) +
    labs(y = Metric_Name, x = " ", title = "Distribution grouped by Modeling Method & Outliers Treatement Method") +
    ggthemes::theme_calc(base_size = 16) +
    theme(
      axis.text = element_text(size = 16),
      plot.background = element_rect(
        color  = 'white',
        size = 2,
        linetype = 'solid',
        fill = "white"
      ),
      legend.direction = "horizontal",
      legend.position = "bottom",
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(face = "italic", hjust = 0.5)
    ) +
    facet_grid(cols = vars(OutLiear_Treatment))

  ## Method by Method + Class_Weights
  plt5 <-
    ggplot2::ggplot(df) + ggplot2::aes(x = "",
                                       y = get(Metric),
                                       fill = Method) +
    geom_boxplot(shape = "circle") +
    scale_fill_brewer(palette = "Set1", direction = 1) +
    labs(y = Metric_Name, x = " ", title = "Distribution grouped by \n Modeling Method & Class Weights") +
    ggthemes::theme_calc(base_size = 16)  +
    theme(
      axis.text = element_text(size = 16),
      plot.background = element_rect(
        color  = 'white',
        size = 2,
        linetype = 'solid',
        fill = "white"
      ),
      legend.direction = "horizontal",
      legend.background = element_rect(fill = alpha("white", 0.1)),
      legend.position = "none",
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(face = "italic", hjust = 0.5)
    ) +
    facet_wrap(vars(Class_Weights))



  #bottom_row <- plot_grid(plt3, plt5)
  #  (plt2)/bottom_row/(plt4)+ theme(plot.background = element_rect(color  = 'white', size = 2,linetype = 'solid', fill ="white"))
  #
  plt2grob <- ggplotGrob(plt2)
  plt4grob <- ggplotGrob(plt4)
  plt_35_grob <-
    ggplotGrob(ggpubr::ggarrange(
      plt3,
      plt5,
      ncol = 2,
      nrow = 1,
      common.legend = FALSE,
      legend = "none"
    ))



  Plot <-
    grid.arrange(
      arrangeGrob(plt2grob, ncol = 1),
      # First row with one plot spaning over 2 columns
      arrangeGrob(plt_35_grob, ncol = 1),
      arrangeGrob(plt4grob, ncol = 1),
      nrow = 3
    )


  # file <- tempfile()
  # ggsave(filename = file, plot = Plot,width = 20, height = 20, units = "cm", device = "svg")
  #
  # ggsave("./mtcarsX2.pdf", width = 20, height = 20, units = "cm")
  # file <- tempfile()
  #Fname <- paste0(getwd(),"/",format(Sys.time(), "%d%M%Y%H%M%S"), sample(1:100000, 1),".pdf")
  #ggsave(filename = Fname ,plot = Plot,width = 45, height = 30,units = "cm",device = "pdf")

  Fname2 <-
    paste0(getwd(),
           "/",
           format(Sys.time(), "%d%M%Y%H%M%S"),
           sample(1:100000, 1),
           ".png")
  ggsave(
    filename = Fname2 ,
    plot = Plot,
    width = 40,
    height = 35,
    units = "cm",
    device = "png"
  )
  dev.off()
  return(Fname2)

}

#
# Function creates coloured \contidional table showing results above and below baselines (Two Class)
#
Make_2Class_Conditional_Table <- function(df,mth,tuning = "No_Tuning"){
  mth_df <- df %>% dplyr::filter(Method == mth) # Filters data for requested modeling method

  context <- mth_df %>% dplyr::select(.,target,Method,Sampling_Method,OutLiear_Treatment,Class_Weights,Tuning) # Splits context data for number rounding
  data <- mth_df %>% dplyr::select(.,AUC,`Balanced Accuracy`,F0.5,F1,F2,Kappa,LogLoss,Mean_Error,Overall_Error,RSME) %>% round(.,4)# round metric values to 4 decimal places

  mth_df <- cbind(context,data) # recombineds data and context

  # Filter data to get baseline values
  mth_baselines <- mth_df %>% filter(Sampling_Method == "No-Sampling" & OutLiear_Treatment =="No_Treatment" & Class_Weights == "No_Weights" & Tuning == "No_Tuning")
  pglen <- nrow(mth_df)
  #Removes Tuning Columns
  mth_df <- mth_df %>% dplyr::select(.,-Tuning)

  # Creates conditional formatted table
  tbl <- datatable(mth_df, rownames = FALSE,options = list(pageLength = pglen)) %>%
    formatStyle(columns = "AUC",background = styleInterval(c(mth_baselines$AUC), c("red","green"))) %>%
    formatStyle(columns = "AUC",background = styleEqual(c(mth_baselines$AUC), c("lightblue"))) %>%
    formatStyle(columns = "Balanced Accuracy",background = styleInterval(c(mth_baselines$`Balanced Accuracy`), c("red","green"))) %>%
    formatStyle(columns = "Balanced Accuracy",background = styleEqual(c(mth_baselines$`Balanced Accuracy`), c("lightblue"))) %>%
    formatStyle(columns = "F0.5", background = styleInterval(c(mth_baselines$F0.5), c("red","green"))) %>%
    formatStyle(columns = "F0.5", background = styleEqual(c(mth_baselines$F0.5), c("lightblue"))) %>%
    formatStyle(columns = "F1", background = styleInterval(c(mth_baselines$F1), c("red","green"))) %>%
    formatStyle(columns = "F1", background = styleEqual(c(mth_baselines$F1), c("lightblue"))) %>%
    formatStyle(columns = "F2", background = styleInterval(c(mth_baselines$F2), c("red","green"))) %>%
    formatStyle(columns = "F2", background = styleEqual(c(mth_baselines$F2), c("lightblue"))) %>%
    formatStyle(columns = "Kappa", background = styleInterval(c(mth_baselines$Kappa), c("red","green"))) %>%
    formatStyle(columns = "Kappa", background = styleEqual(c(mth_baselines$Kappa), c("lightblue"))) %>%
    formatStyle(columns = "LogLoss", background = styleInterval(c(mth_baselines$LogLoss), c("green","red"))) %>%
    formatStyle(columns = "LogLoss", background = styleEqual(c(mth_baselines$LogLoss), c("lightblue")))%>%
    formatStyle(columns = "Mean_Error", background = styleInterval(c(mth_baselines$Mean_Error), c("green","red"))) %>%
    formatStyle(columns = "Mean_Error", background = styleEqual(c(mth_baselines$Mean_Error), c("lightblue"))) %>%
    formatStyle(columns = "Overall_Error", background = styleInterval(c(mth_baselines$Overall_Error), c("green","red"))) %>%
    formatStyle(columns = "Overall_Error", background = styleEqual(c(mth_baselines$Overall_Error), c("lightblue")))%>%
    formatStyle(columns = "RSME", background = styleInterval(c(mth_baselines$RSME), c("green","red"))) %>%
    formatStyle(columns = "RSME", background = styleEqual(c(mth_baselines$RSME), c("lightblue")))

  return(tbl) # output table to viewer
}


#
# Function creates coloured \contidional table showing results above and below baselines (Multi-Class)
#
Make_NClass_Conditional_Table <- function(df,mth,tuning = "No_Tuning"){
  mth_df <- df %>% dplyr::filter(Method == mth)# Filters data for requested modeling method
  context <- mth_df %>% dplyr::select(.,target,Method,Sampling_Method,OutLiear_Treatment,Class_Weights,Tuning) # Seperated context from data for rounding
  data <- mth_df %>% dplyr::select(.,AUC,Kappa,LogLoss,Mean_Error,Overall_Error,Accuracy) %>% round(.,4) # rounds data to 4 digits

  mth_df <- cbind(context,data) # recombined context and rounded data

  mth_baselines <- mth_df  %>% filter(Sampling_Method == "No-Sampling" & OutLiear_Treatment =="No_Treatment" & Class_Weights == "No_Weights" & Tuning == tuning) # get data for baseline
  pglen <- nrow(mth_df)
  mth_df <- mth_df %>% dplyr::filter(Tuning == tuning)
  mth_df <- mth_df %>% dplyr::select(.,-Tuning)


  tbl <- datatable(mth_df, rownames = FALSE,options = list(pageLength = pglen)) %>%
    formatStyle(columns = "AUC",background = styleInterval(c(mth_baselines$AUC), c("red","green"))) %>%
    formatStyle(columns = "AUC",background = styleEqual(c(mth_baselines$AUC), c("lightblue"))) %>%
    formatStyle(columns = "Kappa", background = styleInterval(c(mth_baselines$Kappa), c("red","green"))) %>%
    formatStyle(columns = "Kappa", background = styleEqual(c(mth_baselines$Kappa), c("lightblue"))) %>%
    formatStyle(columns = "LogLoss", background = styleInterval(c(mth_baselines$LogLoss), c("green","red"))) %>%
    formatStyle(columns = "LogLoss", background = styleEqual(c(mth_baselines$LogLoss), c("lightblue")))%>%
    formatStyle(columns = "Mean_Error", background = styleInterval(c(mth_baselines$Mean_Error), c("green","red"))) %>%
    formatStyle(columns = "Mean_Error", background = styleEqual(c(mth_baselines$Mean_Error), c("lightblue"))) %>%
    formatStyle(columns = "Overall_Error", background = styleInterval(c(mth_baselines$Overall_Error), c("green","red"))) %>%
    formatStyle(columns = "Overall_Error", background = styleEqual(c(mth_baselines$Overall_Error), c("lightblue")))%>%
    formatStyle(columns = "Accuracy", background = styleInterval(c(mth_baselines$Accuracy), c("red","green"))) %>%
    formatStyle(columns = "Accuracy", background = styleEqual(c(mth_baselines$Accuracy), c("lightblue")))

  return(tbl)
}

#
#Function below used to generate all modelling data
#
Apply_Caret_Models <- function(ind,Grid, Tune = "No"){
  # Function that take a row of a test
  # modifies the dataset to be the 2 class classification problems or the Multi-Class Claffication problem
  # Applies the specificed sampling technique
  # Calculates and Applies the class weight
  # Specifies if tuning should be applied.
  # Calls the results table based on with the target column is 2Classes , or Mutli_Class



  Method <- Grid[ind,1] %>% as.character # Set varaible base on the row & column of supplied grid
  Target_Column <- Grid[ind,2] %>% as.character # Set varaible base on the row & column of supplied grid
  Sampling <- Grid[ind,3] %>% as.character # Set varaible base on the row & column of supplied grid
  OutLiear_Treatment <- Grid[ind,4] %>% as.character # Set varaible base on the row & column of supplied grid
  Class_Weights <- Grid[ind,5] %>% as.character # Set varaible base on the row & column of supplied grid



  ## Data Prep - Set data frame to be either 2class or multi_class ##
  Mod_data_func <- match.fun("Modify_Data_for_target")
  Mod_data_func(MM,Target_Column) # function output dfm (df)

  dfm <- dfm %>% dplyr::select(.,-UDI) # removes column that is not wanted
  col_order <- c("Type","Air_temperature_K","Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min","target") #set desire column order

  dfm <- dfm[, col_order] # Applies Order of columns
  colnames(dfm) <-c ("Type","Air_temp","Process_temp", "Rotational_speed", "Torque", "Tool_wear","target") #Renames columns as () in column names result in errors


  test_data <- dfm[Test_Index, ] #Creates test data  dataframe
  test_data$target<- as.factor(test_data$target)

  train_data <- dfm[Train_Index, ]#Creates test data  dataframe
  train_data$target<- as.factor(train_data$target)




  # Create list of column that should have outliears removed
  n_cols <- dfm %>% dplyr::select(.,-`Tool_wear`,-target) %>% dplyr::select(where(is.numeric)) %>% colnames()


  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_No_Groups"){
    train_data <- train_data %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)

  }

  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_Type"){
    train_data <- train_data %>%
      group_by(Type) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)
  }

  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_target"){
    train_data <- train_data %>%
      group_by(target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)

  }

  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_ALL"){
    train_data <- train_data %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)
    train_data <- train_data %>%
      group_by(Type) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
    train_data <- train_data %>%
      group_by(target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
    train_data <- train_data %>%
      group_by(Type,target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
  }


  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_Type_target"){
    train_data <- train_data %>%
      dplyr::group_by(Type,target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
  }

  #Checks grid input to see if data need to be modified for test case below
  if (Sampling == "up") {
    set.seed(12505)
    train_data <- caret::upSample(x = train_data[, -ncol(train_data)],y = train_data$target, yname = "target")
  }

  # Creates a set of  weights based on the dataframe being supplied
  # Case weights do not works when upsampling is applied
  Create_Weights <- match.fun("Create_Weights")
  Model_Weights = NULL
  if (Class_Weights == "Applied"){Model_Weights <- Create_Weights(train_data$target)}


  #Set the summary function and metric to be used depending on if the the supplied df
  # is for the 2class or MutliClass dataframe
  if (Target_Column == "target"){crtl_func <- prSummary
  mtx <- "AUC"
  }else{
    mtx <- "Kappa"
    crtl_func <- multiClassSummary}


  # Train Control settings
  number = 3
  repeats = 3
  set.seed(12505)
  control <- caret::trainControl(
    method = "repeatedcv",
    number = number ,
    search = "random",
    repeats = repeats,
    index = createResample(train_data$target, times = repeats *number),
    summaryFunction = crtl_func, #summaryFunction = multiClassSummary, # #
    allowParallel = TRUE,
    classProbs = TRUE,
  )

  TuneGrid = NULL


  ## Model Tuning - Set the tuning grids for each of the models used
  if (Tune == "Yes" && Method == "xgbTree"){TuneGrid <- expand.grid(nrounds = c(50,100,150),
                                                                    eta = c(0.03,0.04),
                                                                    max_depth = c(2,4,6),
                                                                    gamma = 0,
                                                                    colsample_bytree = c(0.8),
                                                                    min_child_weight = 1,
                                                                    subsample = c(0.75,1))}

  if (Tune == "Yes" && Method == "ranger"){TuneGrid <- expand.grid(mtry = 3:6,splitrule =c("gini"),min.node.size =c(3,5,7,8))}



  # Caret Machine Learnign Model
  set.seed(12505)
  model_list <- caret::train(
    target ~ .,
    data = train_data,
    trControl = control,
    weights = Model_Weights,
    tuneGrid = TuneGrid,
    metric = mtx, # metric = "Accuracy",
    method = Method # n My_models varaible Lists all models to train.
  )

  #Generate predictions for trained model
  set.seed(12505)
  pred <- predict(model_list, newdata = test_data, type =  "raw" )

  #generate the proberilities for each case
  set.seed(12505)
  pred_Prob <- predict(model_list, newdata = test_data, type =  "prob" ) %>% as.matrix

  #Creates and store confusion matrix
  Conf <- caret::confusionMatrix(pred,test_data$target)


  Target_Column <- Target_Column

  # Selected the correct results function depending on the trget column of dataframe being used
  if (Target_Column == "target"){df_target_func <- match.fun("Binary_Results")}
  if (Target_Column == "Failure_Type"){df_target_func <- match.fun("Class_Results")}

  #Call result fuction, and supplies the required variables for generate all the stats for the prblem class
  df_target_func(Target_Column,Method,Sampling,OutLiear_Treatment,test_data,Conf,pred,pred_Prob,Class_Weights)

}



Final_Caret_Models_Model_Output <- function(ind,Grid, Tune = "Yes"){

  # Function that take a row of a test
  # modifies the dataset to be the 2 class classification problems or the Multi-Class Claffication problem
  # Applies the specificed sampling technique
  # Calculates and Applies the class weight
  # Specifies if tuning should be applied.
  # Calls the results table based on with the target column is 2Classes , or Mutli_Class



  Method <- Grid[ind,1] %>% as.character # Set varaible base on the row & column of supplied grid
  Target_Column <- Grid[ind,2] %>% as.character # Set varaible base on the row & column of supplied grid
  Sampling <- Grid[ind,3] %>% as.character # Set varaible base on the row & column of supplied grid
  OutLiear_Treatment <- Grid[ind,4] %>% as.character # Set varaible base on the row & column of supplied grid
  Class_Weights <- Grid[ind,5] %>% as.character # Set varaible base on the row & column of supplied grid



  ## Data Prep - Set data frame to be either 2class or multi_class ##
  Mod_data_func <- match.fun("Modify_Data_for_target")
  Mod_data_func(MM,Target_Column) # function output dfm (df)

  dfm <- dfm %>% dplyr::select(.,-UDI) # removes column that is not wanted
  col_order <- c("Type","Air_temperature_K","Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min","target") #set desire column order

  dfm <- dfm[, col_order] # Applies Order of columns
  colnames(dfm) <-c ("Type","Air_temp","Process_temp", "Rotational_speed", "Torque", "Tool_wear","target") #Renames columns as () in column names result in errors


  test_data <- dfm[Val_Index, ] #Creates test data  dataframe
  test_data$target<- as.factor(test_data$target)

  train_data <- dfm[-Val_Index, ]#Creates test data  dataframe
  train_data$target<- as.factor(train_data$target)




  # Create list of column that should have outliears removed
  n_cols <- dfm %>% dplyr::select(.,-`Tool_wear`,-target) %>% dplyr::select(where(is.numeric)) %>% colnames()


  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_No_Groups"){
    train_data <- train_data %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)

  }

  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_Type"){
    train_data <- train_data %>%
      group_by(Type) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)
  }

  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_target"){
    train_data <- train_data %>%
      group_by(target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)

  }

  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_ALL"){
    train_data <- train_data %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% drop_na(.)
    train_data <- train_data %>%
      group_by(Type) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
    train_data <- train_data %>%
      group_by(target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
    train_data <- train_data %>%
      group_by(Type,target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
  }


  #Checks grid input to see if data need to be modified for test case below
  if(OutLiear_Treatment == "rm_OutLiear_Type_target"){
    train_data <- train_data %>%
      dplyr::group_by(Type,target) %>%
      mutate(dplyr::across(all_of(n_cols),RM_Outliers_Oringal)) %>% tidyr::drop_na(.)
  }

  #Checks grid input to see if data need to be modified for test case below
  if (Sampling == "up") {
    set.seed(12505)
    train_data <- caret::upSample(x = train_data[, -ncol(train_data)],y = train_data$target, yname = "target")
  }

  # Creates a set of  weights based on the dataframe being supplied
  # Case weights do not works when upsampling is applied
  Create_Weights <- match.fun("Create_Weights")
  Model_Weights = NULL
  if (Class_Weights == "Applied"){Model_Weights <- Create_Weights(train_data$target)}


  #Set the summary function and metric to be used depending on if the the supplied df
  # is for the 2class or MutliClass dataframe
  if (Target_Column == "target"){crtl_func <- prSummary
  mtx <- "AUC"
  }else{
    mtx <- "Kappa"
    crtl_func <- multiClassSummary}


  # Train Control settings
  number = 3
  repeats = 5
  set.seed(12505)
  control <- caret::trainControl(
    method = "repeatedcv",
    number = number ,
    repeats = repeats,
    index = createResample(train_data$target, times = repeats *number),
    summaryFunction = crtl_func, #summaryFunction = multiClassSummary, # #
    allowParallel = TRUE,
    classProbs = TRUE,
  )

  TuneGrid = NULL


  ## Model Tuning - Set the tuning grids for each of the models used
  if (Tune == "Yes" && Method == "xgbTree"){TuneGrid <- expand.grid(nrounds = c(500),
                                                                    eta = c(0.01,0.03,0.04,0.05),
                                                                    max_depth = c(2,4,6),
                                                                    gamma = 0,
                                                                    colsample_bytree = c(0.8),
                                                                    min_child_weight = 1,
                                                                    subsample = c(0.75,1))}

  if (Tune == "Yes" && Method == "ranger"){TuneGrid <- expand.grid(mtry = 3:6,splitrule =c("gini"),min.node.size =c(3,5,7,8))}



  # Caret Machine Learnign Model
  set.seed(12505)
  model_list <- caret::train(
    target ~ .,
    data = train_data,
    trControl = control,
    weights = Model_Weights,
    maximize = TRUE,
    tuneGrid = TuneGrid,
    metric = mtx, # metric = "Accuracy",
    method = Method # n My_models varaible Lists all models to train.
  )
  return(model_list)


}




##### 2.2 Data Recode #####
#
# The raw data stored in the Machine Failures data frame required transformation to fit the next phase of data exploration and visualization. The list of data transformation performed in the Machine Failures data frame is as follows:
#
#   Change column names: Make the Machine Failures data set more readable by removing spaces in column names.


df1 <- as.data.frame(ls(MM))
colnames(df1) <- c("Original Column Names")



colnames(MM) <- c("UDI","Product_ID" ,"Type" ,"Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm","Torque_Nm","Tool_wear_min","target","Failure_Type")

df2 <- as.data.frame(ls(MM))
colnames(df2) <- c("New Column Names")
df2 <- cbind(df1,df2)
df2 %>%
  kbl(caption = "Recoded Columns Names") %>%
  kable_classic(full_width = F, html_font = "Cambria")
rm(df1,df2)


Change Data Types: This is done to make sure the modelling algorithms know to to correctly handle and models the data that will be supplied to them.


# Original Data Types
df1 <- as.data.frame(sapply(MM, class))
colnames(df1) <- c("Original Data Type")



MM$target <- as.factor(MM$target)
MM$Failure_Type <- as.factor(MM$Failure_Type)
MM$Type <- as.factor(MM$Type)
MM$Product_ID  <- as.factor(MM$Product_ID )
# Data Types after change

df2 <- as.data.frame(sapply(MM, class))
colnames(df2) <- c("New Data Type")
df2 <- cbind(df1,df2)
df2 %>%
  kbl(caption = "Recoded Column Data Types") %>%
  kable_classic(full_width = F, html_font = "Cambria")
rm(df1,df2)

Categorical variables: Remove spaces in strings/characters columns. Below are the detail steps in R for the initial data transformation phase.



# Recode variable levels in Failure Type columns to make more user friendly
# # Original Data Types
df1 <- as.data.frame(levels(MM$Failure_Type))
colnames(df1) <- c("Original Factor Levels")


# Code ran twice to get to capture levels with double spaces
levels(MM$Failure_Type) <- sub(" ", ".", levels(MM$Failure_Type))
levels(MM$Failure_Type) <- sub(" ", ".", levels(MM$Failure_Type))

df2 <- as.data.frame(levels(MM$Failure_Type))
colnames(df2) <- c("New Factor Levels")
df2<-cbind(df1,df2)
df2 %>%
  kbl(caption = "Recoded Factor Levels For Failure Type") %>%
  kable_classic(full_width = F, html_font = "Cambria")
rm(df1,df2)

#####  2.3 Data Cleanse  #####
#
# Discrepancy in Data: Conduct a basic data check to ensure that the data follows a set of logical assumptions. It is worth noting that a minor discrepancy in the date was identified, which could affect the model's performance. Below is the summary count of target grouped by failure type. On inspection, it appears that the two class response ("target") column for the process failure has contradictory labeling for No.Failures with 9 rows in the target column stating that a failure had occurred but the Failure Type states No Failure this is in contradiction to the rest of the data set and will be removed.

### Check targets & Failure Modes

MM %>% dplyr::select(.,target,Failure_Type) %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n()) %>%
  kbl(caption = "Summary Count of Target group by Failure Type") %>% kable_classic(full_width = F, html_font = "Cambria")

MM %>% dplyr::select(.,target,Failure_Type) %>% filter(Failure_Type == 'No.Failure' |  Failure_Type == 'Random.Failures') %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n()) %>%
  kbl(caption = "Summary Count groups that logically seemed to be mislabelled") %>%
  kable_classic(full_width = F, html_font = "Cambria")

# Due to concerns about confusion and complications during the data modelling process, this logical discrepancy in the data was resolved to ensure the best possible outcome. After considering numerous treatment methods for dealing with discrepancy, removing the contradicting rows seemed the most appropriate option.
#
# For simplicity the two class response column factors have been recoded from 0 & 1 to Pass and fail.


#Note: No.Failure misclassified, No failures declared as target events'
#removal of data seems to be the best logical decision to make in this instance

# Remove rows labelled that have conflicting classifications
# Creates column for misclassification label
MM <- MM %>%  mutate(MissClassification = ifelse(target == "1" & Failure_Type == "No.Failure", "MissClass",
                                                 ifelse(target != "1" & Failure_Type != "No.Failure", "Pass",
                                                        ifelse(target == "0" ,"Pass", "Pass"))))


# remove misclassified rows by filter out the missclass label from newly create column
MM <- MM %>% dplyr::filter(MissClassification != "MissClass")
MM <- MM %>% dplyr::select(.,-MissClassification) # remove misclassification column


MM %>% dplyr::select(.,target,Failure_Type) %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n()) %>%  kbl(caption = "Summary Counts of target and Failure types are data correction") %>%
  kable_classic(full_width = F, html_font = "Cambria")

MM <- MM %>% mutate(target = case_when(target == "0" ~ "Pass", Failure_Type != "1" ~ "Fail"))



##### 2.4 Data Summary  #####
#
# The original data set consists of 10 000 data points stored as rows with 14 features in columns.
#
# The data set consists of 10 000 data points stored as rows with fourteen features in columns . UID: unique identifier ranging from 1 to 10000
#
# . productID: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
#
# . air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
#
# . process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
#
# . rotational speed [rpm]: calculated from power of 2860 W, overlaid with a normally distributed noise
#
# . torque [Nm]: torque values are normally distributed around 40 Nm with an Ïf = 10 Nm and no negative values.
#
# . tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
#
# . target: two class response column to indicate if a failure occurred during the process.
#
# . Failure Type: label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.


summarytools::dfSummary(MM, style = "grid", plain.ascii = FALSE,
                        varnumbers = FALSE, valid.col = FALSE, tmp.img.dir = "./img")
#
# The summary shows that for the 10 columns and 9,991 observations that there is no duplication of data and no missing values. The UDI and Product ID are unique for every observation. The distribution for the key process variables seems to be fairly well normally distributed, with rotational speed being slightly shewed to the left had side.
#
# The target columns and failure type columns shows that process failures occur 3.5% of the time making occurrences of failure rather rare for the overall process. When we look at the specific failure types it can be seen that most of failure occurrences of less than one percent. This will most likely mean that performance matrices of RMSE and Accuracy will not be the best measures by which to access model performance, selected metrics will be discussed in a later section.
#
#####  2.5 Data Exploration  #####
#
# This section focuses on visualizing and understanding what how different process variables impact the different failure types that we need to classify or predict. Initially this section looks at each individual process variables and how these process variables influence or are influenced by product type and failure type.
#
# Air Temperature:
#
#   Air temperature is more of an environmental variable than a process variable and can be much harder to control.


ggplot(MM) +
  aes(x = `Air_temperature_K`) +
  geom_histogram(bins = 30L, fill = "#4682B4") +
  labs(title = "Air Temperature distribution") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

MM %>% dplyr::select(.,`Air_temperature_K`,Type, Failure_Type) %>% group_by(Type, Failure_Type) %>% find_skewness(., value = TRUE) %>%  kbl(caption = "Check of Air Temperture Skewness") %>%
  kable_classic(full_width = F, html_font = "Cambria")

ggplot(MM) +
  aes(x = `Air_temperature_K`, fill = Type) +
  geom_density(adjust = 1.5) +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Air Temperature & Type") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Type))

ggplot(MM) +
  aes(x = "", y = `Air_temperature_K`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Air Temperature & Type",
       subtitle = "For Two Class Failures") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(target))

ggplot(MM) +
  aes(x = "", y = `Air_temperature_K`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Air Temperature & Type",
       subtitle = "For Failure Types") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Failure_Type))




# Key takeaways Air Temperature:
#
#   The histograms show that the air temperature follows an approximate normal distribution centered around 300 K
#
# The density plots for the 3 product types are very similar in their shape, with H type profile is slightly less pronounced then the L & M. The density plots clearly show that product types are exposed the same air temperatures during production.
#
# The Failure type box plots allow us to draw some simple conclusions, Heat Dissipation failures are more likely to occur at higher Air Temperature for all product types. Shows that random failure for type M are more likely to occur at lower temperature than for H & L. The box plots all show a very narrow range for which Power Failures & Over strain failure occur for Type H products.
#
# Process Temperature


ggplot(MM) +
  aes(x = `Process_temperature_K`) +
  geom_histogram(bins = 30L, fill = "#4682B4") +
  labs(title = "Process temperature distribution") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

MM %>% dplyr::select(.,`Process_temperature_K`,Type, Failure_Type) %>% group_by(Type, Failure_Type) %>% find_skewness(., value = TRUE) %>%  kbl(caption = "Check of Process Temperture Skewness") %>%
  kable_classic(full_width = F, html_font = "Cambria")

ggplot(MM) +
  aes(x = `Process_temperature_K`, fill = Type) +
  geom_density(adjust = 1.5) +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Process temperature & Type") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Type))

ggplot(MM) +
  aes(x = "", y = `Process_temperature_K`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Process temperature & Type",
       subtitle = "For Two Class Failures") +theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(target))

ggplot(MM) +
  aes(x = "", y = `Process_temperature_K`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Process temperature & Type",
       subtitle = "For each Failure Type") +theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Failure_Type))


# Key takeaways Process Temperatures:
#
#   The histograms show that the process temperature follows an approximate normal distribution centered around 310 K
#
# The density plots for the 3 product types show that type L product has a very visible density peck at around 311'k this peck can be seen in type M but the peak is less distinct and there is no visible peak at all for M. It seems that the process temperature for L & M type products is hotter than M
#
# The box plots are show some really narrow ranges of process temperatures for when heat dissipation failure occur for all product types. Again with random failure we see a very narrow range with short whiskers for Type M which is different to the range for L & H which themselves are very similar.
#
# Rotational Speed



ggplot(MM) +
 aes(x = `Rotational_speed_rpm`) +
 geom_histogram(bins = 30L, fill = "#4682B4") +
 labs(title = "Rotational speed distribution") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

MM %>% dplyr::select(.,`Process_temperature_K`,Type, Failure_Type) %>% group_by(Type, Failure_Type) %>% find_skewness(., value = TRUE) %>%  kbl(caption = "Check of Rotational Speed Skewness") %>%
  kable_classic(full_width = F, html_font = "Cambria")

ggplot(MM) +
  aes(x = `Rotational_speed_rpm`, fill = Type) +
   geom_density(adjust = 1.5)  +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Rotational speed & Type") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Type))

ggplot(MM) +
  aes(x = "", y = `Rotational_speed_rpm`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Rotational speed & Type",
       subtitle = "For Two Class Failures") +theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(target))


ggplot(MM) +
  aes(x = "", y = `Rotational_speed_rpm`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Rotational speed & Type",
       subtitle = "For each Failure Type") +theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Failure_Type))



# Key takeaways Process Temperatures:
#
# The histograms show that the rotational has a slight shrew towards the left hand side of the plot.
#
# The density plots clearly show that there is little difference in the rotational speeds used across the 3 product types.
#
# The box plots for Heat Dissipation, Over-strain, Random Failure are very narrow with smaller whiskers. Heat Dissipation & over-strain interquartile ranges do not heavily overlap with the interquartile range of No Failure, while the interquartile ranges of random failures do. Power Failure has the largest range of the failure types in this plots with a very low median, showing that power failures happen at both high and low rotational speeds with a tendency to occur more at low speed. Tool wear also has a narrow range with the median being towards the low ranges of rpm.
#
# The box plots also indicate that there are a number of statistical outliers predominately for No.Failure classification.
#
# Tool Wear


 ggplot(MM) +
 aes(x = `Tool_wear_min`) +
 geom_histogram(bins = 30L, fill = "#4682B4") +
 labs(title = "Tool wear distribution") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

 MM %>% dplyr::select(.,`Process_temperature_K`,Type, Failure_Type) %>% group_by(Type, Failure_Type) %>% find_skewness(., value = TRUE) %>%  kbl(caption = "Check of Tool wear Skewness") %>%
  kable_classic(full_width = F, html_font = "Cambria")

ggplot(MM) +
  aes(x = `Tool_wear_min`, fill = Type) +
   geom_density(adjust = 1.5)  +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Tool wear & Type") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Type))

ggplot(MM) +
  aes(x = "", y = `Tool_wear_min`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Tool wear & Type",
       subtitle = "For Two Class Failures"
) +theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(target))

ggplot(MM) +
  aes(x = "", y = `Tool_wear_min`, fill = Type) +
  geom_boxplot(shape = "circle") +
  scale_fill_brewer(palette = "Accent",
                    direction = 1) +
  labs(title = "Tool wear & Type",
       subtitle = "For each Failure Type") +theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  facet_wrap(vars(Failure_Type))


# Key takeaways tool wear:
#
# The histograms shows that tool life very quickly drops off after 200 mins of use, by the steep line at the right side.
#
# The density plot clearly show that there is little difference in the tool wear across the 3 product types, it seems that the right hand slope of the density plots is steeper for product H than it is for L & M
#
# The ranges of Heat dissipation, Power failure overlap those of No failure, Overstrain & Tool wear failures have narrow ranges with short whiskers in either direction with no overlap of the interquartile range of No Failure.
#
# After reviewing the process variable is isolation, from the density plots it can be seen that the process variables Air Temperature, process Temperature, tool wear, rotational speed and torque used to manufacture the different product type are similar.
#
# From the box plots it is possible to see what ranges different process failures occur for different product types. In a number of previous box plot it was possible to see what values for the process variable would likely result in some form of failure.
#
# More Exploration:
#
# The aim of the section below is to further the understanding gained from the previous section and this time instead of looking at the process variable in isolation we shall a look how combinations of two different process variables relate to the different failure modes that occur during the manufacturing process.
#
# Air Temperature & Process Temperature


plt1 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Process_temperature_K`
  ) +
  geom_point(shape = "circle", size = .8) +
  geom_smooth(method='lm', se = TRUE)  +
  scale_color_hue(direction = 1) +
  labs(
    title = "Air Temperture & Process Temperature"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Process_temperature_K`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  geom_smooth(method='lm', se = TRUE)  +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# As can be seen from the charts air temperature has a strong correlation to process temperature, this correlation is strongest for Heat Dissipation, and weaker for random & tool wear failures. Heat dissipation failure are all tightly cluster in the top right-hand side of the failure charts. Meaning high air temperature and high process temperatures are ideal conduction for a heat dissipation failure to occur.
#
# Air Temperature & Rotational speed


plt1 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Rotational_speed_rpm`
  ) +
  geom_point(shape = "circle", size = .8) +
  geom_smooth(method='lm', se = TRUE)  +
  scale_color_hue(direction = 1) +
  labs(
    title = "Air Temperture & Rotational speed"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Rotational_speed_rpm`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# Air Temperature & Rotational speed do not have a linear correlation, but the failure type scatter plots have again provided some really nice distinctive clusters of when the failure occur in relation to Air Temperature and Rotational speed. the data point clustering for tool wear is quite large compared to the other failure modes, but it still easy to see what failure can occur and will not occur.
#
# Air Temperature & Torque


plt1 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Torque_Nm`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Air Temperture & Torque"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Torque_Nm`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# Again, it is possible to see obvious data point cluster for each of the failures mode in relation to Torque and Air Temperature. As with the previous plot in this section Heat dissipation power failure and over-strain data points have all created quite easy to see data grouping. Random and Tool wear failures do not seem to want to create nice tight groups like the others, the grouping themselves do not seem to significantly overlap.
#
# Air Temperature & Tool wear


plt1 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Tool_wear_min`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Air Temperture & Tool wear"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Air_temperature_K`,
    y = `Tool_wear_min`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2

# In these plots the data points for tool wear failure have all created a grouping at the top of the charts, which is the first time we have seen a nice tight clustering of its data points. Other previous grouping is still visible although not as densly grouped together as previous plots. Power failure clearly showing that Tool & Air Temperature do not correlate for the occurrence of the failure mode.
#
# Process Temperature & Rotational_speed_rpm


plt1 <- ggplot(MM) +
  aes(
    x = `Process_temperature_K`,
    y = `Rotational_speed_rpm`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Process Temperture & Rotational speed"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Process_temperature_K`,
    y = `Rotational_speed_rpm`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# As with previous plot we are seeing very obvious data points clusters for various failure types.
#
# Process Temperature & Torque


plt1 <- ggplot(MM) +
  aes(
    x = `Process_temperature_K`,
    y = `Torque_Nm`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Process Temperture & Torque"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Process_temperature_K`,
    y = `Torque_Nm`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# Again, very distinct groupings of data points for the failure types. With this plot like the others it is clear between what values different failure modes will and won't occur.
#
# Process Temperature & Tool wear


plt1 <- ggplot(MM) +
  aes(
    x = `Process_temperature_K`,
    y = `Tool_wear_min`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Process Temperture & Tool wear"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Process_temperature_K`,
    y = `Tool_wear_min`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2


# The data points have again created dense clearly visible groupings clearly showing what failure modes can happen between a range of process variables.
#
# Rotational speed & Torque


plt1 <- ggplot(MM) +
  aes(
    x = `Rotational_speed_rpm`,
    y = `Torque_Nm`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Rotational speed  & Torque"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Rotational_speed_rpm`,
    y = `Torque_Nm`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# Contained data point grouping for the of the failure types, show the range for the two-process variable at which the failures occur.


# Rotational speed & Tool wear


plt1 <- ggplot(MM) +
  aes(
    x = `Rotational_speed_rpm`,
    y = `Tool_wear_min`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Process Temperture & Tool wear"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Rotational_speed_rpm`,
    y = `Tool_wear_min`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2


# Torque & Tool Wear


plt1 <- ggplot(MM) +
  aes(
    x = `Torque_Nm`,
    y = `Tool_wear_min`
  ) +
  geom_point(shape = "circle", size = .8) +
  scale_color_hue(direction = 1) +
  labs(
    title = "Torque & Tool wear"
  )


plt2 <- ggplot(MM) +
  aes(
    x = `Torque_Nm`,
    y = `Tool_wear_min`,
    colour = Failure_Type
  ) +
  geom_point(shape = "circle", size = 1) +
  scale_color_brewer(palette = "Dark2", direction = 1)  +
  theme(legend.position = "none") +
  labs(
    subtitle = "Broken down by Failure Type"
  ) +
  facet_wrap(vars(Failure_Type))


plt1
plt2



# By comparing all the different process variables against one another, it is possible to see the ranges in which these failure modes occur. The data clearly shows that the failures do concur with set parameter ranges with some overlap, modelling should be able to achieve some good results overall, misclassification of failure types is likely to occurs because of this overlap.

#####  3 MODELLING  #####

###### 3.1 Partition Data######

# Next, the seed is going to be set to achieve repeatable results and the Machine Failures data set will be 60/20/20 split. This means 60% of the Machine Failures data frame is going to be used for the training data set, train_data, 20% to the testing data set, test_data, and 20% to the testing data set, test_data. The 60/20/20 split ratio has been chosen as the data set is deemed to be large enough. The creation of the testing and training data sets can be found to the script and rmd files.


## Create Training/Test/Validations data sets
### No index for validation as the validation data is essentially the full data set minus the test and train data indexes
### Create Training/Test/Vaildations data sets ###
###
set.seed(12505)
indexs <- splitTools::partition(MM$Failure_Type, p = c(train = 0.6, test = 0.2, final_val = 0.2))
Train_Index <- indexs$train
Test_Index <- indexs$test
Val_Index <- indexs$final_val

MM %>% dplyr::select(.,target,Failure_Type) %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n(), Prop = n()/nrow(MM))%>%  kbl(caption = "Failure Type proportions for entire data set") %>%
  kable_classic(full_width = F, html_font = "Cambria")

MM[Train_Index,] %>% dplyr::select(.,target,Failure_Type) %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n(), Prop = n()/nrow(MM[Train_Index,]))%>%  kbl(caption = "Failure Type proportions for train data set") %>%
  kable_classic(full_width = F, html_font = "Cambria")

MM[Test_Index,] %>% dplyr::select(.,target,Failure_Type) %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n(), Prop = n()/nrow( MM[Test_Index,]))%>%  kbl(caption = "Failure Type proportions for test data set") %>%
  kable_classic(full_width = F, html_font = "Cambria")


MM[Val_Index,] %>% dplyr::select(.,target,Failure_Type) %>% group_by(target,Failure_Type) %>% dplyr::summarise(n = n(), Prop = n()/nrow(MM[Val_Index,]))%>%  kbl(caption = "Failure Type proportions for validation data set") %>%
  kable_classic(full_width = F, html_font = "Cambria")






##### 3.2 Model Baselines  #####
#
# Two modelling algorithms were chosen for this project. They are caret - Ranger and caret - xgbTree. These two modelling algorithms are used for modelling both the two class, and mutli-class response columns.
#
# At this stage each of the two techniques for each of the two target columns are trained. This will be used to create a baseline set of statistics that can be used as a reference points during the models development stage.



cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS another to support load balancing
registerDoParallel(cluster)

### Create Model Base Lines ###
# Place holders for model results
Models_Results_Binary <- NULL
Models_Results_Class_Overall <- NULL
###2 Class Modelling baselines ###



#2Class Grid of all test/modelling conditions including the baselines for models
Grid_2C <- expand.grid(Cart_Methd = c("ranger","xgbTree"), # Specfies what caret method to use
                       Target_Column = c("target"), #,"Failure_Type""target"
                       Sampling = c("No-Sampling","up"),
                       OutLiear_Treatment = c("No_Treatment","rm_OutLiear_No_Groups","rm_OutLiear_Type","rm_OutLiear_target","rm_OutLiear_Type_target"),
                       Class_Weights = c("No_Weights","Applied"))


# VaildateGrid User Defined function for removing test cases that are not needed
# (UP Sampling & Case Weight, Removal of Case weights for models that don't expect them)
Grid_2C <- VaildateGrid(Grid_2C)

# Runs rows 1:5 of grid to get baselines for each of the selected modelling techniques

for( i in 1:2){
  x<- as.numeric(i)
  out <- Apply_Caret_Models(x,Grid = Grid_2C)
  Models_Results_Binary <- rbind(Models_Results_Binary,out)
}



### Mutli Class Modelling baselines ###
#nClass Grid of all test/modelling conditions including the baselines for models
Grid_nC <- expand.grid(Cart_Methd = c("ranger","xgbTree"), # Specfies what caret method to use
                       Target_Column = c("Failure_Type"), #,"Failure_Type""target"
                       Sampling = c("No-Sampling","up"),
                       OutLiear_Treatment = c("No_Treatment","rm_OutLiear_No_Groups","rm_OutLiear_Type","rm_OutLiear_target","rm_OutLiear_Type_target"),
                       Class_Weights = c("No_Weights","Applied"))

Grid_nC <- VaildateGrid(Grid_nC)

for( i in 1:2){
  x <- as.numeric(i)
  out <- Apply_Caret_Models(x, Grid = Grid_nC)
  Models_Results_Class_Overall <- rbind(Models_Results_Class_Overall,out)
}


stopCluster(cluster)



Models_Results_Binary[1:2,] %>% dplyr::select(.,"target",  "Method", "Sampling_Method", "OutLiear_Treatment", "Class_Weights","F0.5", "RSME","Overall_Error","AUC","F1","Balanced Accuracy","Accuracy","Kappa") %>%
  kbl(caption = "Two Class Model Baselines") %>%
  kable_classic(full_width = T, html_font = "Cambria")


# The baselines for the Two Class Classification are all pretty close, ranger seems to be performing slightly better at this stage when we look across the key measures the two measure where ranger is not leading is AUC, and LogLoss.


tmp <- Models_Results_Class_Overall[1:2,] %>%
  dplyr::select(., "target",  "Method", "Sampling_Method", "OutLiear_Treatment", "Class_Weights","Overall_Error","AUC","Mean_Error","LogLoss","Accuracy","Kappa")

row.names(tmp) <- NULL
tmp %>%
  kbl(caption = "Multiclass Model Baselines") %>%
  kable_classic(full_width = T, html_font = "Cambria")

# The baselines for the Mutli Class Classification are also all pretty close, with ranger leading the way performing best for all measures.

#####  3.3 Model Development  #####

# Here the aim is to improve upon the baselines we have already gathered. We shall look at removing outliers using a number of different group options as well as up sampling on the data to improve the prediction outcomes.

# Here the aim is to improve upon the baselines we have already gathered. We shall look at removing outliers using four different group options as well as up sampling on the data to improve the prediction outcomes.

# Outliears Treatments Explained:

  # The interquartilerange is the area between the 75th and the 25th percentile of a distribution or i can be thought of as the middle 50% of the distribution. A point become an outlier if it is above the 75th or below the 25th percentile by a factor of 1.5 times the interquartile range.

# rm.OutLiear_Type: Removal of rows where for "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm","Torque_Nm" or "Tool_wear_min data points are out not with in the second or third quartile.

# rm.OutLiear_Type: Removals of rows where for "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm","Torque_Nm" or "Tool_wear_min data points are out not with in the second or third quartile when group by product type.

# rm.OutLiear_Type:  Removals of rows where for "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm","Torque_Nm" or "Tool_wear_min data points are out not with in the second or third quartile when group target response column.

# rm.OutLiear_Type:  Removals of rows where for"Air_temperature_K", "Process_temperature_K","Rotational_speed_rpm","Torque_Nm" or "Tool_wear_min data points are out not with in the second or third quartile when grouped by both product type andtarget response column.

# Sampling:

  # Upsampling is a process that synthetically generates additional data points (corresponding to minority class) are added into the dataset. After doing so all labels in the data set have equal proportions, following this procedure the model from inclining towards the majority class.

# Down sampling was considered at the early stage but initial testing showed that the resulting samples were to small to provided models enough data for models to have provide accurate results.

# Class Weights:

  # Most machine learning algorithms are not especially useful when working with imbalanced data. Class weights can be used modify the current training algorithm to consider the imbalance of the data classes. This is dine by giving different weights for the class based on the proportions of class with in the training data sets. The weights will influence the classification of the classes during the training phase. The whole purpose is to penalize the misclassification made by the minority class by setting a higher class weight and at the same time reducing weight for the majority class. For this reason testing using Up sampling and case weight have been removed as up sampling create equal proportions of the classes, essentially making all the class weights equal.


### 2 Class Modelling test Cases ###

cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS another to support load balancing
registerDoParallel(cluster)
#2Class Grid of all test/modelling conditions including the test cases for models

# Runs  test cases for each of the selected modelling techniques

for( i in 3:nrow(Grid_2C)){
  x<- as.numeric(i)
  out <- Apply_Caret_Models(x,Grid = Grid_2C)
  Models_Results_Binary <- rbind(Models_Results_Binary,out)
}

Models_Results_Binary <- Models_Results_Binary %>% mutate(Tuning ="No_Tuning")

### Mutli Class Modelling test cases ###
#nClass Grid of all test/modelling conditions including the test cases for models


for( i in 3:nrow(Grid_nC)){
  x<- as.numeric(i)
  out <- Apply_Caret_Models(x,Grid = Grid_nC)
  Models_Results_Class_Overall <- rbind(Models_Results_Class_Overall,out)
}

Models_Results_Class_Overall <- Models_Results_Class_Overall %>% mutate(Tuning ="No_Tuning")

stopCluster(cluster)

#Two Class Modelling Results

#Ranger Current Results:


Make_2Class_Conditional_Table(df = Models_Results_Binary,mth = "ranger")

#XgbTree Current Results:


Make_2Class_Conditional_Table(df = Models_Results_Binary,mth = "xgbTree")

#Multiclass Model Test Case Results

#Ranger Current Results:


Make_NClass_Conditional_Table(df = Models_Results_Class_Overall,mth = "ranger")


### xgbTree Current Results:###


Make_NClass_Conditional_Table(df = Models_Results_Class_Overall,mth = "xgbTree")


#### 3.4 Model Analysis ####

### 2 Class Models down selection ###

#Having run the baselines and model development options its time to down select modelling options before further tuning attempts, for model down selection in have choose 4 metrics to help with down selection these are: F0.5 - fbeta 0.5, AUC, RSME & balanced Acc. I shall select the top 2 performers for each metric, remove any duplicate models that are selected more than once, and then tune to these to help down select the final model.



cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS another to support load balancing
registerDoParallel(cluster)
Binary_Best_Models_Indexs <- NULL
Binary_Best_Models_Indexs <- rbind(Binary_Best_Models_Indexs, t(c(topn(Models_Results_Binary$F0.5, 2, decreasing = TRUE))))
Binary_Best_Models_Indexs <- rbind(Binary_Best_Models_Indexs, t(c(topn(Models_Results_Binary$RSME, 2, decreasing = FALSE))))
Binary_Best_Models_Indexs <- rbind(Binary_Best_Models_Indexs, t(c(topn(Models_Results_Binary$AUC, 2, decreasing = TRUE))))
Binary_Best_Models_Indexs <- rbind(Binary_Best_Models_Indexs, t(c(topn(Models_Results_Binary$F1, 2, decreasing = TRUE))))
Binary_Best_Models_Indexs <- rbind(Binary_Best_Models_Indexs, t(c(topn(Models_Results_Binary$Kappa, 2, decreasing = TRUE))))
Binary_Best_Models_Indexs <- unique(Binary_Best_Models_Indexs)

Index_binary <- c(Binary_Best_Models_Indexs)
length(Index_binary)
Index_binary <- unique(Index_binary)

Binary_Models_DownSelected <- Models_Results_Binary[Index_binary,1:5] %>% as.data.frame
col_order <- c("Method","target","Sampling_Method", "OutLiear_Treatment","Class_Weights")
Binary_Models_DownSelected <- Binary_Models_DownSelected[, col_order]

Models_Results_Binary <- Models_Results_Binary %>% mutate(Tuning ="No_Tuning")


for (k in 1:nrow(Binary_Models_DownSelected)) {
  x<- as.numeric(k)
  out <- Apply_Caret_Models(x, Grid = Binary_Models_DownSelected,Tune = "Yes")
  out <- out %>% mutate(Tuning = "With_Tuning")
  Models_Results_Binary <- rbind(Models_Results_Binary,out)
}

stopCluster(cluster)


Binary_Models_DownSelected %>%
  kbl(caption = "Models Selected for tuning") %>%
  kable_classic(full_width = T, html_font = "Cambria")



Models_Results_Binary %>% dplyr::filter(Tuning == "With_Tuning") %>%  dplyr::select(.,"target","Method", "Sampling_Method", "OutLiear_Treatment", "Class_Weights","F0.5", "RSME","Overall_Error","AUC","F1","Balanced Accuracy","Accuracy","Kappa") %>%
  kbl(caption = "Best performing Two Class Models Results with Tuning") %>%
  kable_classic(full_width = T, html_font = "Cambria")


### 2 Class Modelling Plots & Review ###


img <- Create_Plots(Models_Results_Binary,"F0.5")
im<-load.image(img)
plot(im)



#For F0.5 xgbTree is the performs better that than ranger, having a high mean for F0.5 across all techniques applied. UP sampling has an adverse effect on both modelling methods, resulting in a considerable drop in F0.5. Applying class weight to ranger created a slight shift to F0.5 by increasing the mean F0.5 score when compared with having no weights applied, but ranger achieved higher F0.5 scores when weights were not applied. Any improvement achieved by applying class weight to ranger minimal if any. Removing outliers with no groupings by target or type and removing outliers when grouping by Type both does seems to lift the F0.5 score that the trained models achieved. xgbTree gains the most improvement of the two modelling techniques. It worth noting the Removing outliers when grouping the data by type and target resulted in higher F5.0 scores for ranger, xgbTree did not respond as positively to the same outlier removal technique.


img <- Create_Plots(Models_Results_Binary,"AUC")
im<-load.image(img)
plot(im)
#unlink(img)

#XgbTree definelyperforms better than ranger achieving higher AUC score across the board. The techniques for outlier removal did not improve the overall scores the models achieved.


img <- Create_Plots(Models_Results_Binary,"RSME")
im<-load.image(img)
plot(im)

#Interestingly Removing outliers' grouped by type had a positive effect in reducing the RSME for xgbTree and negligible effect on ranger, but for removing outliers when group by type and target has a positive effect on ranger but hardly any effect on xgbTree.


img <- Create_Plots(Models_Results_Binary,"Overall_Error")
knitr::include_graphics(img)
#unlink(img)

#Up sampling had a detrimental effect on the overall error of the models, removal of outliers with no grouping does seem to reduce the overall error of the models.


img <- Create_Plots(Models_Results_Binary,"Mean_Error")
im<-load.image(img)
plot(im)

#Up sampling does not improve the mean error of the models, removing outliers does seem to reduce the sizes of the distributions for both model techniques and does reduce the average mean error.


img<-Create_Plots(Models_Results_Binary,"LogLoss")
im<-load.image(img)
plot(im)

#xgbTree performs much better than ranger across all techniques. Interestingly the two-modelling method performed very differently based on this metric, it is clear to see here that xgbTree performs much better than ranger in reducing LogLoss during the modelling process.


img <- Create_Plots(Models_Results_Binary,"Balanced Accuracy")
im<-load.image(img)
plot(im)

#Interesting to see how up sampling allowed xgbTree to achieve a 7%-8% improvement in balancedaccuracy while ranger suffers 7% decrease in it balanced accuracy when up sampling is used. Ranger also suffered a considerable drop when outliers where removed when group the data by type.


img <-Create_Plots(Models_Results_Binary,"Accuracy")
im<-load.image(img)
plot(im)

#It possible to see how removal of outliers does provide improvement here by the reduced size of thebox plot distributions.


img <- Create_Plots(Models_Results_Binary,"Kappa")
im<-load.image(img)
plot(im)

#Removing outliers by type again can be shown to have a positive effect on xgbTree method and a detrimental affect on ranger, allowing xgbTree to achieved some of it highest Kappa scores while ranger achieved its lowest Kappa scores for the same outlier treatment method.

#Multi Class Models Down selection


### Mutli Class Modelling test cases ###


cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS another to support load balancing
registerDoParallel(cluster)

NClass_Best_Models_Indexs <- NULL
NClass_Best_Models_Indexs <- rbind(NClass_Best_Models_Indexs, t(c(topn(Models_Results_Class_Overall$Overall_Error, 1, decreasing = TRUE))))
NClass_Best_Models_Indexs <- rbind(NClass_Best_Models_Indexs, t(c(topn(Models_Results_Class_Overall$AUC, 1, decreasing = TRUE)))) #larger better
NClass_Best_Models_Indexs <- rbind(NClass_Best_Models_Indexs, t(c(topn(Models_Results_Class_Overall$Kappa, 1, decreasing = TRUE))))
NClass_Best_Models_Indexs <- rbind(NClass_Best_Models_Indexs, t(c(topn(Models_Results_Class_Overall$Accuracy, 1, decreasing = TRUE))))
NClass_Best_Models_Indexs <- unique(NClass_Best_Models_Indexs)

Index_nClass <- c(NClass_Best_Models_Indexs)

Index_nClass <- unique(Index_nClass)

nClass_Models_DownSelected <- Models_Results_Class_Overall[Index_nClass,1:5] %>% as.data.frame


col_order <- c("Method","target","Sampling_Method", "OutLiear_Treatment","Class_Weights")



nClass_Models_DownSelected <- nClass_Models_DownSelected[, col_order]


Models_Results_Class_Overall <- Models_Results_Class_Overall %>% mutate(Tuning ="No_Tuning")
nClass_Models_DownSelected <- as.data.frame(nClass_Models_DownSelected)


for (k in 1:nrow(nClass_Models_DownSelected)) {
  x<- as.numeric(k)
  out <- Apply_Caret_Models(x,Grid = nClass_Models_DownSelected, Tune = "Yes")
  out <- out %>% mutate(Tuning = "With_Tuning")
  Models_Results_Class_Overall <- rbind(Models_Results_Class_Overall,out)
}

stopCluster(cluster)


rownames(nClass_Models_DownSelected)<-NULL
nClass_Models_DownSelected %>%
  kbl(caption = "Mutli-Class Classification Models Selected for tuning") %>%
  kable_classic(full_width = F, html_font = "Cambria")


rownames(Models_Results_Class_Overall)<-NULL
Models_Results_Class_Overall %>% dplyr::filter(Tuning == "With_Tuning") %>% dplyr::select(., "target",  "Method", "Sampling_Method", "OutLiear_Treatment", "Class_Weights","Overall_Error","AUC","Mean_Error","LogLoss","Accuracy","Kappa") %>%
  kbl(caption = "Best performing Mutli-Class Classification Models Results with Tuning") %>% kable_classic(full_width = F, html_font = "Cambria")


### Mutli Class Modelling Plots & Review ###


img <- Create_Plots(Models_Results_Class_Overall,"Overall_Error")
knitr::include_graphics(img)
#unlink(img)

#The overall Error does not see any improve from any of the techniques applied, and generates the best performance measures when no sampling or outliers removal techniques are applied.


img <- Create_Plots(Models_Results_Class_Overall,"AUC")
knitr::include_graphics(img)
#unlink(img)

#The AUC measure does not seem to improve from any of the techniques that have been applied here. Ranger has larger distribution of the values it achieved as well as higher and lower values than xgbTree.


img <- Create_Plots(Models_Results_Class_Overall,"Mean_Error")
knitr::include_graphics(img)
#unlink(img)

#The mean error does not see any improve from any of the techniques applied, and generates the best performance measures when no sampling or outliers removal techniques are applied.


img <- Create_Plots(Models_Results_Class_Overall,"LogLoss")
knitr::include_graphics(img)
#unlink(img)

#Overall xgbTree does seem to perform better than ranger for this metric, though both model method seem to be negatively impact by the various outlier removal techniques it seems that the detrimental impact to ranger is much more pronounced than theimpact for xgbTree.


img <- Create_Plots(Models_Results_Class_Overall,"Accuracy")
knitr::include_graphics(img)
#unlink(img)

#The mean error does not see any improvement from any of the techniques applied, and generates the best performance measures when no sampling or outliers removal techniques are applied.


img <- Create_Plots(Models_Results_Class_Overall,"Kappa")
knitr::include_graphics(img)
#unlink(img)

#It appears that Up sampling had a negative impact for ranger and xgbTree but the affect is much more pronounced for ranger than xgbTree.

#### 3.5 Final Model Selection ####


### Final Model Down Selection ###
#2 Class Final Model Down Selection
Binary_Final_Models_Indexs <- NULL
Binary_Final_Models_Indexs <-  rbind(Binary_Final_Models_Indexs, t(c(topn(Models_Results_Binary$AUC, 1, decreasing = TRUE))))

Final_Index_binary <- c(Binary_Final_Models_Indexs)
Binary_Models_Final <- Models_Results_Binary[Binary_Final_Models_Indexs,]



NClass_Final_Models_Indexs <- NULL
NClass_Final_Models_Indexs <- rbind(NClass_Final_Models_Indexs, t(c(topn(Models_Results_Class_Overall$AUC, 1, decreasing = TRUE))))
NClass_Final_Models_Indexs <- unique(NClass_Final_Models_Indexs)

Index_Final_nClass <- c(NClass_Final_Models_Indexs)
nClass_Models_Final <- Models_Results_Class_Overall[Index_Final_nClass,]




#Two Class Classification: Data Manipulation Parameters selected for Validation Model


Binary_Models_Final[1:5] %>%
  kbl(caption = "Model Selected for Two Class Classification Validation") %>%
  kable_classic(full_width = F, html_font = "Cambria")


#
### 2 Class Final Model Run ###
#

cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS another to support load balancing
registerDoParallel(cluster)
#
## Run function for final model -- output a model opposed to stats
#
col_order <- c("Method","target","Sampling_Method", "OutLiear_Treatment","Class_Weights")
Binary_Models_Final <- Binary_Models_Final[, col_order]

TwoClass_Final_Model <- Final_Caret_Models_Model_Output(1,Binary_Models_Final)


#
## varibale required for results table functions to work
#
Method <- Binary_Models_Final[1,1] %>% as.character # Set varaible base on the row & column of supplied grid
Target_Column <- Binary_Models_Final[1,2] %>% as.character # Set varaible base on the row & column of supplied grid
Sampling <- Binary_Models_Final[1,3] %>% as.character # Set varaible base on the row & column of supplied grid
OutLiear_Treatment <- Binary_Models_Final[1,4] %>% as.character # Set varaible base on the row & column of supplied grid
Class_Weights <- Binary_Models_Final[1,5] %>% as.character # Set varaible base on the row & column of supplied grid



## Data Prep - Set data frame to be either 2class or multi_class ##
## ** section below taken for Apply_caret: create required df, recode df, create Val data set
Mod_data_func <- match.fun("Modify_Data_for_target")
Mod_data_func(MM,Target_Column) # function output dfm (df)
dfm <- dfm %>% dplyr::select(.,-UDI) # removes column that is not wanted
col_order <- c("Type","Air_temperature_K","Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min","target") #set desire column order
dfm <- dfm[, col_order] # Applies Order of columns
colnames(dfm) <-c ("Type","Air_temp","Process_temp", "Rotational_speed", "Torque", "Tool_wear","target") #Renames columns as () in column names result in errors


Val_data <- dfm[Val_Index, ] #Creates test data  dataframe
Val_data$target<- as.factor(Val_data$target)


set.seed(12505)
TwoClass_Final_pred <- predict(TwoClass_Final_Model, newdata = Val_data, type =  "raw" )

#generate the proberilities for each case
set.seed(12505)
TwoClass_Final_pred_Prob <- predict(TwoClass_Final_Model, newdata = Val_data, type =  "prob" ) %>% as.matrix
#Creates and store confusion matrix
TwoC_Final_Conf <- caret::confusionMatrix(TwoClass_Final_pred,Val_data$target)


#Call result fuction, and supplies the required variables for generate all the stats for the prblem class
TwoClass_Final_Result <- Binary_Results(Target_Column,Method,Sampling,OutLiear_Treatment,Val_data,TwoC_Final_Conf,TwoClass_Final_pred,TwoClass_Final_pred_Prob,Class_Weights,Models_Results_Binary)


stopCluster(cluster)


### Multi Class Classification: Data Manipulation Parameters selected for Validation Model ###


nClass_Models_Final[1:5] %>%
  kbl(caption = "Model Selected for Multi Class Classification Validation") %>%
  kable_classic(full_width = F, html_font = "Cambria")


### MutiClass Final Model Run ###
#
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS another to support load balancing
registerDoParallel(cluster)

#********************* Mutli_Class *********************#

col_order <- c("Method","target","Sampling_Method", "OutLiear_Treatment","Class_Weights")
nClass_Models_Final <- nClass_Models_Final[, col_order]

#
## Run function for final model -- output a model opposed to stats
#
nClass_Final_Model <- Final_Caret_Models_Model_Output(1,nClass_Models_Final)

#
## varibales required for results table functions to work
#
Method <- nClass_Models_Final[1,1] %>% as.character # Set varaible base on the row & column of supplied grid
Target_Column <- nClass_Models_Final[1,2] %>% as.character # Set varaible base on the row & column of supplied grid
Sampling <- nClass_Models_Final[1,3] %>% as.character # Set varaible base on the row & column of supplied grid
OutLiear_Treatment <- nClass_Models_Final[1,4] %>% as.character # Set varaible base on the row & column of supplied grid
Class_Weights <- nClass_Models_Final[1,5] %>% as.character # Set varaible base on the row & column of supplied grid



## Data Prep - Set data frame to be either 2class or multi_class ##
Mod_data_func <- match.fun("Modify_Data_for_target")
Mod_data_func(MM,Target_Column) # function output dfm (df)
dfm <- dfm %>% dplyr::select(.,-UDI) # removes column that is not wanted
col_order <- c("Type","Air_temperature_K","Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min","target") #set desire column order
dfm <- dfm[, col_order] # Applies Order of columns
colnames(dfm) <-c ("Type","Air_temp","Process_temp", "Rotational_speed", "Torque", "Tool_wear","target") #Renames columns as () in column names result in errors


Val_data <- dfm[Val_Index, ] #Creates test data  dataframe
Val_data$target<- as.factor(Val_data$target)

set.seed(12505)
nClass_Final_pred <- predict(nClass_Final_Model, newdata = Val_data, type =  "raw" )

#generate the proberilities for each case
set.seed(12505)
nClass_Final_pred_Prob <- predict(nClass_Final_Model, newdata = Val_data, type =  "prob" ) %>% as.matrix
#Creates and store confusion matrix
nC_Conf <- caret::confusionMatrix(nClass_Final_pred,Val_data$target)


#Call result fuction, and supplies the required variables for generate all the stats for the prblem class
Mutli_Class_Final_Result <- Class_Results(Target_Column,Method,Sampling,OutLiear_Treatment,Val_data,nC_Conf,nClass_Final_pred,nClass_Final_pred_Prob,Class_Weights)



stopCluster(cluster)

### Two Class Model Results ###


TwoClass_baseline <- Models_Results_Binary %>% filter(Method == Binary_Models_Final$Method[1], Sampling_Method == Binary_Models_Final$Sampling_Method[1], OutLiear_Treatment == Binary_Models_Final$OutLiear_Treatment[1], Class_Weights == Binary_Models_Final$Class_Weights[1])


TwoClassBaseLine <- TwoClass_baseline[1,] %>% dplyr::select(.,AUC,
                                                            `Balanced Accuracy`,
                                                            F0.5,
                                                            F1,
                                                            F2,
                                                            Kappa,
                                                            LogLoss,
                                                            Mean_Error,
                                                            RSME)


TwoC_final <- TwoClass_Final_Result %>% dplyr::select(.,AUC,
                                                        `Balanced Accuracy`,
                                                        F0.5,
                                                        F1,
                                                        F2,
                                                        Kappa,
                                                        LogLoss,
                                                        Mean_Error,
                                                        RSME)
Twoc_ImproveInPercent <- (abs((TwoC_final/TwoClassBaseLine)-1)*100)

Twoc_ImproveInPercent <- t(Twoc_ImproveInPercent)
colnames(Twoc_ImproveInPercent) <- c("% Increase on Baselines")

TwoC_Actual_Dif <- abs(TwoC_final - TwoClassBaseLine)

TwoC_Actual_Dif <- t(TwoC_Actual_Dif)
colnames(TwoC_Actual_Dif) <- c("Delta between Final & Baseline")

TwoC_Final_Summary <- cbind(t(TwoClassBaseLine),t(TwoC_final))
colnames(TwoC_Final_Summary) <- c("Baselines","Final Vaildation")
delta_TwoC <- cbind(Twoc_ImproveInPercent,TwoC_Actual_Dif)

TwoC_Final_Summary <- cbind(TwoC_Final_Summary,delta_TwoC) %>% round(5)

TwoClass_Final_Result %>% dplyr::select(target,Method,Sampling_Method,OutLiear_Treatment,Class_Weights) %>%
  kbl(caption = "Modelling varibles that were down selected for Final Vaildation") %>%
  kable_classic(full_width = T, html_font = "Cambria")

TwoC_Final_Summary %>%
  kbl(caption = "Comparison of Final Models Results to Model Baseline") %>%
  kable_classic(full_width = T, html_font = "Cambria")

TwoC_Final_Conf$table %>%
  kbl(caption = "Two Class Final COnfusion Matrix") %>%
  kable_classic(full_width = T, html_font = "Cambria")



#Even though the final modelling parameters that were selected for the validation were the same as the baseline model meaning that not up-sampling, no outliers treatment methods, or class weights were applied the final validation model performed well and resulted in strong improvements on the baseline statistics. The AUC values increase the least percentage wise from 0.96987 to 0.98874 which is a 1.94532% improvement, even though the improvement achieved was not huge still represents and model that is able to predict well between the two class. The final achieved balanced accuracy is a reasonable result and represents a significant improvement on the baselines. The final two class model do produce satisfactory results and all measures do demonstrate a model that does have good ability to classify process failures.



nClass_baseline <- Models_Results_Class_Overall %>% filter(Method == nClass_Models_Final$Method[1], Sampling_Method == nClass_Models_Final$Sampling_Method[1], OutLiear_Treatment == nClass_Models_Final$OutLiear_Treatment[1], Class_Weights == nClass_Models_Final$Class_Weights[1])


nC_BaseLine <- nClass_baseline[1,] %>% dplyr::select(.,AUC,
                                                     Mean_Error,
                                                     LogLoss,
                                                     Accuracy,
                                                     Kappa)


nC_final <- Mutli_Class_Final_Result %>% dplyr::select(.,AUC,
                                                       Mean_Error,
                                                       LogLoss,
                                                       Accuracy,
                                                       Kappa)

nC_ImproveInPercent <- (abs((nC_final/nC_BaseLine)-1)*100)

nC_ImproveInPercent <- t(nC_ImproveInPercent)
colnames(nC_ImproveInPercent) <- c("% Increase on Baselines")

nC_Actual_Dif <- abs(nC_final - nC_BaseLine)

nC_Actual_Dif <- t(nC_Actual_Dif)
colnames(nC_Actual_Dif) <- c("Difference between Final & Baseline")

nC_Final_Summary <- cbind(t(nC_BaseLine),t(nC_final))
colnames(nC_Final_Summary) <- c("Baselines","Final Vaildation")
delta_nC_ <- cbind(nC_ImproveInPercent,nC_Actual_Dif)

nC_Final_Summary <- cbind(nC_Final_Summary,delta_nC_) %>% round(5)

Mutli_Class_Final_Result %>% dplyr::select(target,Method,Sampling_Method,OutLiear_Treatment,Class_Weights) %>%
  kbl(caption = "Modelling variables that were down selected for Final Validation") %>%
  kable_classic(full_width = T, html_font = "Cambria")

nC_Final_Summary %>%
  kbl(caption = "Comparison of Final Models Results to Model Baseline") %>%
  kable_classic(full_width = T, html_font = "Cambria")

nC_Conf$table   %>%
  kbl(caption = "Final Mutli-Class Classification Confusion Matrix") %>%
  kable_classic(full_width = T, html_font = "Cambria")


# Final modelling parameters that were selected for the validation included up-sampling but no outliers treatment methods, or class weights were applied. The final validation model performed well with and resulted in modest improvements on the baseline statistics. the final model showed good abilities to predict Heat Dissipation Failure, overstrain failures, and power failures. The models struggled to predict Tool failure accurately and failed to predict any of the random failure at all. Overall, it is felt that the final model does have some good strengths, even though the final model only predicted one tool wear failures correctly, it is felt that this is still a good achievement considering the class imbalances of the data set. Although the models do have represent an improvement on baselines, there is opportunity to further improve the predictive power of this model to reduce the number of false predictions. If the models at least predicted a random failure even if the prediction were wrong feels like it would also strengthen the model as right now the models fail to predict any random failures at all. The absence of random failure predictions definitely feels like a weakness for this model, and would be a area that worth focusing on in future models.


#### 4.1 Conclusion ####
#
# The fact the two-class prediction model baseline and the final model used the same data is a little surprising, although random forest models do generally perform well with outliers it was not anticipated that after all the data modifications applied to the raw data that the baselines would be the one to be selected for the final model. That aside the final two class models is a solid model that does classify fails well and does perform well especially for AUC and Logloss.
#
# The mutli-class Classification model does predict some cases well, as does improve on the initially baselines. It does ability to make a random failure prediction and does not predict tool wear accurately. That side the models does show strengths in predicting Heat Dissipation, overstrain, and power failures.



#### 4.2 Future work and considerations ####
#
# The following points might be worth considering in the future to further improve the Machine Failure classification engine:
#
# Dimensional reduction techniques being applied to the data and then applying some clustering technique evaluate any opportunities for those models to be used either standalone or as part of an ensemble to help strengthen the current models
#
# Treating tool wear as a ordinal data set to see if that improves on the classification engine ability to predict tool wear, or reduce the number of false predictions.
#



#### 4.3 References ####
#
# Irizarry, Rafael A., "Introduction to Data Science: Data Analysis and Prediction Algorithms in R", webpage:https://rafalab.github.io/dsbook/
#
# LaTeX Equation Builder, webpage:https://latex.codecogs.com/eqneditor/editor.php
#
# knitr::kable and kableExtra Tutorial, webpage:https://cran.r-project.org/web/packages/kableExtra/vignettes/awesome_table_in_html.html
#
# [Chapter 11 Random Forests | Hands-On Machine Learning with R (bradleyboehmke.github.io)] webpage: https://bradleyboehmke.github.io/HOML/random-forest.html#out-of-the-box-performance
#
# Handling Class Imbalance with R and Caret, webpage:[Handling Class Imbalance with R and Caret - An Introduction | Wicked Good Data dpmartin42.github.io)] webpage:https://dpmartin42.github.io/posts/r/imbalanced-classes-part-1
#
# How to Remove Outliers in R, webpage:[How to Remove Outliers in R | R-bloggers]( webpage:https://www.r-bloggers.com/2020/01/how-to-remove-outliers-in-r/
#
# Handling Imbalanced Data - Machine Learning, Computer Vision and NLP, webpage:https://www.analyticsvidhya.com/blog/2020/11/handling-imbalanced-data-machine-learning-computer-vision-and-nlp/

