setwd("~/Coursea/XHarvard Data Sciene Program/Capstone/UCI Heart Disease Data")
library(doBy)
library(rpart)
library(caret)
library(MLmetrics)
library(tidyverse)
library(data.table)
library(GGally)
library(corrplot)
library(verification)
library(ROCR)
library(maptree)
library(glmnet)
library(gridExtra)
library(randomForest)
library(mgcv)
library(nnet)
library(pROC)
library(gbm)
library(e1071)
library(xgboost)
library(DT)
library(NeuralNetTools)
library(rpart.plot)
library(ggplot2)
library(doParallel)
library(magrittr)
library(caretEnsemble)
library(tidyr)
library(dplyr)
require(e1071) #Holds the Naive Bayes Classifier
library(arm)
library(mltools)
library(plyr)
library(party, mboost, plyr, partykit)
library(klaR)
library(ggplot2)
library(forecast)
library(bartMachine)
library(bnclassify)
library(adabag)
library(party)
library(mboost)
library(plyr) 
library(partykit)
library(rotationForest)
library(fastAdaboost)
library(ranger)
library(mice)

registerDoParallel(cores=10)
#"adaboost","bartMachine",
my_modelsx = c("adaboost","lda","knn","treebag","xgbTree","fda","pcaNNet","glmboost",
               "ada",
               "lvq",
               "mda",
               "ranger",
               "rf",
               "RRF",
               "gbm",
               "avNNet",
               "AdaBag",
               "bayesglm",
               "blackboost",
               "rpart","rotationForestCp","DImmp","Split")




##### Load Data #####
UCI <- readr::read_csv("./heart_disease_uci.csv",show_col_types = FALSE)
UCI<- as.data.frame(UCI)

##### Define Global Data Frames for storage of function results ##### 
###### model_perf ###### 
#df to hold results
model_perf <-tibble(Accuracy = 0,AUC=0,Balanced_Accuracy=0,DImpp="NA",Detection_Rate=0,F1=0,Kappa=0,logLoss=0,Model=0,Neg_Pred_Value=0,Pos_Pred_Value=0,prAUC=0,Precision=0,Recall=0,Resample="NA",Sensitivity=0,Specificity=0,Split="NA")

###### model_predictions ###### 
#df to hold results all the prediction results generated
tmp = data.frame(matrix(nrow = 0, ncol = length(my_modelsx)))
# assign column names
n <- length(my_modelsx)
x <- as.factor(rep(NA, n))
tmp <- rbind(x,tmp)

colnames(tmp) = (my_modelsx)
class(tmp)


model_predictions <- tmp %>% drop_na
rm(tmp)
   #tibble(adaboost ="NA",lda ="NA",KNN = "NA",treebag ="NA",xgbTree ="NA",fda ="NA",pcaNNet ="NA",DImpp ="NA",Split="NA") 

#- removed for error - ,RowID ="NA",treebag ="NA"rpart ="NA",

#model_predictions <- tibble(xgbTree ="NA",DImpp ="NA",Split="NA",RowID ="NA")


###### model_Scores ###### 
#df hold ML model measures
model_Scores <- tibble(Model ="NA",RMSE=0,MSE=0,MAE=0,DImpp ="NA",Split="NA") 


##### Functions #####
RMSE <- function(true_ratings, predicted_ratings) {
 sqrt(mean((true_ratings - predicted_ratings) ^ 2))
}

#Function to Calculate RMSE, MSE & MAE 
Model.stats <- function(true_ratings,predicted_ratings) {
 c(RMSE = RMSE(true_ratings,predicted_ratings),MSE = MSE(true_ratings,predicted_ratings),MAE = MAE(true_ratings,predicted_ratings))
}

#Function replaces NA values in numeric column by mean: 
 replace_by_mean <- function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  return(x)
 }

 # A function imputes NA observations for categorical variables by frequency: 
 replace_na_categorical <- function(x) {
   x %>% 
     table() %>% 
     as.data.frame() %>% 
     arrange(-Freq) ->> my_df
   
   n_obs <- sum(my_df$Freq)
   pop <- my_df$. %>% as.character()
   set.seed(29)
   x[is.na(x)] <- sample(pop, sum(is.na(x)), replace = TRUE, prob = my_df$Freq)
   rm(my_df)
   return(x)
 }
 ###### Encode Data ######
 Lbl_encode_Split <-function(df,train.RowIDs,test.RowIDs){
    df <- df %>% 
       mutate_if(is_character, as.factor) %>% 
       mutate_if(is.logical, as.factor) 
    
    cols <- df %>% dplyr::select(.,-Target) %>% dplyr::select(where(is.factor)) %>% colnames
    df[,cols] %<>% lapply(function(x) as.numeric(factor(x))-1) 
    md.pattern(df)
    
    df$Target <- as.factor(df$Target)
    df[,cols] %<>% lapply(function(x) as.numeric(x))
    #df_train_ml <- df_for_ml[T1_id,] 
    df_train_ml <- df %>% filter(rowid %in% train.RowIDs)
    df_train_ml <<- df_train_ml %>% dplyr::select(.,-rowid)
    df_test_ml <- df %>% filter(rowid %in% test.RowIDs) 
    test.rows = df_test_ml$rowid # Allows to tag predictions with row
    #Drop Row ID from Data set - not needed for ML
    df_test_ml <<- df %>% dplyr::select(.,-rowid)
 }
 
 df <- df %>% 
    mutate_if(is_character, as.factor) %>% 
    mutate_if(is.logical, as.factor) 
 
 cols <- df %>% dplyr::select(.,-Target) %>% dplyr::select(where(is.factor)) %>% colnames
 df[,cols] %<>% lapply(function(x) as.numeric(factor(x))-1) 
 md.pattern(df)
 
 df$Target <- as.factor(df$Target)
 df[,cols] %<>% lapply(function(x) as.numeric(x))
 
 Lbl_encode_only <-function(df){
    df <- df %>% 
       mutate_if(is_character, as.factor) %>% 
       mutate_if(is.logical, as.factor) 
    
    cols <- df %>% dplyr::select(.,-Target) %>% dplyr::select(where(is.factor)) %>% colnames
    df[,cols] %<>% lapply(function(x) as.numeric(factor(x))-1) 
    md.pattern(df)
    
    df$Target <- as.factor(df$Target)
    df[,cols] %<>% lapply(function(x) as.numeric(x))
    df <<- df
 }
 
 Scale_data <-function(df,train.RowIDs,test.RowIDs){
    #scale the dataset into a range of [0;1]: 
    class(df$rowid)
    df_for_ml <- df %>%  mutate_if(is.numeric, function(x) {(x - min(x)) / (max(x) - min(x))})
    #df_train_ml <- df_for_ml[T1_id,] 
    df_train_ml <- df_for_ml %>% filter(rowid %in% train.RowIDs)
    df_train_ml <<- df_train_ml %>% dplyr::select(.,-rowid)
    df_test_ml <- df_for_ml %>% filter(rowid %in% test.RowIDs) 
    test.rows = df_test_ml$rowid # Allows to tag predictions with row
    #Drop Row ID from Data set - not needed for ML
    df_test_ml <<- df_test_ml %>% dplyr::select(.,-rowid)
 }
 
 ##### Split S2 #####
 S2_split <-function(df,train.RowIDs,test.RowIDs){
    #scale the dataset into a range of [0;1]: 
    class(df$rowid)
    df_for_ml <- df
    #df_train_ml <- df_for_ml[T1_id,] 
    df_train_ml <- df_for_ml %>% filter(rowid %in% train.RowIDs)
    df_train_ml <<- df_train_ml %>% dplyr::select(.,-rowid)
    df_test_ml <- df_for_ml %>% filter(rowid %in% test.RowIDs) 
    test.rows = df_test_ml$rowid # Allows to tag predictions with row
    #Drop Row ID from Data set - not needed for ML
    df_test_ml <<- df_test_ml %>% dplyr::select(.,-rowid)
 }
 
 
 
 
 #### Modelling Functions ####
 Apply_Caret_Models <-function(df_train_ml,df_test_ml,DataMth,spl_meth) {

     newdata = df_test_ml
     df_train_ml$Target %<>% as.factor
     df_test_ml$Target %<>% as.factor
   set.seed(2)
   number <- 5
   repeats <- 5
   control <- trainControl(method = "repeatedcv", 
                           number = number , 
                           repeats = repeats, 
                           classProbs = TRUE, 
                           savePredictions = "all", 
                           index = createResample(df_train_ml$Target,times = repeats*number), 
                           summaryFunction = multiClassSummary, 
                           allowParallel = TRUE)
   
   
   
   # Simultaneously train some machine learning models: 
   
   set.seed(1255)
   
   # List all models that you want to train. For purpose of explanation 
   # I will only use 5 models: 
   library(fastAdaboost)
   # Train these ML Models: #"adaboost",
   my_models = c("adaboost","lda","knn","treebag","xgbTree","fda","pcaNNet","glmboost",
                 "ada",
                 "lvq",
                 "mda",
               "ranger",
                 "rf",
                 "RRF",
                 "gbm",
                 "avNNet",
                 "AdaBag",
                 "bayesglm",
                 "blackboost",
                 "rpart","rotationForestCp"
                 )
   ##removed "rpart","xgbLinear","bartMachine",
   #my_models = c("xgbTree")
   ##"naive_bayes",
   ##"nb",
   ##"nbDiscrete",
   ##  "superpc",
   # "tan",
   # "tanSearch", "rmda",
   # "dnn","xgbDART",
   
   model_list1 <- caretList(Target ~., 
                            data = df_train_ml,
                            trControl = control,
                            metric = "Accuracy", 
                            methodList = my_models)
   
   # output Performance values for Models #
   # Extract all results from ML ML models: 
   
   list_of_results <- lapply(my_models, function(x) {model_list1[[x]]$resample})
   
   # Convert to data frame: 
   
   df_results <- do.call("bind_rows", list_of_results) #%>% drop_na()
   
   #Add Model labels Data Imput method to data#
   
   df_results <- df_results %>% mutate(Model = lapply(my_models, function(x) {rep(x, number*repeats)}) %>% unlist()) 
   
   df_results <- as.data.frame(df_results)
   df_results <- df_results %>% mutate(DImpp = as.factor(MissDataMth))
   df_results <- df_results %>% mutate(Split = as.factor(spl_meth))
   
   
   model_perf <<- rbind.data.frame(model_perf,df_results)
   print(df_results)
   
   #output predicted values for Models # 
   #List to store predicted values for each of the models
   
   p <-lapply(my_models, function(x) {predict(model_list1[[x]], newdata = df_test_ml)})
   
   #Add Model labels Data Imput method to data 
   
   names(p) <- my_models # labels list with Model Name
   
   # converts list into dataframe
   
   pdf <- do.call("bind_rows", p) 
   pdf <- pdf %>% dplyr::mutate(DImpp = MissDataMth) 
   pdf <- pdf %>% dplyr::mutate(Split = spl_meth)  # removed for error - ,RowID = test.rows

   
   # Stores to master model_predictions df
   
   model_predictions <<- rbind.data.frame(model_predictions,pdf)
   print(pdf)
   
   ##### Confusion mtx code
   # cf <- lapply(my_models, function(x) {confusionMatrix(p[[x]], tdf$T1)})
   # names(cf) <- my_models
   
   #output Stats for Models # 
   #Outputs list of list using Model.stats func, modelName - rmse - mae - mse
   
   stats <-lapply(my_models, function(x) {Model.stats(as.numeric(df_test_ml$Target),as.numeric(p[[x]]))})
   
   names(stats) <- my_models # adds name to model list
   st <- do.call("bind_rows", stats) # converts list of list to df
   stx <- as.data.frame(st) 
   stx <- stx %>% mutate(DImpp = MissDataMth)
   stx <- stx %>% mutate(Split = spl_meth) # Tags models score with dat imp method
   
   row.names(stx) <- my_models #adds rows name to df for each model - 
   stx <- rownames_to_column(stx, var = "Model") # Adds name to top left column
   model_Scores  <<- rbind.data.frame(model_Scores,stx) # Binds the ml scores to global table
   print(stx)
   
   
   
   
   # Comparing among models based on a criteria of model performance selected: 
   
   # plots<-df_results %>%
   #  select(Accuracy, Model) %>%
   #  ggplot(aes(Model, Accuracy, fill = Model, color = Model)) +
   #  geom_boxplot(show.legend = FALSE, alpha = 0.3) +
   #  theme_minimal() +
   #  coord_flip()
   
   
   # Or use some statistics for comparing:
   
   summary_stats <- df_results %>%  dplyr::select(Accuracy, Model) %>%
     group_by(Model) %>%
      dplyr::summarise(across(starts_with("Accuracy"),list(mean = mean,min = min,max = max, sd=sd), .names = "{.col}.{.fn}")) %>% 
     mutate_if(is.numeric, function(x) {round(x, 3)}) 
   

   #knitr::kable()
   
   print(summary_stats)

 }
 
 
 Apply_N_Bayes <- function(df_train_ml,df_test_ml,MissDataMth,spl_meth){
   
 #Make sure the target variable is of a two-class classification problem only
 df_train_ml$Target <- as.factor(df_train_ml$Target)
 df_test_ml$Target <- as.factor(df_test_ml$Target)
 
 
 model <- naiveBayes(Target~., data = df_train_ml) # Train Model
 class(model) # confirm calls of mode;
 
 pred <- predict(model, newdata = df_test_ml) # Create a set of prediction based on trained model on the test data set
 summary(pred)
 p <- as.data.frame(pred) # Convert predictions to data frame
 Pred_Summary <- confusionMatrix(pred, df_test_ml$Target)
 RMSE(as.numeric(df_test_ml$Target),as.numeric(pred)) # Calculate RMSE for Model
 
 Pred_Summary$overall
 Pred_Summary$byClass
 NB_perf <- tibble(Accuracy=Pred_Summary$overall["Accuracy"],
                   AUC="NA",
                   Balanced_Accuracy=Pred_Summary$byClass["Balanced Accuracy"],
                   DImpp="MissDataMth",
                   Detection_Rate=Pred_Summary$byClass["Detection Rate"],
                   F1=Pred_Summary$byClass["F1"],
                   Kappa=Pred_Summary$overall["Kappa"],
                   logLoss="NA",
                   Model="Navie Bayes",
                   Neg_Pred_Value=Pred_Summary$byClass["Neg Pred Value"],
                   Pos_Pred_Value=Pred_Summary$byClass["Pos Pred Value"],
                   prAUC="NA",
                   Precision=Pred_Summary$byClass["Precision"],
                   Recall=Pred_Summary$byClass["Recall"],
                   Resample="NA",
                   Sensitivity=Pred_Summary$byClass["Sensitivity"],
                   Specificity=Pred_Summary$byClass["Specificity"],
                   Split=spl_meth)

 model_perf <<- rbind.data.frame(model_perf,NB_perf)


 Model_Measures <- Model.stats(as.numeric(df_test_ml$Target),as.numeric(pred))
 Model_Measures <- tibble(RMSE = Model_Measures[1], MSE = Model_Measures[2], MAE = Model_Measures[3])
 Model_Measures <- Model_Measures %>% mutate(DImpp = MissDataMth)
 Model_Measures <- Model_Measures %>% mutate(Split = as.character(spl_meth))
 Model_Measures <- Model_Measures %>% mutate(Model = "Navie Bayes")
 print(Model_Measures)
 model_Scores  <<- rbind.data.frame(model_Scores,Model_Measures)
 
 }
 

 XG_boost <- function(df_train_ml,df_test_ml,MissDataMth,spl_meth){
    library(readxl)
    library(tidyverse)
    library(xgboost)
    library(caret)
    #X_train <- xgb.DMatrix(as.matrix(df_train_ml %>% dplyr::select(-Target)))
    X_train <- as.matrix(df_train_ml %>% dplyr::select(-Target))
    y_train <- df_train_ml$Target
    #X_test <- xgb.DMatrix(as.matrix(df_test_ml %>% dplyr::select(-Target)))
    X_test <- as.matrix(df_test_ml %>% dplyr::select(-Target))
    y_test <- df_test_ml$Target
    #Specify cross-validation method and number of folds. Also enable parallel computation#
    
    xgb_trcontrol = trainControl(
       method = "cv",
       number = 4,  
       allowParallel = TRUE,
       verboseIter = FALSE,
       returnData = TRUE
    )
    #This is the grid space to search for the best hyperparameters#
    
    #I am specifing the same parameters with the same values as I did for Python above. The hyperparameters to optimize are found in the website.#
    
    xgbGrid <- expand.grid(nrounds = c(10,20 ),  # this is n_estimators in the python code above
                           max_depth = c(10, 15,20, 25),
                           colsample_bytree = seq(0.5, 0.9, length.out = 5),
                           ## The values below are default values in the sklearn-api. 
                           eta = 0.2,
                           gamma=0,
                           min_child_weight = 1,
                           subsample = 1
    )
    #Finally, train your model#
    
    set.seed(0) 
    
    xgb_model = train(
       X_train, y_train,  
       trControl = xgb_trcontrol,
       tuneGrid = xgbGrid,
       method = "xgbTree"
    )
    
    #Best values for hyperparameters#
    
    xgb_model$bestTune
    xgb_model$results
    # nrounds	max_depth	eta	gamma	colsample_bytree	min_child_weight	subsample
    # 18	200	15	0.1	0	0.8	1	1
    # We see above that we get the same hyperparameter values from both R and Python.
    # 
    # Model evaluation
    predicted = predict(xgb_model, X_test)
    
    RMSE(as.numeric(df_test_ml$Target), as.numeric(predicted))
 }
 
 ##### 3 Way Data Split #####
 #I create three partitions: training (70%), testing (15%) and testing/validation (15%) using
 # Define the partition (e.g. 75% of the data for training)
 trainIndex <- createDataPartition(UCI$T1, p = .70, 
                                   list = FALSE, 
                                   times = 1)
 
 # Split the dataset using the defined partition
 train_data <- UCI[trainIndex, ,drop=FALSE]
 train.rowid <- train_data$rowid
 test_plus_val_data <- UCI[-trainIndex, ,drop=FALSE]
 
 # Define a new partition to split the remaining 25%
 test_plus_val_index <- createDataPartition(test_plus_val_data$T1,
                                            p = .6,
                                            list = FALSE,
                                            times = 1)
 
 # Split the remaining ~25% of the data: 40% (test) and 60% (val)
 test_data <- test_plus_val_data[-test_plus_val_index, ,drop=FALSE]
 test.rowid <- train_data$rowid
 val_data <- test_plus_val_data[test_plus_val_index, ,drop=FALSE]
 val.rowid <- val_data$rowid
 
 
##### Data Exploration #####
 
### Compare imputed data ###
KNN.df$
###### Missing Data #####
 library(mice)
 md.pattern(UCI)
 md.pairs(UCI) 
 
 
 
 
 
 
 
 
 
 
 
 
##### Feature Engineering & Data Splitting #####

#Sum NA values for each row 
UCI <- UCI %>% mutate(MIA = rowSums(is.na(UCI))) 

# Set and Split data: 
set.seed(1255)
# Create column for if the had any heart condition or not
UCI <- UCI %>% mutate(T1_bin = case_when(num == 0 ~ 0,num > 0 ~ 1))
UCI$T1_bin <- as.factor(UCI$T1_bin)
# Convert 0 & 1 to readable labels
UCI <- UCI %>% mutate(T1 = case_when(T1_bin == 1 ~ "Bad", TRUE ~ "Good") %>% as.factor())

UCI <- UCI %>% mutate(T2 = num)

UCI$T2 <- as.factor(UCI$T2)
UCI <- UCI %>% dplyr::select(.,-num,-dataset,-id)

UCI <- tibble::rowid_to_column(UCI)


# ### Split Style 1 - s1
# set.seed(11081995)
# 
# #Subset data by if they have any missing values in the row or not
# #Proportional split the data with NA's and Not
# testdata <- UCI %>% filter(MIA == 0) %>% slice_sample(prop = 0.2)
# testdata_mia <- UCI %>% filter(MIA != 0) %>% slice_sample(prop = 0.2)
# missing_data
# #combine the two test data frames and remove row from UCI by row ID
# s1.testdata <- rbind.data.frame(testdata,testdata_mia)
# s1.traindata <- anti_join(UCI, s1.testdata, by = "rowid")
# 
# #Store Rows that should be in test dt
# s1.train.rowid <- s1.traindata$rowid
# s1.test.rowid <- s1.testdata$rowid
# 
# #Rejoin Split dataset and check for duplicates
# dupp <- rbind.data.frame(s1.traindata,s1.testdata)
# dupp[duplicated(dupp)]
# rm(dupp,testdata,testdata_mia,s1.testdata,s1.traindata)
# #
# #####  Split Style 2 #####  
# ## My attempt to try  have test and training set with equal amounts of imputed data against the target
# 
# #Subset data that has no missing values
#   subset_with_No_missing_Data <- UCI %>% filter(MIA == 0)
#   
#  #Create proportional partion on data based on target value
#   id1 <- createDataPartition(y = subset_with_No_missing_Data$T1, p = 0.7, list = FALSE)
#   
# #create a df of subsetted & partitioned data
#   s2.traindata_1 <- subset_with_No_missing_Data[id1, ]
#    
#   
#   #Subset data by if it has missing vlaues 
#   subset_with_NA_Data <- UCI %>% filter(MIA != 0) 
#   
# #Create proportional partion on data based on target value
#   id2 <- createDataPartition(y = subset_with_NA_Data$T1, p = 0.7, list = FALSE)
#   
#   #create a df of subsetted & partitioned data
#   s2.traindata_2 <- subset_with_NA_Data[id2, ]
#   s2.train.index <- append(id1, id2) 
# # combine the two 
#   s2_traindata <- rbind.data.frame(s2.traindata_1,s2.traindata_2)
#   
# 
# # Remove the data that is in the training set from the UCI df by rowid  
# s2_testdata <- anti_join(UCI, s2_traindata, by = "rowid") 
# 
# #Store Rows that should be in test & train datasets
# train.rowid <- s2_traindata$rowid
# test.rowid <- s2_testdata$rowid
# 
# 
# ## Check for duplication - check to make sure rows are not duplicated acorss the train & test set
# dup <- rbind.data.frame(s2_traindata,s2_testdata)
# dup[duplicated(dup)]
# 
# #Clean-up data that is not required
# rm(dup,subset_with_NA_Data,subset_with_No_missing_Data,id1,id2,s2.traindata_1,s2.traindata_2,s2_testdata,s2_traindata)
# 
# UCI$rowid <- as.factor(UCI$rowid)
# UCI<- UCI %>%
#   mutate_if(is_character, as.factor) %>% 
#   mutate_if(is.logical, as.factor) %>% dplyr::select(.,-MIA)
# ##### Remove Process Columns nd confirm data classes
as.data.frame(sapply(UCI, class))




##### Hot Encode For Data Impute & #####
##use mice and hotenconde to make df for mice
zdf$ca <- as.factor(zdf$ca)
imp.Cols <- zdf %>% select_if(is.numeric)  %>%  colnames()
hot.nums <- zdf %>% dplyr::select(-rowid) %>% select_if(is.numeric)
hot.facts <- zdf %>% dplyr::select(-rowid,-T1) %>% dplyr::select_if(is.factor) 

hf.df <- one_hot(as.data.table(hot.facts))
df  <- cbind(hot.nums,hf.df)


##### Data Imputations #####
######  Row Removal  ######
#Removal of all missing data from test and training sets ###
## Taget = T1, DF - Scaled
# Using Factor labels - Bibart target column removed
# Data with Zero values
df <- UCI %>% dplyr::select(.,-T1_bin) %>% drop_na()
## Using Target Column with Labels
names(df)[names(df) == 'T1'] <- "Target"
ls(df)
as.data.frame(sapply(df, class))
#Remove unwanted Columns
df <- df %>% dplyr::select(.,-T2)


raw.df <- df
rm(df)

###### Mean Replacement  #####
# Replace Zero with NA
df <- UCI
df$chol <- na_if(df$chol, 0)
df$trestbps <- na_if(df$trestbps, 0)
df$thalch  <- na_if(df$thalch, 0)
df$oldpeak <- na_if(df$oldpeak, 0)
# Mean Replacement (numeric) & 
# Frequency replacement(Factors/Characters/logicals) 
df <- df %>% 
   mutate_if(is_character, as.factor) %>% 
   mutate_if(is.logical, as.factor) %>%
   mutate_if(is.numeric, replace_by_mean) %>% 
   mutate_if(is.factor, replace_na_categorical)

## Using Target Column with Labels
names(df)[names(df) == 'T1'] <- "Target"


#Remove unwanted Columns
df <- df %>% dplyr::select(.,-T2,-T1_bin)
mean_place.df <- df

rm(df)


########### IMputed Data with MICE #####
## Zeros replaced with NA
df <- UCI
df$chol <- na_if(df$chol, 0)
df$trestbps <- na_if(df$trestbps, 0)
df$thalch  <- na_if(df$thalch, 0)
df$oldpeak <- na_if(df$oldpeak, 0)

df <- df  %>% dplyr::select(.,-T2,-T1_bin,-rowid)
names(df)[names(df) == 'T1'] <- "Target"
tempData <- mice(df,m=20,maxit=10,seed=500)
summary(tempData)
tempData$meth
completedData <- complete(tempData,1)

xyplot(tempData,ca ~ chol +age,pch=18,cex=1)
densityplot(tempData)
completedData$Target <- as.factor(completedData$Target)
df <- completedData
df$T2 <-UCI$T2
df$rowid <-UCI$rowid
#Remove unwanted Columns
df <- df %>% dplyr::select(.,-T2)

mice.df <- df
rm(completedData,tempData,df)




###### KNN impute ######


df <- UCI  %>% dplyr::select(.,-T2,-T1_bin,-rowid)
names(df)[names(df) == 'T1'] <- "Target"
df$Target <- as.factor(df$Target)
Lbl_encode_only(df)
preProcValues <- preProcess(df,
                            method = c("knnImpute"),
                            k = 100,
                            knnSummary = mean)
imputed_Knn <- predict(preProcValues, df,na.action = na.pass)

procNames <- data.frame(col = names(preProcValues$mean), mean = preProcValues$mean, sd = preProcValues$std)

for(i in procNames$col){imputed_Knn[i] <- imputed_Knn[i]*preProcValues$std[i]+preProcValues$mean[i] 
}

imputed_Knn$rowid <-UCI$rowid

knn.df <- as.data.frame(imputed_Knn)

rm(procNames,df,preProcValues)

#### Run Imputed Data ####
train.RowIDs = train.rowid
test.RowIDs = test.rowid
spl_meth = "S2"
Target = "T1"
set.seed(1255)
number <- 5
repeats <- 5

#### Raw Data - Row Removal
MissDataMth ="Row Removal"
Lbl_encode_Split(raw.df,train.RowIDs,test.RowIDs)
Apply_Caret_Models(df_train_ml,df_test_ml,MissDataMth,spl_meth)
Apply_N_Bayes(df_train_ml,df_test_ml,MissDataMth,spl_meth)
XG_boost(df_train_ml,df_test_ml,MissDataMth,spl_meth)

#### Mean Replacement 
MissDataMth =c("Mean Replacement")
Lbl_encode_Split(mean_place.df,train.RowIDs,test.RowIDs)
XG_boost(df_train_ml,df_test_ml,MissDataMth,spl_meth)
Apply_Caret_Models(df_train_ml,df_test_ml,MissDataMth,spl_meth)
Apply_N_Bayes(df_train_ml,df_test_ml,MissDataMth,spl_meth)

####  Mice Impution
MissDataMth ="MICE Impute"

Lbl_encode_Split(mice.df,train.RowIDs,test.RowIDs)
XG_boost(df_train_ml,df_test_ml,MissDataMth,spl_meth)
Apply_Caret_Models(df_train_ml,df_test_ml,MissDataMth,spl_meth) 
Apply_N_Bayes(df_train_ml,df_test_ml,MissDataMth,spl_meth)

####  KNN Imputation
MissDataMth <- "KNN Impute"
S2_split(knn.df,train.RowIDs,test.RowIDs)
XG_boost(df_train_ml,df_test_ml,MissDataMth,spl_meth)
Apply_Caret_Models(df_train_ml,df_test_ml,MissDataMth,spl_meth)
Apply_N_Bayes(df_train_ml,df_test_ml,MissDataMth,spl_meth)


##### Summary Data #####

model_perf %<>% drop_na
model_Scores 
ls(model_perf)
Summary_Perf <- model_perf %>% dplyr::select(Accuracy,Model,DImpp) %>%
   group_by(DImpp,Model) %>%
   dplyr::summarise(across(starts_with("Accuracy"),list(mean = mean,min = min,max = max, sd=sd), .names = "{.col}.{.fn}")) %>% 
   mutate_if(is.numeric, function(x) {round(x, 3)}) 

####
## Review IMputed data

NA_list <- as.data.frame(colSums(is.na(UCI)))
ls(NA_list)
colnames(NA_list) = c("Count")
NA_list <- NA_list %>% filter(Count > 0)
colnames(NA_list) = c("Col_Name","Count")
dfs <-c(knn.df,raw.df,mice.df,mean_place.df)

chol.df<-UCI%>%dplyr::select(rowid,chol)
chol.df$knn2<-imputed_Knn$chol
chol.df$mice<-mice.df$chol
chol.df$mean<-mean_place.df$chol
chol.df<-chol.df%>%mutate(MIA=rowSums(is.na(chol.df)))
sum.chol.df <- chol.df %>% filter(MIA==1) %>% summary() 

restecg.df<-UCI%>%dplyr::select(rowid,restecg)
restecg.df$knn<-imputed_Knn$restecg
restecg.df$mice<-mice.df$restecg
restecg.df$mean<-mean_place.df$restecg
restecg.df<-restecg.df%>%mutate(MIA=rowSums(is.na(restecg.df)))
sum.restecg.df <- restecg.df%>%filter(MIA==1) %>% summary()



thalch.df<-UCI%>%dplyr::select(rowid,thalch)
thalch.df$knn<-imputed_Knn$thalch
thalch.df$mice<-mice.df$thalch
thalch.df$mean<-mean_place.df$thalch
thalch.df<-thalch.df%>%mutate(MIA=rowSums(is.na(thalch.df)))
thalch.df%>%filter(MIA==1) %>% summary()
sum.thalch.df <- thalch.df%>%filter(MIA==1) %>% summary()

exang.df<-UCI%>%dplyr::select(rowid,exang)
exang.df$knn<-imputed_Knn$exang
exang.df$mice<-mice.df$exang
exang.df$mean<-mean_place.df$exang
exang.df<-exang.df%>%mutate(MIA=rowSums(is.na(exang.df)))
exang.df%>%filter(MIA==1) %>% summary()
sum.exang.df <- exang.df%>%filter(MIA==1) %>% summary()


oldpeak.df<-UCI%>%dplyr::select(rowid,oldpeak)
oldpeak.df$knn<-imputed_Knn$oldpeak
oldpeak.df$mice<-mice.df$oldpeak
oldpeak.df$mean<-mean_place.df$oldpeak
oldpeak.df<-oldpeak.df%>% mutate(MIA=rowSums(is.na(oldpeak.df)))
sum.oldpeak.df <- oldpeak.df %>% filter(MIA==1) %>% summary()



slope.df<-UCI%>%dplyr::select(rowid,slope)
slope.df$knn<-imputed_Knn$slope
slope.df$mice<-mice.df$slope
slope.df$mean<-mean_place.df$slope
slope.df<-slope.df%>%mutate(MIA=rowSums(is.na(slope.df)))
slope.df%>%filter(MIA==1) %>% summary()
sum.slope.df <- slope.df %>%filter(MIA==1) %>% summary()


ca.df<-UCI%>%dplyr::select(rowid,ca)
ca.df$knn<-imputed_Knn$ca
ca.df$mice<-mice.df$ca
ca.df$mean<-mean_place.df$ca
ca.df<-ca.df%>%mutate(MIA=rowSums(is.na(ca.df)))
ca.df%>%filter(MIA==1) %>% summary()
sum.ca.df <- ca.df %>%filter(MIA==1) %>% summary()


thal.df<-UCI%>%dplyr::select(rowid,thal)
thal.df$knn<-imputed_Knn$thal
thal.df$mice<-mice.df$thal
thal.df$mean<-mean_place.df$thal
thal.df<-thal.df%>%mutate(MIA=rowSums(is.na(thal.df)))
thal.df%>%filter(MIA==1) %>% summary()
thal.df <- thal.df%>%filter(MIA==1) %>% summary()

fbs.df<-UCI%>%dplyr::select(rowid,fbs)
fbs.df$knn<-imputed_Knn$fbs
fbs.df$mice<-mice.df$fbs
fbs.df$mean<-mean_place.df$fbs
fbs.df<-fbs.df%>%mutate(MIA=rowSums(is.na(fbs.df)))
fbs.df%>%filter(MIA==1) %>% summary()
fbs.test.row.id <-  fbs.df %>% filter(MIA==1) %>% dplyr::select(rowid)
fbs.train.row.id <-  fbs.df %>% filter(MIA==0) %>% dplyr::select(rowid)




S2_split <-function(fbs.df,fbs.train.row.id,fbs.test.row.id)
df_train_ml$fbs <- as.factor(df_train_ml$fbs)
df_train_ml$Target <- as.numeric(factor(df_train_ml$Target))-1

fbs.model <- train(fbs ~., 
                         data = df_train_ml,
                         method = "xgbTree")
predicted = predict(fbs.model, df_test_ml)

mean.fbs <- fbs.df %>% filter(MIA==1) %>% dplyr::select(mean)
mice.fbs <- fbs.df %>% filter(MIA==1) %>% dplyr::select(mice)
dim(fbs.test.row.id)
RMSE(as.numeric(mean.fbs$mean), as.numeric(predicted))
RMSE(as.numeric(mice.fbs$mice), as.numeric(predicted))


trestbps.df<- UCI %>% dplyr::select(rowid,trestbps)
trestbps.df$knn <- knn.df$trestbps  
trestbps.df$mice <- mice.df$trestbps    
trestbps.df$mean <-mean_place.df$trestbps    
trestbps.df<- trestbps.df %>% mutate(MIA = rowSums(is.na(trestbps.df)))  
trestbps.df %>% filter(MIA == 1) %>% summary()

