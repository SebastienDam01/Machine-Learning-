##########################################
## Variance importance by Random Forest ##
##########################################

library(factoextra)
library(lattice)
library(caret)
library(randomForest)
library(rlang)
library(ggplot2)
library(naivebayes)
library(FNN)
library(MASS)
library(e1071)
library(proxy)
library(nnet) #RL
library(mda) #FDA
library(glmnet) #Ridge
library(randomForest)
library(kernlab)
library(mclust) # GMM
library(mgcv) # GAM

setwd("C:/Users/sebas/Desktop/Supérieur/Branche/GI05/SY19/TP/TP 10")

phonemes <- read.table(file="data/parole_train.txt", sep="")

errors_trees <-  matrix(0, 10, 12)
colnames(errors_trees) <- c("NB", "Knn", "LDA", "QDA", "FDA", "RL", "Ridge", "Lasso", "ElasticNet", "Bagging", "RF", "SVM")

best_hyperpameter = 9 # from Phonemes_BSS
RF = randomForest(as.factor(y)~., data=phonemes, ntree=100, mtry=best_hyperpameter)
varImpPlot(RF)
