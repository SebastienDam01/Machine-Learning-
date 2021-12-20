##############################
## Forward subset selection ##
############################## 

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

errors_FSS <-  matrix(0, 10, 12)
colnames(errors_FSS) <- c("NB", "Knn", "LDA", "QDA", "FDA", "RL", "Ridge", "Lasso", "ElasticNet", "Bagging", "RF", "SVM")

## FSS 

library(leaps)

reg.forward<-regsubsets(as.numeric(factor(y))~., data=phonemes,method='forward',nvmax=100)
plot(reg.forward,scale="bic")

res.forward <- summary(reg.forward)

best <- which.min(res.forward$bic)

best_coef = coef(reg.forward,best)
best_coef

phonemes_FSS <- phonemes[,res.forward$which[best,]]
phonemes_FSS$y = phonemes$y

# K Fold cross validation on Naive Bayes

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  nb.phonemes_FSS <- naive_bayes(y~., data=train_data)
  pred.phonemes_FSS <- predict(nb.phonemes_FSS, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_FSS)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data) 
  errors_FSS[i, c("NB")] <- CV[i]
}

# K Fold cross validation on KNN
# normalize data

# Nested cross-validation
phonemes_FSS$y <- as.numeric(factor(phonemes_FSS$y))
Kmax <- 25
K.sequences <- seq(1, Kmax, by = 1)

K<-10
L<-10

# 1. Divide a dataset into a K cross-validation folds at random

folds.outer <- createFolds(phonemes_FSS$y, k=10)

# 2. For each fold i : outer loop for evaluation of the model with selected hyperparameter
i <- 0
j <- 0
error_fold <- data.frame(matrix(nrow=length(folds.outer), ncol=length(K.sequences)))
colnames(error_fold) <- K.sequences
model.mse.ave <- rep(0,10)
for(outer.fold in folds.outer){
  i<-i+1
  j<-0
  # folds.outer.train is all the data except those in fold i
  folds.outer.train <- folds.outer[-i]
  
  # flatten list of outer.fold and assign data values via the indexes
  outer.fold.indexes <- unlist(outer.fold, use.names = FALSE)
  data.outer.test <- phonemes_FSS[outer.fold.indexes, ]
  
  # flatten list of folds.outer and assign data values via the indexes
  outer.folds.indexes <- unlist(folds.outer.train, use.names = FALSE)
  data.outer.train <- phonemes_FSS[outer.folds.indexes, ]
  
  # randomly split data.outer.train into L folds
  folds.inner <- createFolds(data.outer.train$y, k=10)
  
  # 2.4 For each fold j : inner loop for hyperparameter tuning
  for (inner.fold in folds.inner){
    j<-j+1
    # folds.inner.train is all the data except those in fold i and j
    folds.inner.train <- folds.inner[c(-i, -j)]
    
    # flatten list of inner.fold and assign data values via the indexes
    inner.fold.indexes <- unlist(inner.fold, use.names = FALSE)
    data.inner.val <- phonemes_FSS[inner.fold.indexes, ]
    
    # flatten list of folds and assign data values via the indexes
    inner.folds.indexes <- unlist(folds.inner.train, use.names = FALSE)
    data.inner.train <- phonemes_FSS[inner.folds.indexes, ]
    
    knn.mse.kmin <- rep(0, length(K.sequences))
    # train with each hyperparameter on data.inner.train and evaluate it on data.inner.val
    for(k in K.sequences){
      model <- knn(train=data.inner.train[,1:38], 
                   test = data.inner.val[,1:38], 
                   cl = data.inner.train$y,
                   k=k)
      errc <- 1 - sum(diag(table(data.inner.val$y, model)))/length(model)
      knn.mse.kmin[k] <- knn.mse.kmin[k] + errc
    }
    error_fold[j,] <- knn.mse.kmin
  }
  # calculate the average metrics score over the L folds and choose the best k
  average_error_fold <- colMeans(error_fold)
  best_hyperpameter <- as.integer(names(which.min(average_error_fold)))
  
  # train model with the best hyperparameter on data.outer.train and
  # evaluate its performance on data.outer.test
  model <- knn(train=data.outer.train[,1:38], 
               test = data.outer.test[,1:38], 
               cl = data.outer.train$y,
               k=best_hyperpameter)
  errc <- 1 - sum(diag(table(data.outer.test$y, model)))/nrow(data.outer.test)
  model.mse.ave[i] <- errc
  errors_FSS[i, c("Knn")] <- model.mse.ave[i]
}

phonemes_FSS$y = phonemes$y

# K-Fold cross validation on LDA

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  lda.phonemes_FSS <- lda(y~., data=train_data)
  pred.phonemes_FSS <- predict(lda.phonemes_FSS, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_FSS$class)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data) 
  errors_FSS[i, c("LDA")] <- CV[i]
}

# http://statweb.lsu.edu/faculty/li/teach/exst7152/phoneme-example.html
plot(pred.phonemes_FSS$x,xlab="Dim 1",ylab="Dim 2",col=col[fold[[10]]],pch=col[fold[[10]]],cex=0.7)
legend("topleft",legend=levels(y),pch=1:5,col=1:5,cex=1)

# K Fold cross validation on QDA

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  qda.phonemes_FSS <- qda(y~., data=train_data)
  pred.phonemes_FSS <- predict(qda.phonemes_FSS, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_FSS$class)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data) 
  errors_FSS[i, c("QDA")] <- CV[i]
}

# K Fold cross validation on FDA
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data
  test_data <- phonemes_FSS[fold[[i]], ]
  
  fda.phonemes_FSS <- fda(y~., data=train_data)
  pred.phonemes_FSS <- predict(fda.phonemes_FSS, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_FSS)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data)
  errors_FSS[i, c("FDA")] <- CV[i]
}

# K Fold cross validation on Logistic Regression

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data
  test_data <- phonemes_FSS[fold[[i]], ]
  
  rl.phonemes_FSS <- nnet::multinom(y~., data=train_data)
  pred.phonemes_FSS <- predict(rl.phonemes_FSS, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_FSS)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data)
  errors_FSS[i, c("RL")] <- CV[i]
}

# K Fold cross validation on Ridge

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  x <- as.matrix(train_data[,1:38])
  y <- as.factor(train_data$y)
  
  newX <- model.matrix(~.-y,data=test_data)
  
  X.test <- as.matrix(test_data[,1:38])
  
  ridge.phonemes_FSS <- cv.glmnet(x, y, alpha=0, family='multinomial', nfold=3)
  best.lambda = ridge.phonemes_FSS$lambda.min
  
  pred.phonemes_FSS <- predict(ridge.phonemes_FSS, newx=X.test, s="lambda.min", type="class")
  
  CV[i]<-length(which(pred.phonemes_FSS!=test_data$y))/nrow(X.test)
  errors_FSS[i, c("Ridge")] <- CV[i]
}

# K Fold cross validation on Lasso

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  x <- as.matrix(train_data[,1:38])
  y <- as.factor(train_data$y)
  
  newX <- model.matrix(~.-y,data=test_data)
  
  X.test <- as.matrix(test_data[,1:38])
  
  lasso.phonemes_FSS <- cv.glmnet(x, y, alpha=1, family='multinomial', nfold=3)
  best.lambda = lasso.phonemes_FSS$lambda.min
  
  pred.phonemes_FSS <- predict(lasso.phonemes_FSS, newx=X.test, s="lambda.min", type="class")
  
  CV[i]<-length(which(pred.phonemes_FSS!=test_data$y))/nrow(X.test)
  errors_FSS[i, c("Lasso")] <- CV[i]
}

# K Fold cross validation on Elastic Net

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  train_control = trainControl(method = "cv", number = 10)
  elastic_net_cv <- train(y ~ ., data = train_data, method = "glmnet", trControl = train_control )
  best.alpha <- elastic_net_cv$bestTune[1,1]
  best.lambda <- elastic_net_cv$bestTune[1,2]
  
  x <- as.matrix(train_data[,1:38])
  y <- as.factor(train_data$y)
  
  X.train <- train_data[,1:38]
  y.train <- train_data[, 39]
  X.test <- as.matrix(test_data[,1:38])
  y.test <- test_data[, 39]
  
  elastic_net.phonemes_FSS <- glmnet(X.train, y.train, alpha=best.alpha, lambda=best.lambda, family='multinomial', nfold=3)
  
  pred.phonemes_FSS <- predict(elastic_net.phonemes_FSS, X.test, type="class")
  
  CV[i]<-1-sum(diag(table(y.test,pred.phonemes_FSS)))/nrow(X.test)
  errors_FSS[i, c("ElasticNet")] <- CV[i]
}

# KFold cross validation on Bagging

CV <- rep(0,10)
p <- ncol(phonemes_FSS) - 1
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  bag.phonemes_FSS <- randomForest(as.factor(y) ~ ., train_data, mtry=p)
  pred.phonemes_FSS <- predict(bag.phonemes_FSS, newdata=test_data, type="response")
  
  CV[i]<-1-sum(diag(table(test_data$y,pred.phonemes_FSS)))/nrow(test_data)
  errors_FSS[i, c("Bagging")] <- CV[i]
}

# KFold cross validation on Random Forest

p.sequences <- c(1:38)

# 1. Divide a dataset into a K cross-validation folds at random

folds.outer <- createFolds(phonemes_FSS$y, k=10)

# 2. For each fold i : outer loop for evaluation of the model with selected hyperparameter
i <- 0
j <- 0

error_fold <- data.frame(matrix(nrow=length(folds.outer), ncol=length(p.sequences)))
colnames(error_fold) <- p.sequences
model.mse.ave <- rep(0,10)

for(outer.fold in folds.outer){
  i<-i+1
  j<-0
  # folds.outer.train is all the data except those in fold i
  folds.outer.train <- folds.outer[-i]
  
  # flatten list of outer.fold and assign data values via the indexes
  outer.fold.indexes <- unlist(outer.fold, use.names = FALSE)
  data.outer.test <- phonemes_FSS[outer.fold.indexes, ]
  
  # flatten list of folds.outer and assign data values via the indexes
  outer.folds.indexes <- unlist(folds.outer.train, use.names = FALSE)
  data.outer.train <- phonemes_FSS[outer.folds.indexes, ]
  
  # randomly split data.outer.train into L folds
  folds.inner <- createFolds(data.outer.train$y, k=10)
  
  # 2.4 For each fold j : inner loop for hyperparameter tuning
  for (inner.fold in folds.inner){
    j<-j+1
    # folds.inner.train is all the data except those in fold i and j
    folds.inner.train <- folds.inner[c(-i, -j)]
    
    # flatten list of inner.fold and assign data values via the indexes
    inner.fold.indexes <- unlist(inner.fold, use.names = FALSE)
    data.inner.val <- phonemes_FSS[inner.fold.indexes, ]
    
    # flatten list of folds and assign data values via the indexes
    inner.folds.indexes <- unlist(folds.inner.train, use.names = FALSE)
    data.inner.train <- phonemes_FSS[inner.folds.indexes, ]
    
    rdf.mse.pmin <- rep(0, length(p.sequences))
    # train with each hyperparameter on data.inner.train and evaluate it on data.inner.val
    for(p in 1:length(p.sequences)){
      model <- randomForest(as.factor(y)~., data=data.inner.train, ntree=100, mtry=p.sequences[p])
      pred <- predict(model, newdata=data.inner.val)
      errc <- mean(data.inner.val$y != pred)
      rdf.mse.pmin[p] <- rdf.mse.pmin[p] + errc
    }
    error_fold[j,] <- rdf.mse.pmin
  }
  # calculate the average metrics score over the L folds and choose the best p
  average_error_fold <- colMeans(error_fold)
  best_hyperpameter <- as.integer(names(which.min(average_error_fold)))
  
  # train model with the best hyperparameter on data.outer.train and
  # evaluate its performance on data.outer.test
  model <- randomForest(as.factor(y)~., data=data.outer.train, ntree=100, mtry=best_hyperpameter)
  n <- nrow(data.inner.val)
  pred <- predict(model, newdata=data.outer.test)
  errc <- mean(data.outer.test$y != pred)
  model.mse.ave[i] <- errc
  errors_FSS[i, c("RF")] <- model.mse.ave[i]
}

# SVM

phonemes_FSS$y=as.numeric(factor(phonemes_FSS$y))
C_list<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(C_list)
CV<-rep(0,N)
for(i in 1:N){
  CV[i]<-cross(
    ksvm(as.factor(y)~.,data=phonemes_FSS,type="C-svc",kernel="rbfdot",C=C_list[i],cross=10)
  )
}

plot(C_list, CV, pch="o", type="b", log="x", xlab="C", ylab="Taux d'erreur", main="Taux d'erreur en fonction de C")

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_FSS$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_FSS[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_FSS[fold[[i]], ]
  
  svm.phonemes_FSS <-ksvm(as.factor(y)~ .,data=train_data,type="C-svc",kernel="rbfdot",C=1)
  pred.phonemes_FSS=predict(svm.phonemes_FSS,test_data[1:38])
  
  CV[i]<-1 - sum(diag(table(test_data$y, pred.phonemes_FSS)))/nrow(test_data)
  errors_FSS[i, c("SVM")] <- CV[i]
}

phonemes_FSS$y = phonemes$y

# Plot

# Plot interval errors_FSS 
plot.cv.error <- function(data, x.title="x"){
  ic.error.bar <- function(x, lower, upper, length=0.1){ 
    arrows(x, upper, x, lower, angle=90, code=3, length=length, col='red')
  }
  stderr <- function(x) sd(x)/sqrt(length(x))
  # mean and standard errors_FSS
  means.errs <- colMeans(data)
  std.errs <- apply(data, 2, stderr)
  # plotting  
  x.values <- 1:ncol(data)
  
  ggplot(data.frame(model=x.title, mean=as.vector(means.errs)),aes(x=model, y=mean))+
    geom_point(aes(x=model, y=mean))+
    ggtitle("Intervalle de confiance des erreurs en fonction du prédicteur FSS") +
    xlab("Prédicteurs")+
    ylab("Intervalles de confiance des erreurs")+
    geom_errorbar(aes(ymin=mean - 1.6*std.errs, ymax= mean + 1.6*std.errs), width=.2,
                  position=position_dodge(.9)) +
    theme(plot.title = element_text(hjust = 0.5),axis.text.x=element_text(angle=60, hjust=1))
}

plot.cv.error(errors_FSS, c("Naive Bayes", "KNN", "LDA", "QDA", "FDA", "RL", "Ridge", "Lasso", "ElasticNet", "Bagging", "RF", "SVM"))

#boxplot(errors_FSS,main="Boîtes à moustache des erreurs selon le prédicteur", xlab='Prédicteurs', ylab="Pourcentage d'erreur")