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

res.pca <- prcomp(phonemes[, 1:256], scale = TRUE)
phonemes_pc <- res.pca$x
phonemes_pc <- phonemes_pc[,1:50]
phonemes_pc <- as.data.frame(phonemes_pc)

phonemes_pc$y = phonemes$y

y = factor(phonemes_pc$y)
col=as.numeric(factor(y))

errors <-  matrix(0, 10, 12)
colnames(errors) <- c("NB", "Knn", "LDA", "QDA", "FDA", "RL", "Ridge", "Lasso", "ElasticNet", "Bagging", "RF", "SVM")

# K Fold cross validation on Naive Bayes

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  nb.phonemes_pc <- naive_bayes(y~., data=train_data)
  pred.phonemes_pc <- predict(nb.phonemes_pc, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_pc)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data) 
  errors[i, c("NB")] <- CV[i]
}

# K Fold cross validation on KNN

# Nested cross-validation
phonemes_pc$y <- as.numeric(factor(phonemes_pc$y))
Kmax <- 25
K.sequences <- seq(1, Kmax, by = 1)

K<-10
L<-10

# 1. Divide a dataset into a K cross-validation folds at random

folds.outer <- createFolds(phonemes_pc$y, k=10)

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
  data.outer.test <- phonemes_pc[outer.fold.indexes, ]
  
  # flatten list of folds.outer and assign data values via the indexes
  outer.folds.indexes <- unlist(folds.outer.train, use.names = FALSE)
  data.outer.train <- phonemes_pc[outer.folds.indexes, ]
  
  # randomly split data.outer.train into L folds
  folds.inner <- createFolds(data.outer.train$y, k=10)
  
  # 2.4 For each fold j : inner loop for hyperparameter tuning
  for (inner.fold in folds.inner){
    j<-j+1
    # folds.inner.train is all the data except those in fold i and j
    folds.inner.train <- folds.inner[c(-i, -j)]
    
    # flatten list of inner.fold and assign data values via the indexes
    inner.fold.indexes <- unlist(inner.fold, use.names = FALSE)
    data.inner.val <- phonemes_pc[inner.fold.indexes, ]
    
    # flatten list of folds and assign data values via the indexes
    inner.folds.indexes <- unlist(folds.inner.train, use.names = FALSE)
    data.inner.train <- phonemes_pc[inner.folds.indexes, ]
    
    knn.mse.kmin <- rep(0, length(K.sequences))
    # train with each hyperparameter on data.inner.train and evaluate it on data.inner.val
    for(k in K.sequences){
      model <- knn(train=data.inner.train[,1:50], 
                   test = data.inner.val[,1:50], 
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
  model <- knn(train=data.outer.train[,1:50], 
               test = data.outer.test[,1:50], 
               cl = data.outer.train$y,
               k=best_hyperpameter)
  errc <- 1 - sum(diag(table(data.outer.test$y, model)))/nrow(data.outer.test)
  model.mse.ave[i] <- errc
  errors[i, c("Knn")] <- model.mse.ave[i]
}

phonemes_pc$y = phonemes$y

# K-Fold cross validation on LDA

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  lda.phonemes_pc <- lda(y~., data=train_data)
  pred.phonemes_pc <- predict(lda.phonemes_pc, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_pc$class)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data) 
  errors[i, c("LDA")] <- CV[i]
}

# http://statweb.lsu.edu/faculty/li/teach/exst7152/phoneme-example.html
plot(pred.phonemes_pc$x,xlab="Dim 1",ylab="Dim 2",col=col[fold[[10]]],pch=col[fold[[10]]],cex=0.7)
legend("topleft",legend=levels(y),pch=1:5,col=1:5,cex=1)

# K Fold cross validation on QDA

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  qda.phonemes_pc <- qda(y~., data=train_data)
  pred.phonemes_pc <- predict(qda.phonemes_pc, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_pc$class)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data) 
  errors[i, c("QDA")] <- CV[i]
}

# K Fold cross validation on FDA
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data
  test_data <- phonemes_pc[fold[[i]], ]
  
  fda.phonemes_pc <- fda(y~., data=train_data)
  pred.phonemes_pc <- predict(fda.phonemes_pc, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_pc)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data)
  errors[i, c("FDA")] <- CV[i]
}

# K Fold cross validation on Logistic Regression

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data
  test_data <- phonemes_pc[fold[[i]], ]
  
  rl.phonemes_pc <- multinom(y~., data=train_data)
  pred.phonemes_pc <- predict(rl.phonemes_pc, newdata=test_data)
  
  perf <- table(test_data$y, pred.phonemes_pc)
  
  CV[i]<-1-sum(diag(perf))/nrow(test_data)
  errors[i, c("RL")] <- CV[i]
}

# K Fold cross validation on Ridge

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  x <- as.matrix(train_data[,1:50])
  y <- as.factor(train_data$y)
  
  newX <- model.matrix(~.-y,data=test_data)
  
  X.test <- as.matrix(test_data[,1:50])
  
  ridge.phonemes_pc <- cv.glmnet(x, y, alpha=0, family='multinomial', nfold=3)
  best.lambda = ridge.phonemes_pc$lambda.min
  
  pred.phonemes_pc <- predict(ridge.phonemes_pc, newx=X.test, s="lambda.min", type="class")
  
  CV[i]<-length(which(pred.phonemes_pc!=test_data$y))/nrow(X.test)
  errors[i, c("Ridge")] <- CV[i]
}

# K Fold cross validation on Lasso

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  x <- as.matrix(train_data[,1:50])
  y <- as.factor(train_data$y)
  
  newX <- model.matrix(~.-y,data=test_data)
  
  X.test <- as.matrix(test_data[,1:50])
  
  lasso.phonemes_pc <- cv.glmnet(x, y, alpha=1, family='multinomial', nfold=3)
  best.lambda = lasso.phonemes_pc$lambda.min
  
  pred.phonemes_pc <- predict(lasso.phonemes_pc, newx=X.test, s="lambda.min", type="class")
  
  CV[i]<-length(which(pred.phonemes_pc!=test_data$y))/nrow(X.test)
  errors[i, c("Lasso")] <- CV[i]
}

# K Fold cross validation on Elastic Net

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  train_control = trainControl(method = "cv", number = 10)
  elastic_net_cv <- train(y ~ ., data = train_data, method = "glmnet", trControl = train_control )
  best.alpha <- elastic_net_cv$bestTune[1,1]
  best.lambda <- elastic_net_cv$bestTune[1,2]
  
  x <- as.matrix(train_data[,1:50])
  y <- as.factor(train_data$y)
  
  X.train <- train_data[,1:50]
  y.train <- train_data[, 51]
  X.test <- as.matrix(test_data[,1:50])
  y.test <- test_data[, 51]
  
  elastic_net.phonemes_pc <- glmnet(X.train, y.train, alpha=best.alpha, lambda=best.lambda, family='multinomial', nfold=3)
  
  pred.phonemes_pc <- predict(elastic_net.phonemes_pc, X.test, type="class")
  
  CV[i]<-1-sum(diag(table(y.test,pred.phonemes_pc)))/nrow(X.test)
  errors[i, c("ElasticNet")] <- CV[i]
}

# KFold cross validation on Bagging

CV <- rep(0,10)
p <- ncol(phonemes_pc) - 1
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  bag.phonemes_pc <- randomForest(as.factor(y) ~ ., train_data, mtry=p)
  pred.phonemes_pc <- predict(bag.phonemes_pc, newdata=test_data, type="response")
  
  CV[i]<-1-sum(diag(table(test_data$y,pred.phonemes_pc)))/nrow(test_data)
  errors[i, c("Bagging")] <- CV[i]
}

# KFold cross validation on Random Forest

p.sequences <- c(1:50)

# 1. Divide a dataset into a K cross-validation folds at random

folds.outer <- createFolds(phonemes_pc$y, k=10)

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
  data.outer.test <- phonemes_pc[outer.fold.indexes, ]
  
  # flatten list of folds.outer and assign data values via the indexes
  outer.folds.indexes <- unlist(folds.outer.train, use.names = FALSE)
  data.outer.train <- phonemes_pc[outer.folds.indexes, ]
  
  # randomly split data.outer.train into L folds
  folds.inner <- createFolds(data.outer.train$y, k=10)
  
  # 2.4 For each fold j : inner loop for hyperparameter tuning
  for (inner.fold in folds.inner){
    j<-j+1
    # folds.inner.train is all the data except those in fold i and j
    folds.inner.train <- folds.inner[c(-i, -j)]
    
    # flatten list of inner.fold and assign data values via the indexes
    inner.fold.indexes <- unlist(inner.fold, use.names = FALSE)
    data.inner.val <- phonemes_pc[inner.fold.indexes, ]
    
    # flatten list of folds and assign data values via the indexes
    inner.folds.indexes <- unlist(folds.inner.train, use.names = FALSE)
    data.inner.train <- phonemes_pc[inner.folds.indexes, ]
    
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
  errors[i, c("RF")] <- model.mse.ave[i]
}

phonemes_pc$y=as.numeric(factor(phonemes_pc$y))
C_list<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(C_list)
CV<-rep(0,N)
for(i in 1:N){
  CV[i]<-cross(
    ksvm(as.factor(y)~.,data=phonemes_pc,type="C-svc",kernel="rbfdot",C=C_list[i],cross=10)
  )
}

plot(C_list, CV, pch="o", type="b", log="x", xlab="C", ylab="Taux d'erreur", main="Taux d'erreur en fonction de C")

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- phonemes_pc[-fold[[i]], ]
  #Creating test data 
  test_data <- phonemes_pc[fold[[i]], ]
  
  svm.phonemes_pc <-ksvm(as.factor(y)~ .,data=train_data,type="C-svc",kernel="rbfdot",C=1)
  pred.phonemes_pc=predict(svm.phonemes_pc,test_data[1:50])
  
  CV[i]<-1 - sum(diag(table(test_data$y, pred.phonemes_pc)))/nrow(test_data)
  errors[i, c("SVM")] <- CV[i]
}

phonemes_pc$y = phonemes$y

# GMM

phonemes_gmm <- Mclust(phonemes_pc[,1:50])
summary(phonemes_gmm)

plot(phonemes_gmm, "BIC")

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
train_data <- phonemes_pc[-fold[[1]], ]
test_data <- phonemes_pc[fold[[1]], ]

X.train <- train_data[,1:50]
y.train <- train_data[, 51]
X.test <- as.matrix(test_data[,1:50])
y.test <- test_data[, 51]

phonemesMclustDA <- MclustDA(X.train, y.train)

summary(phonemesMclustDA, newdata = X.test, newclass = y.test)

# GAM

phonemes_pc$y=as.numeric(factor(phonemes_pc$y))
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(phonemes_pc$y, k=10))
train_data <- phonemes_pc[-fold[[1]], ]
test_data <- phonemes_pc[fold[[1]], ]

X.train <- train_data[,1:51]
X.test <- test_data[,1:51]

# fit <- gam(y ~ s(PC1) + s(PC2) + s(PC3) + s(PC4) + s(PC5) + s(PC6) + s(PC7) + s(PC8) + s(PC9) + s(PC10) 
#            + s(PC11) + s(PC12) + s(PC13) + s(PC14) + s(PC15) + s(PC16) + s(PC17) + s(PC18) + s(PC19) + s(PC20)
#            + s(PC21) + s(PC22) + s(PC23) + s(PC24) + s(PC25) + s(PC26) + s(PC27) + s(PC28) + s(PC29) + s(PC30)
#            + s(PC31) + s(PC32) + s(PC33) + s(PC34) + s(PC35) + s(PC36) + s(PC37) + s(PC38) + s(PC39) + s(PC40)
#            + s(PC41) + s(PC42) + s(PC43) + s(PC44) + s(PC45) + s(PC46) + s(PC47) + s(PC48) + s(PC49) + s(PC50)
#            , data = X.train, family = gaussian)

fit <- gam(y ~ s(PC1) + s(PC2), data = X.train, family = gaussian)
pred <- predict(fit, newdata = X.test)
mean(pred == X.test$y)
phonemes_pc$y = phonemes$y

# Plot

# Plot interval errors 
plot.cv.error <- function(data, x.title="x"){
  ic.error.bar <- function(x, lower, upper, length=0.1){ 
    arrows(x, upper, x, lower, angle=90, code=3, length=length, col='red')
  }
  stderr <- function(x) sd(x)/sqrt(length(x))
  # mean and standard errors
  means.errs <- colMeans(data)
  std.errs <- apply(data, 2, stderr)
  # plotting  
  x.values <- 1:ncol(data)
  
  ggplot(data.frame(model=x.title, mean=as.vector(means.errs)),aes(x=model, y=mean))+
    geom_point(aes(x=model, y=mean))+
    ggtitle("Intervalle de confiance des erreurs en fonction du prédicteur") +
    xlab("Prédicteurs")+
    ylab("Intervalles de confiance des erreurs")+
    geom_errorbar(aes(ymin=mean - 1.6*std.errs, ymax= mean + 1.6*std.errs), width=.2,
                  position=position_dodge(.9)) +
    theme(plot.title = element_text(hjust = 0.5),axis.text.x=element_text(angle=60, hjust=1))
}

plot.cv.error(errors, c("Naive Bayes", "KNN", "LDA", "QDA", "FDA", "RL", "Ridge", "Lasso", "ElasticNet", "Bagging", "RF", "SVM"))

#boxplot(errors,main="Boîtes à moustache des erreurs selon le prédicteur", xlab='Prédicteurs', ylab="Pourcentage d'erreur")
