# Fonctions de prédiction aussi performantes que possible 
# Pour les données de location de vélos, il faudra également 
# déterminer quelles sont les variables qui influent le plus sur le nombre de locations,
# et analyer le sens de cette influence.

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

setwd("C:/Users/azzer/Desktop/Travail/utc/A21/sy19/TP/TP10")

letter <- read.table(file="data/letters_train.txt", sep="")

missing_val<-data.frame(apply(letter,2,function(x){sum(is.na(x))}))
names(missing_val)[1]='missing_val'
missing_val

table(letter$Y)


## load
data_class <- letter

## resumer
summary(data_class)
head(data_class)

## s?parattion en essemble d'entrainement et de validation
n <- nrow(data_class)
p <- ncol(data_class)-1
nb.train <- round(2*n/3)
nb.test <- n - nb.train
# seed
set.seed(123) # the Hardy-Ramanujan number
# Training/Testing data
train <- sample(1:n, nb.train)
data_class.train <- data_class[train,]
data_class.test <- data_class[-train,]

missing_val<-data.frame(apply(letter,2,function(x){sum(is.na(x))}))
names(missing_val)[1]='missing_val'

errors <-  matrix(0, 10, 19)
colnames(errors) <- c("KNN", "LDA", "QDA", "FDA", "NB", "RL", "Ridge", "Lasso", "ElasticNet", "TREE", "pTREE", "bag", "RF", "mtry", "SVM", "GMM", "GMM_EDDA", "GAM", "nn")




# K Fold cross validation on KNN
# normalize data

# Nested cross-validation
data_class$Y <- as.numeric(factor(data_class$Y))
Kmax <- 25
K.sequences <- seq(1, Kmax, by = 1)

K<-10
L<-10

# 1. Divide a dataset into a K cross-validation folds at random

folds.outer <- createFolds(data_class$Y, k=10)

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
  data.outer.test <- data_class[outer.fold.indexes, ]
  
  # flatten list of folds.outer and assign data values via the indexes
  outer.folds.indexes <- unlist(folds.outer.train, use.names = FALSE)
  data.outer.train <- data_class[outer.folds.indexes, ]
  
  # randomly split data.outer.train into L folds
  folds.inner <- createFolds(data.outer.train$Y, k=10)
  
  # 2.4 For each fold j : inner loop for hyperparameter tuning
  for (inner.fold in folds.inner){
    j<-j+1
    # folds.inner.train is all the data except those in fold i and j
    folds.inner.train <- folds.inner[c(-i, -j)]
    
    # flatten list of inner.fold and assign data values via the indexes
    inner.fold.indexes <- unlist(inner.fold, use.names = FALSE)
    data.inner.val <- data_class[inner.fold.indexes, ]
    
    # flatten list of folds and assign data values via the indexes
    inner.folds.indexes <- unlist(folds.inner.train, use.names = FALSE)
    data.inner.train <- data_class[inner.folds.indexes, ]
    
    knn.mse.kmin <- rep(0, length(K.sequences))
    # train with each hyperparameter on data.inner.train and evaluate it on data.inner.val
    for(k in K.sequences){
      model <- knn(train=data.inner.train[,2:17], 
                   test = data.inner.val[,2:17], 
                   cl = data.inner.train$Y,
                   k=k)
      errc <- 1 - sum(diag(table(data.inner.val$Y, model)))/length(model)
      knn.mse.kmin[k] <- knn.mse.kmin[k] + errc
    }
    error_fold[j,] <- knn.mse.kmin
  }
  # calculate the average metrics score over the L folds and choose the best k
  average_error_fold <- colMeans(error_fold)
  best_hyperpameter <- as.integer(names(which.min(average_error_fold)))
  print(best_hyperpameter)
  # train model with the best hyperparameter on data.outer.train and
  # evaluate its performance on data.outer.test
  model <- knn(train=data.outer.train[,2:17], 
               test = data.outer.test[,2:17], 
               cl = data.outer.train$Y,
               k=best_hyperpameter)
  errc <- 1 - sum(diag(table(data.outer.test$Y, model)))/nrow(data.outer.test)
  model.mse.ave[i] <- errc
  errors[i, c("KNN")] <- model.mse.ave[i]
}

data_class$Y = letter$Y


# K Fold cross validation on LDA

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  lda <- lda(Y ~ ., data=train_data)
  pred.lda <- predict(lda, newdata=test_data)
  perf.lda <- table(test_data$Y, pred.lda$class)
  
  CV[i]<-1-sum(diag(perf.lda))/nrow(test_data) 
  errors[i, c("LDA")] <- CV[i]
}

# K Fold cross validation on QDA

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  qda <- qda(Y ~ ., data=train_data)
  pred.qda <- predict(lda, newdata=test_data)
  perf.qda <- table(test_data$Y, pred.qda$class)
  
  CV[i]<-1-sum(diag(perf.qda))/nrow(test_data) 
  errors[i, c("QDA")] <- CV[i]
}

# K Fold cross validation on FDA
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data
  test_data <- data_class[fold[[i]], ]
  
  fda <- fda(Y~., data=train_data)
  pred.fda <- predict(fda, newdata=test_data)
  perf.fda <- table(test_data$Y, pred.fda)
  
  CV[i]<-1-sum(diag(perf.fda))/nrow(test_data)
  errors[i, c("FDA")] <- CV[i]
}


# K Fold cross validation on NB

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  naive <- naive_bayes(as.factor(Y) ~ . ,data=train_data)
  pred.naive <- predict(naive, newdata=test_data)
  perf.naive <- table(test_data$Y, pred.naive)
  
  CV[i]<-1-sum(diag(perf.naive))/nrow(test_data) 
  errors[i, c("NB")] <- CV[i]
}

# K Fold cross validation on Logistic Regression

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data
  test_data <- data_class[fold[[i]], ]
  
  rl <- nnet::multinom(Y~., data=train_data)
  pred.rl <- predict(rl, newdata=test_data)
  perf.rl <- table(test_data$Y, pred.rl)
  
  CV[i]<-1-sum(diag(perf.rl))/nrow(test_data)
  errors[i, c("RL")] <- CV[i]
}

############### ridge, lasso, elastic net ###############

# K Fold cross validation on Ridge

CV <- rep(0,10)
best.lambda.ridge.CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  x <- as.matrix(train_data[,2:17])
  y <- as.factor(train_data$Y)
  
  newX <- model.matrix(~.-Y,data=test_data)
  
  X.test <- as.matrix(test_data[,2:17])
  
  ridge <- cv.glmnet(x, y, alpha=0, family='multinomial', nfold=3)
  best.lambda = ridge$lambda.min
  best.lambda.ridge.CV[i] <- best.lambda
  
  pred.ridge <- predict(ridge, newx=X.test, s="lambda.min", type="class")
  
  CV[i]<-length(which(pred.ridge!=test_data$Y))/nrow(X.test)
  errors[i, c("Ridge")] <- CV[i]
}

# K Fold cross validation on Lasso

CV <- rep(0,10)
best.lambda.lasso.CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  x <- as.matrix(train_data[,2:17])
  y <- as.factor(train_data$Y)
  
  newX <- model.matrix(~.-Y,data=test_data)
  
  X.test <- as.matrix(test_data[,2:17])
  
  lasso <- cv.glmnet(x, y, alpha=1, family='multinomial', nfold=3)
  best.lambda = lasso$lambda.min
  best.lambda.lasso.CV[i] <- best.lambda
  
  pred.lasso <- predict(lasso, newx=X.test, s="lambda.min", type="class")
  
  CV[i]<-length(which(pred.lasso!=test_data$Y))/nrow(X.test)
  errors[i, c("Lasso")] <- CV[i]
}

# K Fold cross validation on Elastic Net

best.alpha.EL.CV <- rep(0,10)
best.lambda.EL.CV <- rep(0,10)
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  train_control = trainControl(method = "cv", number = 10)
  elastic_net_cv <- train(Y ~ ., data = train_data, method = "glmnet", trControl = train_control)
  best.alpha <- elastic_net_cv$bestTune[1,1]
  best.lambda <- elastic_net_cv$bestTune[1,2]
  
  best.alpha.EL.CV[i] <- best.alpha
  best.lambda.EL.CV[i] <- best.lambda
  
  x <- as.matrix(train_data[,2:17])
  y <- as.factor(train_data$Y)
  
  X.train <- train_data[,2:17]
  y.train <- train_data[,1]
  X.test <- as.matrix(test_data[,2:17])
  y.test <- test_data[,1]
  
  elastic_net <- glmnet(X.train, y.train, alpha=best.alpha, lambda=best.lambda, family='multinomial', nfold=3)
  
  pred.elastic_net <- predict(elastic_net, X.test, type="class")
  
  CV[i]<-1-sum(diag(table(y.test, pred.elastic_net)))/nrow(X.test)
  errors[i, c("ElasticNet")] <- CV[i]
}



############### TREE ###############

CV1 <- rep(0,10)
CV2 <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  fit <- rpart(Y ~ ., data = train_data, method="class", parms = list(split = 'gini'))
  pruned_tree<-prune(fit,cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
  # TODO : tester avec autre indicateurs
  
  pred.tree<-predict(fit,newdata=test_data,type ='class')
  perf.tree<-table(test_data$Y, pred.tree)
  
  pred.ptree<-predict(pruned_tree,newdata=test_data,type ='class')
  perf.ptree<-table(test_data$Y, pred.ptree)
  
  CV1[i]<-1-sum(diag(perf.tree))/nrow(test_data) 
  CV2[i]<-1-sum(diag(perf.ptree))/nrow(test_data) 
  errors[i, c("TREE")] <- CV1[i]
  errors[i, c("pTREE")] <- CV2[i]
}

############### FOREST ###############

### BAGING

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  bag.class <- randomForest(as.factor(Y) ~., data=data_class, subset=train, mtry=p)
  pred.bag <- predict(bag.class, newdata=test_data, type="response")
  perf.bag <- table(test_data$Y, pred.bag)
  
  CV[i]<-1-sum(diag(perf.bag))/nrow(test_data) 
  errors[i, c("bag")] <- CV[i]
}

### RF

CV <- rep(0,10)
pb = txtProgressBar(min = 0, max = 10, initial = 0) 
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  setTxtProgressBar(pb,i)
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  my_mtry <- sqrt(p)
  
  RF.class <- randomForest(as.factor(Y) ~., data=data_class, subset=train, mtry=p)
  pred.RF <- predict(RF.class, newdata=test_data, type="response")
  perf.RF <- table(test_data$Y, pred.RF)
  
  CV[i]<-1-sum(diag(perf.RF))/nrow(test_data) 
  errors[i, c("RF")] <- CV[i]
}
close(pb)

# pour l'analyse explo
my_mtry <- sqrt(p)
RF.class <- randomForest(as.factor(Y) ~., data=data_class, subset=train, mtry=p, importance = TRUE)
pred.RF <- predict(RF.class, newdata=data_class.test, type="response")
perf.RF <- table(data_class.test$Y, pred.RF)
varImpPlot(RF.class)


### find best param mtry

CV <- rep(0,10)
bestmtry.CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  bestmtry.CV[i] <- tuneRF(data_class.train[,2:17], as.factor(data_class.train$Y), stepFactor=1.5, improve=1e-5, ntree=500)
  
  RF.class <- randomForest(as.factor(Y) ~., data=data_class, subset=train, mtry=bestmtry.CV[i])
  pred.RF <- predict(RF.class, newdata=test_data, type="response")
  perf.RF <- table(test_data$Y, pred.RF)
  
  CV[i]<-1-sum(diag(perf.RF))/nrow(test_data) 
  errors[i, c("mtry")] <- CV[i]
}


############### GMM ###############

class <- data_class$Y
X <- data_class[,2:17]

data_class_gmm <- MclustDA(X, class)
cv <- cvMclustDA(data_class_gmm)
errors[, c("GMM")] <- 1 - sum(diag(table(cv$classification, factor(data_class$Y))))/nrow(data_class)

#x = c("X10", "X1", "X2", "X15", "X16", "X14", "X11", "X8", "X7", "X6" ,"X5")

# general covariance structure selected by BIC
letterMclustDA <- MclustDA(X, class, modelType = "EDDA")
summary(letterMclustDA)#, parameters = TRUE)
cv <- cvMclustDA(letterMclustDA, nfold = 10)
cv$ce
errors[, c("GMM_EDDA")] <- 1 - sum(diag(table(cv$classification, factor(data_class$Y))))/nrow(data_class)

#plot(letterMclustDA)

############### SVM ###############

#x = c("Y", "X10", "X1", "X2", "X15", "X16", "X14", "X11", "X8", "X7", "X6" ,"X5", "X4", "X13", "X12")

data_class$Y=as.numeric(factor(data_class$Y))

# kernel = c("rbfdot", "tanhdot", "polydot", "laplacedot")

C_list<-c(0.001,0.01,0.1,1,10,100,1000)
N<-length(C_list)
CV<-rep(0,N)
pb = txtProgressBar(min = 0, max = N, initial = 0) 
for(i in 1:N){
  setTxtProgressBar(pb,i)
  CV[i]<-cross(
    ksvm(as.factor(Y)~.,data=data_class,type="C-svc",kernel="polydot",C=C_list[i],cross=10)
  )
  print(CV[i])
}
plot(C_list, CV, pch="o", type="b", log="x", xlab="C", ylab="Taux d'erreur", main="Taux d'erreur en fonction de C")
close(pb)

# best_C = c("10", "0.01", "1", "100")

pb = txtProgressBar(min = 0, max = 10, initial = 0) 
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  setTxtProgressBar(pb,i)
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  svmc <- ksvm(as.factor(Y) ~ ., type = "C-svc", data=train_data, kernel="laplacedot",C=100)
  ypred <- predict(svmc, test_data[2:17])
  perf.svmc <- table(test_data$Y, ypred)
  
  CV[i]<-1-sum(diag(perf.svmc))/nrow(test_data)
  errors[i, c("SVM")] <- CV[i]
}

close(pb)


data_class$Y = letter$Y

# GAM

data_class$Y=as.numeric(factor(data_class$Y))

## reassign values to y from 0 to 4 to fit multinom condition on gam
data_class$Y = data_class$Y -1

pb = txtProgressBar(min = 0, max = 10, initial = 0) 
CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  setTxtProgressBar(pb,i)
  train_data <- data_class[-fold[[1]], ]
  test_data <- data_class[fold[[1]], ]
  
  X.train <- train_data[,1:17]
  X.test <- test_data[,1:17]
  

  fit <- gam(Y ~ s(X1) + s(X2) + s(X3) + s(X4) + s(X5) + s(X6) + s(X7) + s(X8) + s(X9) 
             + s(X10) + s(X11) + s(X12) + s(X13) + s(X14) + s(X15) + s(X16)
             , data = X.train, family = gaussian)
  
  pred <- predict(fit, newdata = X.test, type="response")
  pc <- apply(pred,1,function(x) which(max(x)==x)[1])-1
  mean(pc == X.test$Y)
  
  CV[i]<-1 - mean(pc == X.test$Y)
  errors[i, c("GAM")] <- CV[i]
  
}

close(pb)

data_class$Y = letter$Y



####################### NN ####################

CV <- rep(0,10)
#Creating folds
fold <- unname(createFolds(data_class$Y, k=10))
for(i in (1:10)){
  #Training data
  train_data <- data_class[-fold[[i]], ]
  #Creating test data 
  test_data <- data_class[fold[[i]], ]
  
  nn <- nnet(as.factor(Y) ~ ., data=train_data, size=5, linout = TRUE)
  pred.nn <- predict(nn, newdata=test_data, type="class")
  perf.nn <- table(test_data$Y, pred.nn)
  
  CV[i] <- 1-sum(diag(perf.nn))/nrow(test_data)
  errors[i, c("nn")] <- CV[i]
}

#############################################################

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
    xlab("Pr�dicteurs")+
    ylab("Intervalles de confiance des erreurs")+
    geom_errorbar(aes(ymin=mean - 1.6*std.errs, ymax= mean + 1.6*std.errs), width=.2,
                  position=position_dodge(.9)) +
    theme(plot.title = element_text(hjust = 0.5),axis.text.x=element_text(angle=60, hjust=1))
}

plot.cv.error(errors, colnames(errors) )

colMeans(errors)
