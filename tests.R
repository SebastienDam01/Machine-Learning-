# Fonctions de prédiction aussi performantes que possible 
# Pour les données de location de vélos, il faudra également 
# déterminer quelles sont les variables qui influent le plus sur le nombre de locations,
# et analyer le sens de cette influence.

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

# Correlation avec y
c <- cor(data_class[,2:p], as.numeric(data_class$Y))
plot(c)

# variables corrélés entre elle
c <- cor(data_class)
best <- c > 0.1
c[best]


library(MASS)

lda <- lda(Y ~ . , data=data_class.train)
pred.lda <- predict(lda, newdata=data_class.test)
perf.lda <- table(data_class.test$Y, pred.lda$class)
1-sum(diag(perf.lda))/nb.test

qda <- qda(Y ~ . , data=data_class.train)
pred.lda <- predict(lda, newdata=data_class.test)
perf.lda <- table(data_class.test$Y, pred.lda$class)
1-sum(diag(perf.lda))/nb.test

library(naivebayes)

naive <- naive_bayes(Y ~ . ,data=data_class.train)
pred.naive <- predict(naive, newdata=data_class.test)
perf.naive <- table(data_class.test$Y, pred.naive)
1-sum(diag(perf.naive))/nb.test


############### TREE ###############

library(rpart)

fit <- rpart(Y ~ ., data = data_class, method="class", subset=train, parms = list(split = 'gini'))
# TODO : tester avec autre indicateurs
yhat=predict(fit,newdata=data_class.test,type ='class')
y.test=data_class.test$Y
table(y.test,yhat)
err_tree<-1-mean(y.test==yhat)

plot(fit,margin = 0.05)
text(fit,pretty=0,cex=0.8)

fit$variable.importance

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

pruned_tree<-prune(fit,cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
plot(pruned_tree,margin = 0.05)
text(pruned_tree,pretty=0)

yhat=predict(pruned_tree,newdata=data_clas.test,type='class')
y.test=data_clas.test$y
table(y.test,yhat)
err_prunned<-1-mean(y.test==yhat)


############### FOREST ###############

library(randomForest)

### BAGING

bag.class = randomForest(as.factor(Y) ~., data=data_class, subset=train, mtry=p)
yhat1=predict(bag.class, newdata=data_class.test, type="response")
table(y.test,yhat1)
err_bagging<-1-mean(y.test==yhat1)

### RF

my_mtry <- sqrt(p)

RF.class = randomForest(as.factor(Y) ~., data=data_class, subset=train, mtry=my_mtry,importance=TRUE)
yhat2 = predict(RF.class, newdata=data_class.test, type="response")
table(y.test,yhat2)
err_RF<-1-mean(y.test==yhat2)

varImpPlot(RF.clas)
RF.class$importance
RF.class$importance[order(RF.class$importance[, 1], decreasing = TRUE), ]

### find best param mtry

bestmtry <- tuneRF(data_class.train[,2:16], data_class.train$Y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)
RF.clas = randomForest(as.factor(Y) ~., data=data_class, subset=train,mtry=bestmtry,importance=TRUE)
yhat2 = predict(RF.class, newdata=data_class.test, type="response")
table(y.test,yhat2)
err_mtry<-1-mean(y.test==yhat2)


############### GMM ###############

library(mclust)

class.train <- data_class.train$Y
X.train <- data_class.train[,2:16]
# general covariance structure selected by BIC
letterMclustDA <- MclustDA(X.train, class.train, modelType = "EDDA")
summary(letterMclustDA)#, parameters = TRUE)

class.test <- data_class.test$Y
X.test <- data_class.test[,2:16]
summary(letterMclustDA, class.test, X.test)

#plot(letterMclustDA)

############### SVM ###############

library(kernlab)

# tanhdot > rbfdot & polydot
fit<- ksvm(as.factor(Y) ~ .,kernel="tanhdot",C=1,data=data_class.train)
# Number of votes for each class
pred<-predict(fit,newdata=data_class.test,type = "votes")
# Error rate calculation
pred<-predict(fit,newdata=data_class.test,type = "response")
mean(data_class.test$Y != pred)