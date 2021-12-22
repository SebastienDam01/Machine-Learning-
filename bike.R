# TODO :
#   - améliorer svm (plus prometteur)
#   - implementer cv pour le meilleur subset

library(corrplot)
library(dplyr)
library(gridExtra) 
library(ggplot2) 
library(tidyverse)
library(caret)
library(leaps)
library(MASS)
library(glmnet)
library(gam)
library(flexmix)

bike <- read.csv("C:/Users/assil/Desktop/UTC/BRANCHE/GI04_A21/SY19/Projet_2/bike_train.csv")


dates = as.Date(bike$dteday)
dates <- as.POSIXct(dates, format = "%Y-%m-%d")
bike$day=as.numeric(format(dates, format="%d"))
bike <- subset(bike, select=c(day,dteday:cnt))





# Instant correspond simplement à l'indice de la ligne des données, ce n'est
# pas une donnée interessante, la variable yr est toujours à 0, on peut retrouver 
# cette information dans la variable dteday qui contient la date sous forme 
# "YYYY-MM-DD", nous allons donc récuperer la variable yr dans dans dteday et puis
# supprimer cette dernière variable car elle est redondante avec la variable mnth.
# Edit, la partie YYYY dans dteday est une constante egale à 2011. Inutilisable
# pour faire des prédictions

# Nous pouvons par contre extraire le jour dans dteday et creer une nouvelle 
# variable day. La variable season parait également supérflue car la variable 
# month donne sans ambiguité la saison, le correlogramme montre qu'elles sont 
# fortement correlées.Les variables temp et atemp sont aussi fortement corrélées
# nous ne garderons que temp

bike$instant<-NULL
bike$yr<-NULL
bike$dteday<-NULL
bike$season<-NULL
bike$atemp<-NULL



#Corrélogramme
matCor <- cor(bike)
corrplot(matCor, type="upper", method ="color" )

n <- dim(bike)[1]
p <- dim(bike)[2]


MSE = function(y.actual,y.predicted){
  mean((y.actual-y.predicted)^2)
}


models.list <- c("reg", "knn")
trials <- 100
n_folds <- 5
k_candidates <- 9
nb.predictors_canditates<- 9

models.MSE <- as.data.frame(matrix(0, nrow=trials, ncol=length(models.list)))
colnames(models.MSE) <- models.list

k.error <- as.data.frame(matrix(0, nrow=trials, ncol=k_candidates))
k.error.scaled <- as.data.frame(matrix(0, nrow=trials, ncol=k_candidates))
nb.predictors.error <- as.data.frame(matrix(0, nrow=trials, ncol=nb.predictors_canditates))

for(k in c(1:k_candidates))
{
  old.name <- paste("V",k, sep = "")
  new.name <- paste("K=",k, sep = "")
  names(k.error)[names(k.error) == old.name]<- new.name
  names(k.error.scaled)[names(k.error.scaled) == old.name]<- new.name
}



#~~~~~~~~~~~~~~~~~~~~~#
#  KNN cv for best K  #
#~~~~~~~~~~~~~~~~~~~~~#
for(k in c(1:k_candidates))
{
  for(i in c(1:trials))
  {
    folds <- sample(rep(1:n_folds, length.out=n))
    trial.knn.MSE <- 0
    for(j in 1:n_folds) 
    {
      bike.train <- bike[folds!=j,]
      bike.train.X <- bike.train[,-p]
      bike.train.y <- bike.train$cnt
      bike.test <- bike[folds==j,]
      bike.test.X <- bike.test[,-p]
      bike.test.y <- bike.test$cnt
      
      knn.fit = knnreg(bike.train.X, bike.train.y, k=k)
      knn.pred = predict(knn.fit, bike.test.X)
      trial.knn.MSE = MSE(bike.test.y, knn.pred)
    }
    k.error[i,k] <- trial.knn.MSE/n_folds
  }
}
boxplot(k.error, main="Erreur par k-voisins")
DF <- seq(1, 9)
plot(DF, colMeans(k.error), type='b',
     xlab='K neighbours', ylab='CV-error')

k.optimal <- match(min(colMeans(k.error)),colMeans(k.error))


#~~~~~~~~~~~~~~~~~~~~~#
#  CV for best model  #
#~~~~~~~~~~~~~~~~~~~~~#
for(model in models.list)
{
  for(i in c(1:trials)) 
  {
    folds <- sample(rep(1:n_folds, length.out=n))
    trial.MSE <- 0
    for(j in 1:n_folds)
    {
      bike.train <- bike[folds!=j,]
      bike.train.X <- bike.train[,-p]
      bike.train.y <- bike.train$cnt
      bike.test <- bike[folds==j,]
      bike.test.X <- bike.test[,-p]
      bike.test.y <- bike.test$cnt
      
      if(model == "reg")
      {
        fit <- lm(bike.train.y~., data = bike.train)
        pred <- predict(fit, newdata=bike.test)
        fold.MSE <- MSE(bike.test.y, pred)
        trial.MSE <- trial.MSE + fold.MSE
      }
      else if(model == "knn")
      {
        fit = knnreg(bike.train.X, bike.train.y, k=k.optimal)
        pred = predict(fit, bike.test.X)
        fold.MSE = MSE(bike.test.y, pred)
        trial.MSE <- trial.MSE + fold.MSE
      }
      else
      {
        class.model <- NULL
      }
    }
    models.MSE[i,model] <- trial.MSE/n_folds
  }
}
boxplot(models.MSE, main="MSE par modèle")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     Best subset of predictors     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Essayer forward et backward
leaps <- regsubsets(bike$cnt~. , data=bike, method="forward", nvmax=9)
summary(leaps)
reg_summary = summary(leaps)
names(reg_summary)
par(mfrow = c(2,3))

plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
rss_min = which.min(reg_summary$rss) #50
points(rss_min, reg_summary$rss[rss_min], col ="red", cex = 2, pch = 20)

plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
adj_r2_max = which.max(reg_summary$adjr2) #21
points(adj_r2_max, reg_summary$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)

plot(reg_summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
cp_min = which.min(reg_summary$cp) #11
points(cp_min, reg_summary$cp[cp_min], col = "red", cex = 2, pch = 20)

plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(reg_summary$bic) #4
points(bic_min, reg_summary$bic[bic_min], col = "red", cex = 2, pch = 20)

plot(reg_summary$rsq, xlab = "Number of Variables", ylab = "RSq", type = "l")
rsq_max = which.max(reg_summary$rsq) #50
points(rsq_max, reg_summary$rsq[rsq_max], col = "red", cex = 2, pch = 20)


plot(leaps, scale = "bic")
plot(leaps, scale = "r2")
plot(leaps, scale = "adjr2")
plot(leaps, scale = "Cp")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#        cv for number of predictors by model          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

leaps <- regsubsets(bike$cnt~. , data=bike, method="backward", nvmax=9)
summary(leaps)
reg_summary = summary(leaps)
names(reg_summary)
par(mfrow = c(1,1))

for(nb.predictors in c(4:nb.predictors_canditates))
{
  columns <- coef(leaps, nb.predictors)
  columns.names <- names(columns)
  columns.names <- columns.names[-1]
  
  bike.tmp <- cbind(bike[columns.names],bike[,p])
  names(bike.tmp)[ncol(bike.tmp)] <- "cnt"
  bike.tmp <- as.data.frame(bike.tmp)
  
  n <- dim(bike.tmp)[1]
  p <- dim(bike.tmp)[2]
  for(i in c(1:trials)) 
  {
    folds <- sample(rep(1:n_folds, length.out=n))
    trial.MSE <- 0
    for(k in 1:n_folds)
    {
      fold.MSE <- 0
      bike.train <- as.data.frame(bike.tmp[folds!=k,])
      bike.train.X <- as.data.frame(bike.train[,-p])
      bike.train.y <- bike.train$cnt
      bike.test <- as.data.frame(bike.tmp[folds==k,])
      bike.test.X <- as.data.frame(bike.test[,-p])
      bike.test.y <- bike.test$cnt
      
      fit = knnreg(bike.train.X, bike.train.y, k=k.optimal)
      pred = predict(fit, bike.test.X)
      fold.MSE = MSE(bike.test.y, pred)
      trial.MSE <- trial.MSE + fold.MSE
      
    }
    nb.predictors.error[i,nb.predictors] <- trial.MSE/n_folds
  }
}
boxplot(nb.predictors.error, main="Erreur par nombre de prédicteurs, knn")
# Meilleurs resultats avec 4 predicteurs




