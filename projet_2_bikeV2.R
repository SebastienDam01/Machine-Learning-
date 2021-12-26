install.packages('Rcpp')
library(Rcpp)

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
library(e1071)
library(kernlab)

bike <- read.csv("C:/Users/assil/Desktop/UTC/BRANCHE/GI04_A21/SY19/Projet_2/bike_train.csv")


dates = as.Date(bike$dteday)
dates <- as.POSIXct(dates, format = "%Y-%m-%d")
bike$day=as.numeric(format(dates, format="%d"))
bike <- subset(bike, select=c(day,dteday:cnt))





# Instant correspond simplement à l'indice de la ligne des données, ce n'est
# pas une donnée interessante, la variable yr est toujours à 0. dteday contient 
# la date sous forme "YYYY-MM-DD".

# Nous pouvons par contre extraire le jour dans dteday et creer une nouvelle 
# variable day. 

bike$instant<-NULL
bike$yr<-NULL
bike$dteday<-NULL



#Corrélogramme
matCor <- cor(bike)
corrplot(matCor, type="upper", method ="color" )

# Comme nous pouvions l'imaginer, certaines variables sont fortement corrélées
# (temp et atemp,  season et mnth). Nous pourrion  enlever au préalable 
# les varables qui n'apportent pas d'information. Au lieu de cela nous allons
# laisser le dataset dans l'état et laisser la fonction regsubset selectionner 
# les prédicteurs plus significatifs. 

n <- dim(bike)[1]
p <- dim(bike)[2]




acp<-princomp(scale(bike[,-p]))
Z<-acp$scores
lambda<-acp$sdev
plot(cumsum(lambda)/sum(lambda),type="l",xlab="Nombre de prédicteurs",
     ylab="proportion of explained variance")

# La courbe s'aplatit au niveau des abscisses les plus grandes, 
# ce qui confirme qu'il existe un sous ensemble de variables significatives et
# que certaines variables n'apportent pas d'information ultérieures.


MSE = function(y.actual,y.predicted){
  mean((y.actual-y.predicted)^2)
}


models.list <- c("lm", "knn", "svm.line", "svm.rad", "svm.poly",
                 "ridge", "lasso", "gam", "bag.tree", "boost.tree",
                 "rndm.forest", "reg.tree")

models.MSE <- as.data.frame(matrix(0, nrow=1, ncol=length(models.list)))
colnames(models.MSE) <- models.list



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     Best subset of predictors     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# 4 méythodes possible "exhaustive", "backward", "forward" et "seqrep",
# les 4 donnent les mêmes subsets

leaps <- regsubsets(bike$cnt~. , data=bike, 
                    method='exhaustive',
                    nvmax=11)
summary(leaps)
reg_summary = summary(leaps)
names(reg_summary)
par(mfrow = c(2,3))

plot(reg_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
rss_min = which.min(reg_summary$rss) 
points(rss_min, reg_summary$rss[rss_min], col ="red", cex = 2, pch = 20)

plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
adj_r2_max = which.max(reg_summary$adjr2) 
points(adj_r2_max, reg_summary$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)

plot(reg_summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
cp_min = which.min(reg_summary$cp) 
points(cp_min, reg_summary$cp[cp_min], col = "red", cex = 2, pch = 20)

plot(reg_summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(reg_summary$bic) 
points(bic_min, reg_summary$bic[bic_min], col = "red", cex = 2, pch = 20)

plot(reg_summary$rsq, xlab = "Number of Variables", ylab = "RSq", type = "l")
rsq_max = which.max(reg_summary$rsq)
points(rsq_max, reg_summary$rsq[rsq_max], col = "red", cex = 2, pch = 20)


plot(leaps, scale = "bic")
plot(leaps, scale = "r2")
plot(leaps, scale = "adjr2")
plot(leaps, scale = "Cp")



models <- leaps
summary(models)

# Fonction qui cree la formule du modèle, elle sera utilisée pour
# creer dynamiquement les formules avec differents nombres de predicteurs

get_model_formula <- function(id, object, outcome){
  models <- summary(object)$which[id,-1]
  predictors <- names(which(models == TRUE))
  predictors <- paste(predictors, collapse = "+")
  as.formula(paste0(outcome, "~", predictors))
}


# Paramètres utilisées pour le tuning des modèles avec le package caret
# Tuing grid pour le modèle boosted tree
gr.gbm <-  expand.grid(interaction.depth = c(1, 5, 9), 
                       n.trees = (1:30)*50, 
                       shrinkage = c(0.01, 0.1, 1),
                       n.minobsinnode = 20)

par(mfrow = c(1,1))
set.seed(123)
train.control <- trainControl(method = "repeatedcv", number = 5,  repeats = 10)
model.ids <- 1:11


# Cette boucle, grace aux package caret, fait une cv (5 folds 10 tests)
# sur chaque modèle, pour chaque modèle elle teste les 10 sous ensembles possibles 
# (de 1 à 10 prédicteurs) tout en testant pour chaque combinaison, 
# les paramètres de tuning du modèle. A la fin pour chaque modèle on obtient 
# la méilleure performance (obtenue sur le meilleur sous ensemble avec les 
# meilleurs paramètres du modèle). La métrique utilisée est le RMSE
# qu'on eleve au carré pour obtenir le MSE. La combinatoire est très élevée 
# et donc cette boucle peut prendre quelques heures pour etre parcourue.

for(model in models.list)
{
  if(model == "lm")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "lm",
                  trControl = train.control)
      cv$results$RMSE
    }
  }
  else if(model == "knn")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "knn",
                  preProcess = c("center","scale"), 
                  trControl = train.control,
                  tuneLength = 9)
      cv$results$RMSE
    }
  }
  else if(model == "svm.rad")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "svmRadial",
                  preProcess = c("center","scale"),
                  trControl = train.control,
                  tuneLength = 10)
      cv$results$RMSE
    }
  }
  else if(model == "svm.line")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "svmLinear",
                  preProcess = c("center","scale"),
                  trControl = train.control,
                  tuneLength = 10)
      cv$results$RMSE
    }
  }
  else if(model == "svm.poly")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "svmPoly",
                  trControl = train.control,
                  preProcess = c("center","scale"), 
                  tuneLength = 4)
      cv$results$RMSE
    }
  }
  else if(model == "ridge")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "bridge",
                  trControl = train.control)
      cv$results$RMSE
    }
  }
  else if(model == "lasso")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "blassoAveraged",
                  trControl = train.control)
      cv$results$RMSE
    }
  }
  else if(model == "gam")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "gamSpline",
                  preProcess = c("center","scale"), 
                  trControl = train.control,
                  df = 5)
      cv$results$RMSE
    }
  }
  else if(model == "reg.tree")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "rpart",
                  preProcess = c("center","scale"), 
                  trControl = train.control,
                  tuneLength = 100)
      cv$results$RMSE
    }
  }
  else if(model == "bag.tree")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "treebag",
                  preProcess = c("center","scale"), 
                  trControl = train.control)
      cv$results$RMSE
    }
  }
  else if(model == "boost.tree")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "gbm",
                  preProcess = c("center","scale"), 
                  trControl = train.control,
                  verbose = FALSE, 
                  tuneGrid = gr.gbm)
      cv$results$RMSE
    }
  }
  else if(model == "rndm.forest")
  {
    get_cv_error <- function(model.formula, data){
      cv <- train(model.formula, data = data, "cforest",
                  preProcess = c("center","scale"), 
                  trControl = train.control,
                  tuneLength = 10)
      cv$results$RMSE
    }
  }
  else
  {
    class.model <- NULL
  }
  cv.errors <-  map(model.ids, get_model_formula, models, "cnt") %>%
    map(get_cv_error, data = bike) %>%
    unlist()
  
  models.MSE[model] <- min(cv.errors)^2
}


par(cex.axis= 0.7)
boxplot(models.MSE, main= "Minimum MSE par modèle, 
        sans séléction de variables préalable")


# On peut voir que le modèle qui donne le MSE le plus faible est le svm
# avec radial kernel. Par contre nous ne savons pas quelle combinaison
# (nombre de predicteurs et paramètres du modèle) a permi d'obtenir ce resultat.
# Nous allons maintenant mener une recherche plus fine sur ce modèle.


get_cv_error <- function(model.formula, data){
  set.seed(1)
  train.control <- trainControl(method = "repeatedcv", number = 5,  repeats = 10)
  cv <- train(model.formula, data = data, "svmRadial",
              preProcess = c("center","scale"),
              trControl = train.control,
              tuneLength = 10)
  cv$results$RMSE
}


svm.radial.cv.errors <-  map(model.ids, get_model_formula, models, "cnt") %>%
  map(get_cv_error, data = bike) %>%
  unlist()

min.MSE.svm.radial <-min(svm.radial.cv.errors)^2

plot(svm.radial.cv.errors, ann=FALSE)
title(main = "RMSE en fonction des combinasons de C
     et du sous ensemble de prédicteurs", font.main=4)
mse_min = which.min(svm.radial.cv.errors) 
points(mse_min, svm.radial.cv.errors[mse_min], col ="red", cex = 2, pch = 20)
min.MSE.svm.radial
# svm radial lowest MSE         : 298 495

# Dans ce plot nous pouvons voir un peu mieux comment la boucle au dessus procède,
# on teste le modèle avec 10 sous ensembles de prédicteurs (de 1 à 10) et pour
# chaque sous ensemble on teste 10 valeurs du parametre spécifique au modèle
# dans le cas du svm avec radial kernel c'est le C (cost). On a donc fait 100
# cross validations pour aboutir au meilleur paramétrage du modèle. La métrique 
# utilisée est toujours le RMSE. On voit qu'il y a 10 paraboles 
# (une par sous ensemble de predicteurs), chaqune constuée de 10 points 
# (un par valeur de C). La valeur de RMSE la plus faibles se trouve sur 
# la 6ème parabole. Cela signifie que ça a été obtenu avec le sous ensemble de
# 6 prédicteurs. Voici le sous ensemble de predicteurs en question :

columns <- coef(leaps, 6)
columns.names <- names(columns)
columns.names
# "day", "season", "weathersit","temp" ,"hum", "windspeed"  

# Nous savons maintenant que ces 6 prédicteurs permettent d'obtenir le MSE
# le plus faible avec modèle basé sur le svm radial kernel. Mais nous ne savons
# pas quel sont les paramètres C et sigma qui permettent  d'obtenir ce resultat. 
# Pous ce faire nous utilisons le package caret qui permet d'entreiner le modèle
# et renvoie les paramètres qui permettent d'optimiser le RMSE.

svm.radial <- train(
  cnt~season+weathersit+temp+hum+windspeed+day, data = bike, method = "svmRadial",
  preProcess = c("center","scale"),
  trControl = train.control,
  tuneLength = 10)

# On cherche les meilleurs parametres
sigma.opt = svm.radial$bestTune[1]
c.opt = svm.radial$bestTune[2]

# La valeur du paramètre sigma qui a permis d'obtenir le resultat optimal ets donc :
sigma.opt

# La valeur du paramètre C qui a permis d'obtenir le resultat optimal ets donc :
c.opt

# En relançant cette cv plusieurs fois on se rend compte que la valuer de C
# est parfois égale à 1 et parfois égale à 2. Nous décidons alors de lancer 
# une validation croisée plus fine qui teste le MSE en fonction de toutes 
# les valeurs de C entre 1 et 2 par pas de 0.1, en gardant la valeur de sigma
# retrouvée précedemment.

# On peut determiner le parametre C avec une cv

data <- bike
data$holiday<- NULL
data$weekday<- NULL
data$workingday<- NULL
data$mnth<- NULL


p <- ncol(data)
n <-  nrow(data)

trials <- 100
n_folds <- 5


cc <- seq(from = 0.1, to = 2, by = 0.1)
c.MSE <- as.data.frame(matrix(0, nrow=trials, ncol=length(cc)))

for(c in 1:length(cc))
{
  for(i in c(1:trials)) 
  {
    folds <- sample(rep(1:n_folds, length.out=n))
    trial.MSE <- 0
    for(j in 1:n_folds)
    {
      data.train <- data[folds!=j,]
      data.train.X <- data.train[,-p]
      data.train.y <- data.train$cnt
      data.test <- data[folds==j,]
      data.test.X <- data.test[,-p]
      data.test.y <- data.test$cnt
      
      fit =svm(data.train.y ~ .,data.train.X,kernel="radial",cost=cc[c],
               sigma=sigma.opt, scale=T)
      pred = predict(fit, newdata= data.test.X)
      fold.MSE = MSE(data.test.y, pred)
      trial.MSE <- trial.MSE + fold.MSE
      
    }
    c.MSE[i,c] <- trial.MSE/n_folds
  }
}
boxplot(c.MSE, main="MSE en foction du paramètre C")
plot(cc, colMeans(c.MSE), type='l',
     xlab='cost', ylab='MSE, radial kernel svm', main = "MSE en fonction de C")

means.c.MSE <- colMeans(c.MSE)
c.mse_min = which.min(means.c.MSE)
c.opt <- cc[c.mse_min]

# Cette recherche se montre utile car effectivement la valeur de C qui
# minise de MSE se trouve entre 1 et 2 et  vaut :

c.opt


# On teste le modèle obtenu :
p <- ncol(data)
n <-  nrow(data)

nb.train <- round(2*n/3)
nb.test <- n - nb.train

set.seed(123) 

# Training/Testing data
train <- sample(1:n, nb.train)

data.train <- data[train,]
data.train.X <- data.train[,-p]
data.train.y <- data.train$cnt
data.test <- data[-train,]
data.test.X <- data.test[,-p]
data.test.y <- data.test$cnt

fit =svm(data.train.y ~ ., data.train.X, kernel="radial", cost=c.opt, 
         sigma=sigma.opt, scale=T)
pred = predict(fit, newdata= data.test.X)

#Le MSE vaut :
MSE(data.test.y, pred)

# Le resultat est coherent avec les calculs obtenus sur les differentes
# cross validations



