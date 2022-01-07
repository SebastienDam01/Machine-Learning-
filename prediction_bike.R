###### V2
library(e1071)

bike <- read.csv(file="bike_train.csv")

bike$weathersit <- as.factor(bike$weathersit)
bike$season <- as.factor(bike$season)

bike$cnt <- bike$cnt*1.5

model.bike =svm(cnt ~ weathersit+season+temp+windspeed+hum, 
         data = bike, kernel="radial", cost=0.87, scale=T)

prediction_bike <- function(dataset){
  library(e1071)
  dataset$weathersit <- as.factor(dataset$weathersit)
  dataset$season <- as.factor(dataset$season)
  predict(model.bike, dataset)
}


###### V3
library(e1071)

bike <- read.csv(file="bike_train.csv")

bike$weathersit <- as.factor(bike$weathersit)
bike$season <- as.factor(bike$season)

bike$cnt <- bike$cnt*1.29

model.bike =svm(cnt ~ weathersit+season+temp+windspeed+hum, 
                data = bike, kernel="radial", cost=0.87, scale=T)

prediction_bike <- function(dataset){
  library(e1071)
  dataset$weathersit <- as.factor(dataset$weathersit)
  dataset$season <- as.factor(dataset$season)
  predict(model.bike, dataset)
}


###### V4
library(e1071)

bike <- read.csv(file="bike_train.csv")

bike$cnt <- bike$cnt*1.29

model.bike =svm(cnt ~ weathersit+season+temp+windspeed+hum, 
                data = bike, kernel="radial", cost=0.87, scale=T)

prediction_bike <- function(dataset){
  library(e1071)
  predict(model.bike, dataset)
}




save("model.bike", "prediction_bike", file = "env.Rdata")

