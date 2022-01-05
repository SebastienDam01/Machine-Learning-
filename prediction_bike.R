###### V1
library(e1071)


bike <- read.csv(file="data/parole_train.txt")


model.bike <- svm(cnt~weathersit+temp+windspeed+season+hum, data=bike, kernel="radial", cost=1.5, scale=T)

prediction_bike <- function(dataset){
  library(e1071)
  predict(model.bike, dataset)
}

save("model.bike", "prediction_bike", file = "env.Rdata")

##### fin V1

###### V2
library(e1071)


bike <- read.csv(file="data/parole_train.txt")


model.bike <- svm(cnt~weathersit+temp+windspeed+season+hum+atemp+mnth+
                    workingday+weekday+holiday, data=bike, kernel="radial", 
                  cost=4, scale=T)

prediction_bike <- function(dataset){
  library(e1071)
  predict(model.bike, dataset)
}

save("model.bike", "prediction_bike", file = "env.Rdata")
