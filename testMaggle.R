dataPhoneme <- read.table("parole_train.txt")
pred.phon <- prediction_phoneme(dataPhoneme)
pred.phon


dataLetter <- read.table("letters_train.txt")
pred.lett <- prediction_letter(dataLetter)
pred.lett


dataBike <- read.table("bike_train.csv",  sep=",",header=TRUE)
pred.bike <- prediction_bike(dataBike)
pred.bike

MSE = function(y.actual,y.predicted){
  mean((y.actual-y.predicted)^2)
}

MSE(dataBike$cnt,pred.bike)

