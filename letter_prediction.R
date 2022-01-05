letter <- read.table(file="data/letters_train.txt", sep="")

# Classification with RF (personalized mtry)

library(randomForest)

model.letter <- randomForest(as.factor(Y) ~., data=letter, mtry=3)

prediction_phoneme <- function(dataset){
  library(randomForest)
  
  predict(model.letter, newdata=dataset, type="response")
}

save("model.letter", "prediction_letter", file = "env.Rdata")