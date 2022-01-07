letter <- read.table(file="data/letters_train.txt", sep="")

# Classification with ksvm (personalized kernel and C)

library(kernlab)

model.letter <- ksvm(as.factor(letter$Y) ~ ., type = "C-svc", data=letter, kernel="laplacedot",C=100)

prediction_letter <- function(dataset){
  library(kernlab)
  
  predict(model.letter, dataset[2:17])
}

save("model.letter", "prediction_letter", file = "env.Rdata")