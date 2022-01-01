library(factoextra) # PCA

phonemes <- read.table(file="data/parole_train.txt", sep="")

# PCA

res.pca <- prcomp(phonemes[, 1:256], scale = TRUE)
phonemes_pc <- res.pca$x
phonemes_pc <- phonemes_pc[,1:50]
phonemes_pc <- as.data.frame(phonemes_pc)

phonemes_pc$y = phonemes$y

# Classification with lasso

library(glmnet)

model.phoneme <- cv.glmnet(phonemes_pc[, 1:50], phonemes_pc$y, alpha=1, family='multinomial', nfold=3)

prediction_phoneme <- function(dataset){
  library(glmnet)
  
  predict(model.phoneme, newx=dataset, s="lambda.min", type="class")
}

# Classification with ElasticNet

library(glmnet)

model.phoneme <- glmnet(phonemes_pc[, 1:50], phonemes_pc$y, alpha=1, lambda=0.00661, family='multinomial', nfold=3)

prediction_phoneme <- function(dataset){
  library(glmnet)
  
  predict(model.phoneme, dataset, type="class")
}

# Classification with SVM-rbfdot

library(kernlab)

model.phoneme <- ksvm(as.factor(y)~ ., data=phonemes_pc, type="C-svc", kernel="rbfdot", C=1)

prediction_phoneme <- function(dataset){
  library(kernlab)
  
  predict(model.phoneme, dataset)
}

# Classification with RDA

library(klaR)

model.phoneme <- rda(y~ ., data=phoneme_pc, lambda=0.83, gamma=0.29)

prediction_phoneme <- function(dataset){
  library(klaR)
  
  predict(model.phoneme, dataset)$class
}

save("model.phoneme", "prediction_phoneme", file = "env.Rdata")