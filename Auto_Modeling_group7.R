#Load different datasets
data <- lapply(Sys.glob("df_train_*.csv"), read.csv)
#Load library
library(tree)
library(e1071)

#Train classification trees based on different data and random predictors
trees <- vector('list', 125)

for (i in seq_along(data)){
  for (col in colnames(data[[i]][,])){
    data[[i]][[col]] = factor(data[[i]][[col]])
  }
  s = sample(2:17, sample(2:6,1))
  trees[[i]] = tree(click~.,data=data[[i]][,(c(1,s))])
  print(i)
}
#Save classification trees
for (t in seq_along(trees)){
  saveRDS(trees[[t]], sprintf("./model_%s.rds", t))
}

#Train NB models based on different data and random predictors
NBs <- vector('list', 125)

for (i in seq_along(data)){
  for (col in colnames(data[[i]][,])){
    data[[i]][[col]] = factor(data[[i]][[col]])
  }
  s = sample(2:17, sample(2:6,1))
  NBs[[i]] = naiveBayes(click~.,data=data[[i]][,(c(1,s))])
  print(i)
}

#Save NB models
for (n in seq_along(NBs)){
  saveRDS(NBs[[n]], sprintf("./model_%s.rds", (n+125)))
}

#Train logistic models based on different data and random predictors
LGs <- vector('list', 125)

for (i in seq_along(data)){
  for (col in colnames(data[[i]][,])){
    data[[i]][[col]] = factor(data[[i]][[col]])
  }
  s = sample(2:17, sample(2:6,1))
  LGs[[i]] = glm(click~.,data=data[[i]][,(c(1,s))], family="binomial")
  print(i)
}

#Save logistic models
for (l in seq_along(LGs)){
  saveRDS(LGs[[l]], sprintf("./model_%s.rds", (l+250)))
}
