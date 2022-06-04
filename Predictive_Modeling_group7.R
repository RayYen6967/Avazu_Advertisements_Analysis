#Load data
df_train = read.csv("df_train_1.csv")
df_test = read.csv("df_test_0.csv")

#Factor categorical features
for (col in colnames(df_train[,])){
  df_train[[col]] = factor(df_train[[col]])
}
for (col in colnames(df_test[,])){
  df_test[[col]] = factor(df_test[[col]])
}

##Test different algorithms 
#Down sample for quick performance testing 
set.seed(123)
sample2 = sample(nrow(df_train), 0.01*nrow(df_train))
df_train_sample3 = df_train[sample2,]
set.seed(123)
df_test_sample3 = df_test[sample2,]
#Prune classification tree
library(tree)
tree=tree(click~.,data=df_train_sample3)
set.seed(123)
cv_df=cv.tree(tree)
plot(cv_df$size,cv_df$dev,type='b')
z = which.min(cv_df$size)
prune_tree = prune.tree(tree,best=z)
prune_tree_pred = predict(prune_tree, newdata=df_test_sample3, type="class")
(c = table(prune_tree_pred,df_test_sample3$click))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))
#Bagging 
library(randomForest)
set.seed(1)
rf=randomForest(click~.,data=df_train_sample3,mtry=16,importance=TRUE)
rf
yhat.rf = predict(rf,newdata=df_test_sample3[,-1])
rf.test = df_test_sample3[,"click"]
(c = table(rf.test,yhat.rf))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))
#Random forest 
set.seed(1)
rf=randomForest(click~.,data=df_train_sample3,mtry=2,importance=TRUE)
rf
yhat.rf = predict(rf,newdata=df_test_sample3[,-1])
rf.test = df_test_sample3[,"click"]
(c = table(rf.test,yhat.rf))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))
#Boosting
library(gbm)
set.seed(1)
df_train_sample3$click = as.numeric(df_train_sample3$click)-1
boost = gbm(click~.,data=df_train_sample3,distribution="bernoulli",n.trees=5000,interaction.depth=4)
summary(boost)
yhat.boost=predict(boost,newdata=df_test_sample3[,-1],n.trees=5000,type="response")
predicted <- ifelse(yhat.boost>=0.5,1,0)
yhat.test= df_test_sample3$click
(c = table(predicted,yhat.test))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))
df_train_sample3$click = factor(df_train_sample3$click)
#Navie Bayes
library(e1071)
nb = naiveBayes(click~., data=df_train_sample3)
nb
nb.test = predict(nb, newdata = df_test_sample3)
yhat.test=df_test_sample3$click
(c = table(nb.test,yhat.test))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))
#XGB
library(xgboost)
label = as.numeric(df_train_sample3$click)-1
mat.df_train_sample3 = as.matrix(df_train_sample3[, 2:17])
mode(mat.df_train_sample3) <- 'double'
bst <- xgboost(data = mat.df_train_sample3, label = label, eta = 0.001, nround = 800, objective = "binary:logistic")

labelT = as.numeric(df_test_sample3$click)-1
dataT = as.matrix(df_test_sample3[, 2:17])
mode(dataT) <- 'double'
pred <- predict(bst, dataT)
predicted <- ifelse(pred>0.5,1,0)
(c = table(labelT,predicted))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))
#KNN
#Load data
library(fastDummies)
df_train.knn_dum <- dummy_cols(df_train, select_columns = colnames(df_train[,-1]), remove_selected_columns=TRUE)
df_test.knn_dum <- dummy_cols(df_test, select_columns = colnames(df_test[,-1]), remove_selected_columns=TRUE)
#Normalize each variable
fun <- function(x){ 
  a <- mean(x) 
  b <- sd(x) 
  (x - a)/(b) 
} 
df_train.knn_dum[,-1] <- apply(df_train.knn_dum[,-1], 2, fun)
df_test.knn_dum[,-1] <- apply(df_test.knn_dum[,-1], 2, fun)
#Load data
df_train_sample3.knn = df_train.knn_dum[sample2,]
df_test_sample3.knn = df_test.knn_dum[sample2,]
#knn may be found in the library class
library(class)
train_input.knn <- as.matrix(df_train_sample3.knn[,-1])
train_output.knn <- as.vector(df_train_sample3.knn[,1])
test_input.knn <- as.matrix(df_test_sample3.knn[,-1])
#
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)
#
set.seed(12345)
for (i in 1:kmax){
  prediction <- knn(train_input.knn, train_input.knn,train_output.knn, k=i)
  prediction2 <- knn(train_input.knn, test_input.knn,train_output.knn, k=i)
  # The confusion matrix for training data is:
  CM1 <- table(df_train_sample3.knn$click, prediction)
  # The training error rate is:
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  # The confusion matrix for validation data is: 
  CM2 <- table(df_test_sample3.knn$click, prediction2)
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
}
z <- which.min(ER2)
#Scoring at optimal k
prediction <- knn(train_input.knn, train_input.knn, train_output.knn, k=z)
prediction2 <- knn(train_input.knn, test_input.knn, train_output.knn, k=z)
(c = table(df_test_sample3.knn$click, prediction2))
(acc = (c[1,1]+c[2,2])/sum(c))
(sensitivity = (c[2,2])/sum(c[2,]))
(specificity = (c[1,1])/sum(c[1,]))

##Ensemble all 375 de-correlated models

#Load different individual models
models <- lapply(Sys.glob("model_*.rds"), readRDS)

#Build and test our own ensemble algorithm 
#Classification Tree + NaiveBayes + Logistic Regression
final_predictions = rep(0,1000)

for (i in 1:1000){
  predictions = rep(0,375)
  for (m in seq_along(models)){
    if (class(models[[m]]) == "tree"){
      predictions[m] = predict(models[[m]], df_test_sample3[i, -1], type="vector")[,2]
    }
    else if (class(models[[m]]) == "naiveBayes"){
      predictions[m] = predict(models[[m]], df_test_sample3[i, -1], type="raw")[,2]
    }
    else if (class(models[[m]])[1] == "glm") {
      predictions[m] = predict(models[[m]], df_test_sample3[i, -1], type="response")[1]
    }
  }
  final_predictions[i] = mean(predictions)
  print(i)
}

#Get predictive classes from predictive probabilities
final_predictions_classes = factor(ifelse(final_predictions > 0.5, 1, 0))
actual_classes = df_test_sample3$click

#Confusion matrix & performance matrics
CM = table(actual_classes,final_predictions_classes)
(accuracy = (CM[1,1]+CM[2,2])/sum(CM))
(sensitivity = (CM[2,2])/sum(CM[2,]))
(specificity = (CM[1,1])/sum(CM[1,]))

#ROC curve
library(pROC)
par(pty="s")
roc_rose <- plot(roc(actual_classes, final_predictions), print.auc = TRUE, legacy.axes=TRUE,col = "blue", main = 'ROC curves')

#Lift chart
actual <- as.numeric(actual_classes)-1
predicted.probability <- final_predictions

df1V <- data.frame(predicted.probability,actual)
df1VS <- df1V[order(-predicted.probability),]
df1VS$Gains <- cumsum(df1VS$actual)
plot(df1VS$Gains,type="n",main="Lift Chart",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1VS$Gains)
abline(0,sum(df1VS$actual)/nrow(df1VS),lty = 2, col="red")