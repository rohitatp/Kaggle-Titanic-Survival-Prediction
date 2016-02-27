rm (list = ls())

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)

# Load training and testing data of titanic crew and passengers surviving/not surviving the crash

train <- read.csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test <- read.csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
str(train)
str(test)

# Test whether sex played a role in survival rate or not - it is more likely that females were saved before men
table(train$Survived)
prop.table(table(train$Survived)) # As proportions
table(train$Sex, train$Survived)  # Males & females that survived vs males & females that passed away
prop.table(table(train$Sex, train$Survived), margin=1)

# Test whether age played a role in survival rate or not - it is more likely that children were saved first
# Create the column child, and indicate whether child or no child
train$Child <- (train$Age < 18)
train$Child[train$Child == T] = 1
train$Child[train$Child == F] = 0
prop.table(table(train$Child, train$Survived), margin=1)

# We see that females had over a 50% chance of surviving and males had less than a 50% chance of surviving. Lets make a test predictor that saves all females.

test_one <- test
test_one$Survived <- 0
test_one$Survived[test_one$Sex == "female"] = 1 # Set Survived to 1 if Sex equals "female"
# Check for accuracy of this test on the training set
train_copy <- train
train_copy$hypo_test1<- 0
train_copy$hypo_test1[train$Sex == "female"] = 1 # Set Survived to 1 if Sex equals "female"

table(train_copy$Survived, train_copy$hypo_test1)

# We notice that this test has a high error. Note that we could have also used a cross validation set via partitioning the training set via createDataPartition command in Caret package but we do not do that here to save time.
# createDataPartition(train$Age, p = 0.7, list=FALSE)

# model using decision trees
model <- rpart(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = train, method = "class")
plot(model)
text(model)
fancyRpartPlot(model)

# Get predictions on test data
predictions <- predict(model, test, method="class")
# solution <- predict(model, test)
solution_v2 <- numeric(nrow(predictions))
solution_v2[predictions[,1] < 0.5] <- 1
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = solution_v2)
write.csv(my_solution, "my_solution.csv", row.names = FALSE)

# Use random forests to make decision tree split only when the number of entries in a bucket is 50 and the number of splits can be any number "cp".
# We further use the idea that larger families need more time to get together on a sinking ship, and hence have less chance of surviving. 
# Family size is determined by the variables SibSp and Parch, which indicate the number of family members a certain passenger is traveling with. 
# So when doing feature engineering, we add a new variable family_size, which is the sum of SibSp and Parch plus one (the observation itself), to the test and train set.

train_two <- train # create a new train set with the new variable
train_two$family_size <- train_two$SibSp + train_two$Parch + 1
my_tree_two<- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size, data = train_two, method="class", control = rpart.control(minsplit = 50, cp = 0))
fancyRpartPlot(my_tree_two)

# We note that family_size is not included in the split, so our hypothesis was incorrect in this case. Try another model with inclusion of title of passengers
#my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked+Title, data = train, control = rpart.control(minsplit=50, cp=0), method="class")
#fancyRpartPlot(my_tree_three )
#my_prediction_v2 <- predict(my_tree_three , test)
#my_solution_v2 <- data.frame(test_new$PassengerId, my_prediction_v2)
#write.csv(my_solution_v2, "my_solution.csv", row.names=FALSE)

## Use random forest, clean the data since there should be no missing element, predict age first, then construct new training and test set
test$Survived <- 0
test$Child <- (test$Age<18)
str(train)
str(test)
total <- rbind(train,test)
str(total)
# Passenger on row 62 and 830 do not have a value for embarkment. Since many passengers embarked at Southampton, we give them the value S.
total$Embarked[c(62, 830)] <- "S" 

# Factorize embarkment codes.
total$Embarked <- factor(total$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
total$Fare[1044] <- median(total$Fare, na.rm = TRUE)
# Age is not given for many entries so we predict age first.
model <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked, data = total[!is.na(total$Age),], method="anova") # We use Anova because we are predicting a continuous random variable
total$Age[is.na(total$Age)] <- predict(model, total[is.na(total$Age),])

# Split the data back into a train set and a test set
train <- total[1:891,]
test <- total[892:1309,]
set.seed(111)

# Use random forest. Why? Because random forests grows multiple (very deep) classification trees using the training set. At the time of prediction, 
# each tree is used to come up with a prediction and every outcome is counted as a vote. This gives redundancy to our predictor and increases accuracy. This is also called bagging (= Bootstrap AGGregatING).
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare, total, importance = TRUE, ntree = 1000)
my_prediction <- predict(my_forest, test)
my_solution_v3 <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
write.csv(my_solution_v3, file = "my_solution_v3.csv", row.names = FALSE)

varImpPlot(my_forest)


