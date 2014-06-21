Practical Machine Learning Course Project
=========================================

# Exploration


```r
train = read.csv("pml-training.csv")
# we check names(train) and spare the long output
dim(train)
```

```
## [1] 19622   160
```

```r
# head(train)
```


Many variables have no values, we also see training specific data (such as names).

Interestingly, the test and training sets contains names of participants as well as  timestamps of the exercise. It would be interesting to see if the test set can be predicted with just those.

# Design

In order to assess which methodology works best, we split 80, 20 and will pick the models that performs best on the 20%. 


```r
library(ggplot2)
library(lattice)
library(rpart)
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(caret)
set.seed(123)
trainIdx = createDataPartition(train$classe, p = 0.8, list = F)
# we drop the initial column with row indices
train80 = train[trainIdx, -1]
cv20 = train[-trainIdx, -1]
```


# Using Variables Not Ordinarily Available

We keep only `user_name` and _raw timestamps_ and see what we can get with a rpart method. Intuitively, we cheat by training our model to pick a prediction based on the user and when they perform their test.


```r
name_ts_train = train80[, c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "classe")]
names(name_ts_train)
```

```
## [1] "user_name"            "raw_timestamp_part_1" "raw_timestamp_part_2"
## [4] "classe"
```

```r
name_ts_fit = train(classe ~ ., method = "rpart", data = name_ts_train)
name_ts_cv = cv20[, c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2")]
name_ts_cv_result = predict(name_ts_fit, name_ts_cv)
name_ts_error_rate = sum(name_ts_cv_result != cv20$classe)/length(cv20$classe)
name_ts_cm = confusionMatrix(name_ts_cv_result, cv20$classe)
name_ts_cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1112    2    0    0    0
##          B    4  754    7    0    0
##          C    0    3  675    3    0
##          D    0    0    2  638    1
##          E    0    0    0    2  720
```


We get a very high accuracy 0.9939. This is somewhat expected as we take advantage of data that is very specific to the training conditions (name of users) and supposed date and time of when the recording took place, data which we would not have in general.

# Using Random Forest on Sensor Variables

## Only Acceleration Variables

Let's assume a more useful case and disallow usage of names and timestamps. This leaves all the actual sensor variables and leaves out time series information. To reduce the time it takes to train the model, we limit to sensor variable with accel in the name.


```r
library(randomForest)
vars = grep("^accel", names(train80))
sensor_train = train80[, vars]
set.seed(123)
sensor_fit = randomForest(train80$classe, x = sensor_train, type = "classification")
sensor_cv = cv20[, vars]
sensor_cv_result = predict(sensor_fit, sensor_cv)
sensor_cm = confusionMatrix(sensor_cv_result, cv20$classe)
sensor_cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1069   14   11   14    1
##          B    6  713   20    3   10
##          C   14   24  643   24    4
##          D   27    1    7  598    5
##          E    0    7    3    4  701
```


The accuracy is 0.9493 which is not bad. Can we do better?

## Adding Forearm Variables

Given the weight lifting exercise, it is conceivable that the forearm sensor will have useful information. Let's rerun the training with more forearm variables.


```r
library(randomForest)
vars = grep("^accel|^gyros_forearm_|^(roll|yaw|pitch)_forearm", names(train80))
sensor_train1 = train80[, vars]
set.seed(123)
sensor_fit1 = randomForest(train80$classe, x = sensor_train1, type = "classification")
sensor_cv1 = cv20[, vars]
sensor_cv_result1 = predict(sensor_fit1, sensor_cv1)
sensor_cm1 = confusionMatrix(sensor_cv_result1, cv20$classe)
sensor_cm1$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1098   12    3    3    0
##          B    7  738   12    1    1
##          C    2    6  664   18    0
##          D    8    1    5  618    3
##          E    1    2    0    3  717
```


The accuracy is 0.9776 which is an improvement.

## Adding Arm and Dumbell Variables

Finally, let's add some arm and dumbbell variables.


```r
library(randomForest)
vars = grep("^accel|^gyros_forearm_|^(roll|yaw|pitch)_forearm|^gyros_arm|^(roll|yaw|pitch)_arm|^gyros_dumb|^(roll|yaw|pitch)_dumb", 
    names(train80))
sensor_train2 = train80[, vars]
set.seed(123)
sensor_fit2 = randomForest(train80$classe, x = sensor_train2, type = "classification")
sensor_cv2 = cv20[, vars]
sensor_cv_result2 = predict(sensor_fit2, sensor_cv2)
sensor_cm2 = confusionMatrix(sensor_cv_result2, cv20$classe)
sensor_cm2$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1108    6    0    1    0
##          B    3  749    8    0    1
##          C    1    4  671   14    1
##          D    4    0    4  626    0
##          E    0    0    1    2  719
```


The accuracy is 0.9873 so we get a modest improvement. So we settle on the following variables:


```r
names(sensor_train2)
```

```
##  [1] "accel_belt_x"     "accel_belt_y"     "accel_belt_z"    
##  [4] "roll_arm"         "pitch_arm"        "yaw_arm"         
##  [7] "gyros_arm_x"      "gyros_arm_y"      "gyros_arm_z"     
## [10] "accel_arm_x"      "accel_arm_y"      "accel_arm_z"     
## [13] "roll_dumbbell"    "pitch_dumbbell"   "yaw_dumbbell"    
## [16] "gyros_dumbbell_x" "gyros_dumbbell_y" "gyros_dumbbell_z"
## [19] "accel_dumbbell_x" "accel_dumbbell_y" "accel_dumbbell_z"
## [22] "roll_forearm"     "pitch_forearm"    "yaw_forearm"     
## [25] "gyros_forearm_x"  "gyros_forearm_y"  "gyros_forearm_z" 
## [28] "accel_forearm_x"  "accel_forearm_y"  "accel_forearm_z"
```


# Generating Test Predictions


```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

test = read.csv("pml-testing.csv")
vars = grep("^accel|^gyros_forearm_|^(roll|yaw|pitch)_forearm|^gyros_arm|^(roll|yaw|pitch)_arm|^gyros_dumb|^(roll|yaw|pitch)_dumb", 
    names(test))
sensor_test = test[, vars]
sensor_test_result = predict(sensor_fit2, sensor_test)
name_ts_test_result = predict(name_ts_fit, test[, c("user_name", "raw_timestamp_part_1", 
    "raw_timestamp_part_2")])
confusionMatrix(sensor_test_result, name_ts_test_result)$table
```

```
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 8 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
```

```r

pml_write_files(sensor_test_result)
```


Note that our prediction between sensor and using names and timestamp agree, which is a good sign...

