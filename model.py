#import for model
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from StringIO import StringIO


creditcardcsv = #path to credit card fraud data
df = pd.read_csv(StringIO(creditcardcsv))


"""
Create training, testing, and validation set from raw data
"""

#split the data based on its label
fraud = df.loc[df['Class']==1]
normal = df.loc[df['Class']==0]

#randomize the order of data
fraud = fraud.reindex(np.random.permutation(fraud.index))
normal = normal.reindex(np.random.permutation(normal.index))

#separate fraud into training, testing, validation
fraud_training = fraud.sample(frac=0.8)

#fraud_temp holds the testing and validation data
fraud_temp = fraud.loc[~fraud.index.isin(fraud_training.index)]
fraud_testing = fraud_temp.sample(frac=0.8)
fraud_validation = fraud_temp.loc[~fraud_temp.index.isin(fraud_testing.index)]

#uncomment to confirm the size of fraud variables
#printlen(fraud_training)
#printlen(fraud_temp)
#printlen(fraud_testing)
#printlen(fraud_validation)

#separate normal into training, testing, validation
normal_training = normal.sample(frac=0.8)
normal_temp = normal.loc[~normal.index.isin(normal_training.index)]
normal_testing = normal_temp.sample(frac=0.8)
normal_validation = normal_temp.loc[~normal_temp.index.isin(normal_testing.index)]

#uncomment to confirm the size of normal variables
#len(normal_training)
#len(normal_temp)
#len(normal_testing)
#len(normal_validation)

#create the full training, testing, and validation data by merging normal and fraud data
training_set = pd.concat([fraud_training, normal_training], axis = 0)
training_set = training_set.reindex(np.random.permutation(training_set.index))
testing_set = pd.concat([fraud_testing, normal_testing], axis = 0)
testing_set = testing_set.reindex(np.random.permutation(testing_set.index))
validation_set = pd.concat([fraud_validation, normal_validation], axis = 0)
validation_set = validation_set.reindex(np.random.permutation(validation_set.index))

#uncomment to confirm the size of training, testing, and validataion sets
#len(training_set)
#len(testing_set)
#len(validation_set)

"""
Create functions that process the data.  
"""

def preprocess_targets(fraud_data):
    """
    Arguments: A dataframe that consists of credit card data
    Returns: Returns the processed labels from the dataframe
    """
    
    label1 = pd.Series((len(normal)/len(fraud))*fraud_data[fraud_data.columns[30]])
    label2 = pd.Series(1-fraud_data[fraud_data.columns[30]])
    
    output_targets = pd.DataFrame({'fraud':label1, 'not fraud':label2})
    
  
    return output_targets

processed_training_labels = preprocess_targets(training_set)
processed_testing_labels = preprocess_targets(testing_set)
processed_validation_labels = preprocess_targets(validation_set)

def preprocess_features(fraud_data):
  """
    Arguments: A dataframe that consists of credit card data
    Returns: Returns processed features from the dataframe
  """

  #normalize the features
  fraud_data_features = fraud_data.columns.values
  for feature in fraud_data_features:
      mean, std = fraud_data[feature].mean(), fraud_data[feature].std()
      fraud_data.loc[:, feature] = (fraud_data[feature] - mean) / std

  processed_features = fraud_data.drop(['Class'], axis=1)

  return processed_features


processed_training_features = preprocess_features(training_set)
processed_testing_features= preprocess_features(testing_set)
processed_validation_features = preprocess_features(validation_set)

"""
Create Neural Network for your model
"""

#define constants for the graph
#Number of inputs or features
FEATURE_SIZE = len(processed_training_features.columns)

#Number of units in hidden layers
HIDDEN1_UNITS = 20
HIDDEN2_UNITS = 15

#Number of labels or targets
NUM_LABELS = 2

# input layer
serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'nn_features': tf.FixedLenFeature(shape=[FEATURE_SIZE], dtype=tf.float32),}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
nn_features = tf.identity(tf_example['nn_features'], name='nn_features')  # use tf.identity() to assign name

#layer 1
weights_l1 = tf.Variable(tf.truncated_normal(shape=(FEATURE_SIZE, HIDDEN1_UNITS), stddev = 0.1 ))
biases_l1 = tf.Variable(tf.truncated_normal([HIDDEN1_UNITS], stddev = 0.1))
hidden1 = tf.nn.sigmoid(tf.matmul(nn_features, weights_l1)+ biases_l1)

#layer 2
weights_l2 = tf.Variable(tf.truncated_normal(shape =(HIDDEN1_UNITS, HIDDEN2_UNITS), stddev = 0.1 ))
biases_l2 = tf.Variable(tf.truncated_normal([HIDDEN2_UNITS], stddev = 0.1))
hidden2= tf.nn.sigmoid(tf.matmul(hidden1, weights_l2)+ biases_l2)

#layer 3 - softmax
weights_l3 = tf.Variable(tf.truncated_normal(shape =(HIDDEN2_UNITS, NUM_LABELS), stddev = 0.1 ))
biases_l3 = tf.Variable(tf.truncated_normal([NUM_LABELS], stddev = 0.1))
softmax_layer = tf.nn.softmax(tf.matmul(hidden2, weights_l3) + biases_l3)

nn_output = softmax_layer
nn_label =tf.placeholder(tf.float32, shape=(None, NUM_LABELS))

"""
Train the model and display the accuracy of the model
"""

#parameters
learning_rate = 0.06
num_epoch = 600
display_step = 30

cost = -tf.reduce_sum(nn_label*tf.log(nn_output))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# predict classes
values, indices = tf.nn.top_k(nn_output, NUM_LABELS)
table = tf.contrib.lookup.index_to_string_table_from_tensor(
    tf.constant(["Fraud", "Normal"]))
prediction_classes = table.lookup(tf.to_int64(indices))

#accuracy equations
result_y_ = 1 - tf.argmax(nn_label, 1)
result_y = 1 - tf.argmax(nn_output, 1)
correct_prediction = tf.equal(result_y, result_y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
total_fraud = tf.reduce_sum(tf.cast(result_y_, tf.float32))
correct_fraud = tf.reduce_sum(tf.cast(result_y * result_y_, tf.float32))
predict_fraud = tf.reduce_sum(tf.cast(result_y, tf.float32))

#set inputs as_matrix()
train_input = processed_training_features.as_matrix()
train_output = processed_training_labels.as_matrix()

test_input = processed_testing_features.as_matrix()
test_output = processed_testing_labels.as_matrix()

validation_input = processed_validation_features.as_matrix() 
validation_output = processed_validation_labels.as_matrix()

#build signature_def_map
classification_inputs = tf.saved_model.utils.build_tensor_info(
  serialized_tf_example)
classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
  indices)
classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)


classification_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
    inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs},
    outputs={
      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
        classification_outputs_classes,
      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
        classification_outputs_scores
    },
    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

tensor_info_x = tf.saved_model.utils.build_tensor_info(nn_features)
tensor_info_y = tf.saved_model.utils.build_tensor_info(nn_output)

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'x': tensor_info_x},
    outputs={'scores': tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder = tf.saved_model.builder.SavedModelBuilder("/path_for_local_model")


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            classification_signature,
      },
      legacy_init_op=legacy_init_op)
  
  for epoch in range(num_epoch):
           
    sess.run([optimizer], feed_dict = {nn_features: train_input, nn_label: train_output})
    
    if (epoch % display_step) == 0:
        training_sess = sess.run([cost, accuracy, correct_fraud, total_fraud, predict_fraud], feed_dict = {nn_features: train_input, nn_label: train_output})

        print("training: ", training_sess)

  testing_sess = sess.run([cost, accuracy, correct_fraud, total_fraud, predict_fraud], feed_dict={nn_features: test_input, nn_label: test_output})
  print("test set:")
  print(testing_sess)
  
  validation_sess = sess.run([cost, accuracy, correct_fraud, total_fraud, predict_fraud], feed_dict={nn_features: validation_input, nn_label: validation_output})
  print("validation set:")
  print(validation_sess)
  
builder.save()



