"""
This Simplified CNN is to be used as a proof of concept; after 1000 Steps, this CNN should reach around 0.4 Accuracy in classifying among the 10 classes in Cifar10.
A greater number of steps, as well as a more complex neural architecture is needed to increase the Accuracy so as to be good enough for practical uses.
"""

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO) # Set how much Logging Info you want to be printed out during the Training, Evaluation of the Model.


# Create a Model Function that would be eventually used to create our Estimator.
# Arguments: 'features' will be the Inputted Images in numpy Array form; 'labels' will be a numpy Array that hold the labels of the data in 'features'; 'mode' can be TRAIN,EVALUATE,or PREDICT [like enum]. 
def cnn_model_fn(features, labels, mode):

    # Get the Data from the argument 'features' and shape it in [BATCH_SIZE, 32, 32, 3] Size Array
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])  # The '3' in the Size refers to the Number of Channels in the Input Image, RGB will have 3 Channel. 

    # Create a Convolutional Layer; Note that 'padding=same' means that the Height and Width of the Output from Convolutional Layer will be Same as the Input to the Convolutional Layer i.e. the Convolution will not cause the Shrinking of the Input Image due to Zero Padding at the Borders of the Input Image.    
    # Output Image of this Convolutional Layer will be [BATCH_SIZE, 32, 32, 64]. The last 32 is the Number of Filers
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)   # ReLU Activation function max(0, f) will be applied.

    # Another Convolutional Layer
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)  


    # Flatten/Reshape the 2nd Convolution Layer so as to prepare for the Next, Fully Connected Layer.
    flat = tf.reshape(conv2, [-1, 32 * 32 * 64])  # The Reshaped Image will be of the Size [BATCH_SIZE, 32768]

    # Create a Fully Connected Layer that has 384 Neurons/Nodes. And it uses the ReLU Activation Function.
    # Output Image of this Fully Connected Layer will be of the Size [BATCH_SIZE, 384]
    dense1 = tf.layers.dense(inputs=flat, units=384, activation=tf.nn.relu)

    # Another Fully Connected Layer
    dense2 = tf.layers.dense(inputs=dense1, units=192, activation=tf.nn.relu)

    
    # The Logit Layer is the Last Fully Connected Layer will be output the Results of the Neuron Network.
    # There is a Total of 10 Neurons/Node in this Layer.
    # Output Image of this Logit Layer will be of the Size [BATCH_SIZE, 10]
    logits = tf.layers.dense(inputs=dense2, units=10)
    
    # Use a Dict called 'predictions' to store the Results of the 'logit layer' i.e. Store the Most Probable Class for each Example in the Batch in 'classes' and the Probabilities of each of the 10 Classes for each Example in the Batch using the 'probabilities'.
    # tf.argmax() will give us the Most Probable Class.
    # tf.nn.softmax() will apply the Softmax Function [not the Softmax Cost Function] to give us the estimated probabilities for each of the 10 Classes, for each of the Examples in the Batch
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }



    # Depending on what is the 'mode' [an Argument to this Function], different Specifications of the Estimator will be returned by this Function

    # If asked to Predict, just return the Predictions in 'predictions'
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Forming the Loss Function [Cost Function] 'loss' :
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10) # Get one_hot version of the 'labels' Argument
    loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits) # Use the one-hot Label to get our Loss Function [In Details, we first apply the SoftMax on the Logit Layer's Results to get the estimated probabilities of each Example in each Class, then we find the Cross Entropy using the correct one-hot 'labels' with the estimated probabiltiies to find the Cost].

    # If asked to Train, we will do Gradient Descent ['train_op' is an Operation that does Gradient Descent]:
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Form an Evaluation Metrics to evalute how well our Model is doing:
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    # If asked to Evaluate, we will return the following:
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main(unused_argv):

    # Load Cifar-10 Dataset
    data_dir = '/Users/chankennard/Desktop/Desktop/MLTest/own_tensorflow_projects/Vanilla_CNN_cifar10/cifar-10-batches-py'
    # For this Vanilla CNN, we will only use 2 out of the 6 batches/parts of the Cifar10 Dataset
    # Loading the 1st batch - 'data_batch_1'
    data_batch_1_dir = data_dir + '/data_batch_1' 
    data_batch_1 = unpickle(data_batch_1_dir)
    data_batch_1_data = data_batch_1[b'data']   # a 10000x3072 numpy; each row is a 32x32 colored image; The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    data_batch_1_data = np.reshape(data_batch_1_data,(10000,1024,3), 'F') # Reshape to get the 3 channels into a structure
    data_batch_1_labels = data_batch_1[b'labels']   

    data_batch_2_dir = data_dir + '/data_batch_2' 
    data_batch_2 = unpickle(data_batch_2_dir)
    data_batch_2_data = data_batch_2[b'data']   # a 10000x3072 numpy; each row is a 32x32 colored image; The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    data_batch_2_data = np.reshape(data_batch_2_data,(10000,1024,3), 'F') # Reshape to get the 3 channels into a structure
    data_batch_2_labels = data_batch_2[b'labels']

    data_batch_3_dir = data_dir + '/data_batch_3' 
    data_batch_3 = unpickle(data_batch_3_dir)
    data_batch_3_data = data_batch_3[b'data']   # a 10000x3072 numpy; each row is a 32x32 colored image; The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    data_batch_3_data = np.reshape(data_batch_3_data,(10000,1024,3), 'F') # Reshape to get the 3 channels into a structure
    data_batch_3_labels = data_batch_3[b'labels']

    data_batch_4_dir = data_dir + '/data_batch_4' 
    data_batch_4 = unpickle(data_batch_4_dir)
    data_batch_4_data = data_batch_4[b'data']   # a 10000x3072 numpy; each row is a 32x32 colored image; The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    data_batch_4_data = np.reshape(data_batch_4_data,(10000,1024,3), 'F') # Reshape to get the 3 channels into a structure
    data_batch_4_labels = data_batch_4[b'labels']


    data_batch_5_dir = data_dir + '/data_batch_5' 
    data_batch_5 = unpickle(data_batch_5_dir)
    data_batch_5_data = data_batch_5[b'data']   # a 10000x3072 numpy; each row is a 32x32 colored image; The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    data_batch_5_data = np.reshape(data_batch_5_data,(10000,1024,3), 'F') # Reshape to get the 3 channels into a structure
    data_batch_5_labels = data_batch_5[b'labels']


    data_batch_data = np.vstack((data_batch_1_data, data_batch_2_data))
    data_batch_data = np.vstack((data_batch_data, data_batch_3_data))
    data_batch_data = np.vstack((data_batch_data, data_batch_4_data))
    data_batch_data = np.vstack((data_batch_data, data_batch_5_data))
    
    data_batch_labels = np.hstack((data_batch_1_labels, data_batch_2_labels))
    data_batch_labels = np.hstack((data_batch_labels, data_batch_3_labels))
    data_batch_labels = np.hstack((data_batch_labels, data_batch_4_labels))
    data_batch_labels = np.hstack((data_batch_labels, data_batch_5_labels))
    
    # Loading the 6th batch - 'test_batch'
    test_batch_dir = data_dir + '/test_batch'
    test_batch = unpickle(test_batch_dir)
    test_batch_data = test_batch[b'data']    # a 10000x3072 numpy; each row is a 32x32 colored image; The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    test_batch_data = np.reshape(test_batch_data,(10000,1024,3), 'F') # Reshape to get the 3 channels into a structure
    test_batch_labels = test_batch[b'labels']
    

    # A series of numpy arrays [4 arrays in total]:
    train_data = np.asarray(data_batch_data, dtype=np.float32)  # Training Set
    train_labels = np.asarray(data_batch_labels, dtype=np.int32) # Labels for the Training Set
    eval_data = np.asarray(test_batch_data, dtype=np.float32)  # Returns Evaluation Set
    eval_labels = np.asarray(test_batch_labels, dtype=np.int32) # Labels for the Evaluation Set


    # From the Evaluation Set, get 3 first Elements to form our Test Set and its Labels
    test_data = eval_data[0:3]
    test_label = eval_labels[0:3]


    # Create the Estimator using the 'cnn_model_fn' that we created way above. [To create an Estimator, we need to first build the model function 'cnn_model_fn]
    # Estimators deal with all the details of creating computational graphs, initializing variables, training the model [thus Estimators also handle running the Sessions] and saving checkpoint and logging files for Tensorboard behind the scene. 
    cifar10_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model") # model_dir can be any directory that you like.


  # Set up logging for predictions
  # Log the values in the "Softmax" tensor using the label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"} # 'probabilities' is any name that you wish to use; 'softmax_tensor' is the Name of a Tensor in the Tensorflow Graph 
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50) # Create the Logging Hook

    # Train the model
    # First, we build an Input Function that contains/hold the Training Data and its Labels
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Now, we call the 'train' method of our Estimator [mnist_classifier]. Notice that we added in the Logging Hook as an argument.
    cifar10_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])


    # Evaluate the model and print results
    # Similarly, we have to first build an Input Function that contains/hold the Evaluation Data and its Labels
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    
    # Now, we call the 'evaluate' method of our Estimator [mnist_classifier]
    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


    # Predict using the Model
    # Similarly, we building an Input Function that contains/hold the Test Data and its Labels
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x":test_data},
        y = test_label,
        num_epochs=1,
        shuffle=False)

    # Now, we will call the 'predict' method of our Estimator [mnist_classifier]
    test_results = cifar10_classifier.predict(input_fn = test_input_fn)
    for i in test_results:
        print(i)
    print("actual ans:")
    print(test_label)

# Starting Point of the Program;
# If this Program is ran directly as a Source File, the following If-Statement will be true
# tf.app.run() will run the 'main()' function above.
if __name__ == "__main__":
    tf.app.run()
    
