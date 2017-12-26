# Arabic-Handwritten-Text-Image-Recognition
Convert Arabic Handwritten Images to Text

### Problem Statement: 
Given a Arabic handwritten word in image form. Convert it into text form or recognise the word and get the word in text form


### Solution Implemented:
Used 3 layer CNN to learn the features of Arabic Text. Fed that leaning into dynamic_rnn module with LSTM cell to predict the output.


### Results: 
The model is trainable and able to converge on small dataset.


#### File Description:
train.py: to train the model
cnn_lstm_model.py : Implemntation of CNN-LSTM model using tensorflow.
configuration.py: Configuration file for all the hyperparameters and variable
helpers.py: contains helpers function to create batch to feed into model, prepare arabic character set and dictionary


#### Further Work:
Need good amount of GPU resource to train the model on full dataset to get the Results.
