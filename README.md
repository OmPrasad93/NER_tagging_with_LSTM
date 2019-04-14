# NER_tagging_with_LSTM
# Dataset Preparation
This project has been created on ConLL2003 dataset. The datset format has a word followed by its POS_TAG,CHUNK type and NER tags with spacing 
provided as sentence demarcators.
We load the dataset and create a sentence index in our dataframe to demarcate each and every sentence.
The dataset is trasformed into a list of tuples for every sentence wrt its sentence index. For example a sentence like "I am going" is 
transformed into [("I",<TAG>),("am",<TAG>),("going",<TAG>)].
From the format we take the the distinct words and use word2id to convert them into numerical values and same is also done for the tags.
The numerical values are in-turn replaced in the sentence list.
Padding is done for both the word ids and the tag ids.
Here the specific padding length is 130, as it was greater than the longest sentence in train,test and validation sets.
The tag ids are then turned into one hot encoded style arrays.
The data is now ready to be fed into the Keras Sequential model.

# The LSTM model:
The input layer is of shape (130,) as there will be 130 words in each list. Each list denotes a sentence.
second layer is an embedding layer which creates a embedding matrix of dimension 130 from the vocabulary.
In this project we choose a unidirectional LSTM netowrk with dropout of 0.1 and recurrent dropouts of 0.1. 
It is just a primary initialization but the results seem decent.
Output layer is a softmax layer.
The metrics used are accuracy from keras and for f1 score I use Keras_metrics as F1 score,precision and recall are now taken off from Keras.
We fit this model on the training set with the below parameters:-
-batch_size=24
-epochs=3
The accuracy for the model in 3 epochs is 99.5%. But the accuracy is pretty much unusable due to the imbalanced data in the dataset.
So, the F1 score is used which is a harmonic mean of Precision and recall.
The F1 score with 3 epochs is 54.18%. Which should increase given additional epochs.

#Observations:-
Changing the batch size from 24 to 32 did not make any significant changes as such. Increasing it to 
much higher values would slow down the trainiing process. Also decreasing it to really low values might make the gradient descent 
stop at some local minima.
The dropout chosen is low as it should be started from a low value. With the low value combined with only 3 epochs we get an F1 of 54.18%.
The imbalance has a big role to play in the dataset. The "O" tag is has huge number of records compared to any other tag. 
I tried providing a class weight to each but didn't succeed while using them in Keras. Oversampling and bootstrapped approaches 
might not work here as this is a sentence level model and adding words with other tags to a sentence might change the structure of it
therefore rendering the LSTM useless. The class weight seems to be the best way to fix this.
Given an option BI-LSTM with CRFs will be a really good approach to solve preoblems like these. While the Bi-LSTM learns the 
long term relationships from both the sides of the sentence CRFs will be helpful in deciphering the features like Dates,Numbers etc.
BI-LSTM with CNN will also be a good approach for the same.
Specifically for the ConLL2003, the pare suggests a maximum entropy markovian model has been the most succesful.





