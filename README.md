# Neural Machine-Translation/ Language-Modelling


## Neural Language Model - Text Generation

### Project Goal 

Develop a language model to predict the next word in the sequence based on the specific words that have come 
before it in the sequence. Internally, the task is to predict the probability of the next word in the sequence, 
based on the words already observed in the sequence

Neural Language Modelling finds its applications in the following areas :  
1) Machine Translation
2) Spell Correction 
3) Speech Recognition
4) Summarization
5) Question Answering
6) Sentiment analysis

### Inner Working : Simplified

The language modelling task performed here makes use of the text from "The Republic" written by Plato.

The downloaded text book is stripped of the beggining and the end content, and the reamining text is passed through the data 
preprocessing pipeline, which performs the following tasks: 

a) Replace ‘-’ with a white space so we can split words better
b) Split words based on white space
c) Remove all punctuation from words to reduce the vocabulary size (e.g. ‘What?’ becomes ‘What’)
d) Remove all words that are not alphabetic to remove standalone punctuation tokens
e) Normalize all words to lowercase to reduce the vocabulary size

Training the Language Model : 

The model we will train is a neural language model. It has a few unique characteristics:
a) It uses a distributed representation for words so that different words with similar meanings will have a similar representation
b) It learns the representation at the same time as learning the model
c) It learns to predict the probability for the next word using the context of the last 100 words

We will use an Embedding Layer to learn the representation of words, and a Long Short-Term Memory (LSTM) recurrent neural 
network to learn to predict words based on their context.

The cleaned text is tokenized, meaning each word in the text is given a unique number and the words in each line are added to 
the vocabulary list. This vocabulary file would be used to prepare embeddings for each word by the model.


Model Architecture : 

a) Embedding Layer : To represent each word in a vectorized format
b) LSTM Layer (Hidden Layer)
c) LSTM Layer (Hidden Layer)
d) Dense Layer (Hidden Layer)
e) Dense Layer (Outer Layer)

Next, each sequence fed into the model would be uniform length (50 words) and we would then predict the most likely word
following this sequence of 50 words, the model architecture is made dynamic to self train itself and include the predictions
at time step T to include in the predictions for the time step T+1.

## Neural Machine Translation

### Project Goal

The Project Goal is to develop a text-text application for translating input sequences in one language to ouput sequences in 
another language

The project can be expanded to the following areas :
1) Text-to-speech
2) Speech-to-text
3) Speech-to-speech

### Inner Working : Simplified

The dataset used here makes use of German to English terms for language learning available from http://www.manythings.org/.

The text phrases are passed through a preprocessing pipeline which does the following steps : 

1) Remove all non-printable characters.
2) Remove all punctuation characters.
3) Normalize all Unicode characters to ASCII (e.g. Latin characters)
4) Normalize the case to lowercase.
5) Remove any remaining tokens that are not alphabetic

The Cleaned text is then tokenized ( as previously described ) and then the tokenized text is converted into sequence of 
numbers and then padded to the maximum sequence length to be passed into the model. We then use word embedding for the 
input sequences and one hot encode the output sequences.

Model Architecture :

The model used here uses an encoder-decoder LSTM architecture.  In this architecture, the input sequence is encoded by a 
front-end model called the encoder then decoded word by word by a backend model called the decoder.

The model is trained using the efficient Adam approach to stochastic gradient descent and minimizes the categorical loss 
function because we have framed the prediction problem as multiclass classification.

The Layers Used are as follows : 

a) Embedding Layer
b) Bidirectional LSTM ( Hidden Layer )
c) LSTM ( Hidden Layer ) ( 3 stacks )
d) Dropout ( 2 stacks )
e) Dense Layer ( Output Layer )

The model developed translated german text to english with a bleu score of 0.15

