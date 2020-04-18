# TextClassification

## Experiment 1 : Neural Bag Of Words for Sentiment Analysis 

Under a broader category of text classification, this project like others was an experiment to undertand the power of applying
Convolutional Neural Networks (CNNs) in the area of Natural Language Processing for Text Classification

### Project Goal:

The project's goal is to process a group of documents  generate features from those documents and classify each document 
to the correct category it belongs to. The project was dynamically designed to process large chuncks of text documents.

The larger goal of the different versions of this project is to develop a Text Classifier which if tweaked could 
find applications in the following areas to name a few: 

1) Tagging content or products using categories as a way to improve browsing or to identify related content on company website.
2) Classifying customer feedback based on its sentiment to positive, negative or neutral (sentiment analysis)
3) Classifying customer questions according what product or which part of the product architecture the question regards
4) Classifying content in offer documents according to predefined criteria to for instance quality, price, schedule, etc.
5) Classifying emails into non-spam and spam.

### Inner Working : Simplified

The code written takes in each word document and does the following :
a) Preprocesses the text document to remove any punctuations
b) Keeps only alphanumeric characters
c) Removes all english words which are used frequentyly, does so to keep the essence of the text useful enough for classification
d) Removes any word whose length is shorter than 1 

As each document is processed and curated for classification, each word in that document is added to a vocabulary list which 
would later be used to develop a Deep Learning model for classifying those docuemtns which were used to test the performance
of the model developed

Meanwhile, as Machine Learning/ Deep Learning models work on numbers rather than text or any other data format, each word in 
each document is given a number by a process known as Tokenization. These word-number pairing is then used to transform each
sentence into a group of numbers  where each word is replaced by the number it was given initally during Tokenization.

Next, as the sentences could vary in length they are transformed to uniform length sequences so that it becomes easy for the 
model to process data fed into it.

Model Configurations :

a) Embedding Layer
b) Convolutioanl 1D Layer
c) MaxPooling Layer
d) Flatten Layer
e) Dense Layer ( hidden )
f) Dense Layer ( output )

As each document is processed and curated for classification, each word in that document is added to a vocabulary list which 
would later be used to develop a Deep Learning model for classifying those docuemtns which were used to test the performance
of the model developed

Meanwhile, as Machine Learning/ Deep Learning models work on numbers rather than text or any other data format, each word in 
each document is given a number by a process known as Tokenization. These word-number pairing is then used to transform each
sentence into a group of numbers  where each word is replaced by the number it was given initally during Tokenization.

Next, as the sentences could vary in length they are transformed to uniform length sequences so that it becomes easy for the 
model to process data fed into it.

Model Configurations :

a) Dense Layer ( hidden )
b) Dense Layer ( output )

As the model is trained, each word is converted into a vector ( known as word embedding ) and then those group of vectors are 
classified into categories depending on the target label.

The model is trained on the following configurations of BOW :
a) binary	
b) count	
c) tfidf	
d) freq

Based on the compuational effciency of my laptop, the entire dataset was trained 10 times for each configuration of BOW, and it's 
performance of classifying text documents (using binary configuration) into the right category was 92.6%

The model's performance could be easily lifted by training for longer durations using AWS or using a more deeper architecture.


## Experiment 2 : Embedding + CNN Model for Sentiment Analysis 

Under a broader category of text classification, this project like others was an experiment to undertand the power of applying
Convolutional Neural Networks (CNNs) in the area of Natural Language Processing for Text Classification and expands on the 
limitations of the previous project by replacing BOW approach with ingeniously calculated Word Embeddings.

### Project Goal :

The project's goal is to process a group of documents  generate features from those documents and classify each document 
to the correct category it belongs to. The project was dynamically designed to process large chuncks of text documents.

### Inner Working : Simplified

The code written takes in each word document and does the following :
a) Preprocesses the text document to remove any punctuations
b) Keeps only alphanumeric characters
c) Removes all english words which are used frequentyly, does so to keep the essence of the text useful enough for classification
d) Removes any word whose length is shorter than 1 

As each document is processed and curated for classification, each word in that document is added to a vocabulary list which 
would later be used to develop a Deep Learning model for classifying those docuemtns which were used to test the performance
of the model developed

Meanwhile, as Machine Learning/ Deep Learning models work on numbers rather than text or any other data format, each word in 
each document is given a number by a process known as Tokenization. These word-number pairing is then used to transform each
sentence into a group of numbers  where each word is replaced by the number it was given initally during Tokenization.

Next, as the sentences could vary in length they are transformed to uniform length sequences so that it becomes easy for the 
model to process data fed into it.

Model Configurations :

a) Embedding Layer
b) Convolutioanl 1D Layer
c) MaxPooling Layer
d) Flatten Layer
e) Dense Layer ( hidden )
f) Dense Layer ( output )

As the model is trained, each word is converted into a vector ( known as word embedding ) and then those group of vectors are 
classified into categories depending on the target label.

Based on the compuational effciency of my laptop, the entire dataset was trained 10 times, and it's performance of classifying 
text documents into the right category on test dataset was 88.99%. 

The model's performance could be easily lifted by training for longer durations using AWS or using a more deeper architecture.


## Experiment 3 : n-gram CNN Model for Sentiment Analysis

Taking a final shot at improving the model's performance, this experiment focussed on using an alternate approach to classify
documents into their respective categories. Here, unlike the previous approach we use multiple input channels and parallaly 
calculate embeddings to given an output label for each document.

All the tasks performed prior to modl building are similar to the above experiment, however the model architecture varies 
significantly, as is as follows : 

a) Embedding Layer 1
b) Convolutioanl 1D Layer 1
c) MaxPooling Layer 1
d) Flatten Layer 1

e) Embedding Layer 2
f) Convolutioanl 1D Layer 2
g) MaxPooling Layer 2
h) Flatten Layer 2

i) Embedding Layer 3
j) Convolutioanl 1D Layer 3
k) MaxPooling Layer 3
l) Flatten Layer 3

m) Dense 1 ( merges the three flatten layers )
n) Dense Layer ( hidden )
o) Dropout
p) Dense Layer ( output )

The model was trained was the following batch sizes  : [15,32,64] and epoch : [10,20,30]
The model was trained EC2 instance on p2xlarge instance type and it's performance of classifying text documents into the 
right category on test dataset was 88.99%. 

