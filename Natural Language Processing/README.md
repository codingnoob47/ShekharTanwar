# Natural Language Processing 

This Sub Directory Contais the Following Folders : 

## 1) NLP Projects 
## 2) Neural Machine Translation Language Modelling Projects
## 3) Text Classification Projects
## 4) Document Search : Locality Sensitive Hashing


The Subfolders have projects on the following NLP Problems :

## NLP Projects 

### Parts Of Speech Tagging :  HMM Tagger ( Hidden Markov Model Tagger)

Usage - Simple Description & Applications : 

In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST), also called grammatical tagging or word-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context — i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. A simplified form of this is commonly taught to school-age children, in the identification of words as nouns, verbs, adjectives, adverbs, etc.

### Topic Modelling : Latent Dirichlet Allocation

Usage - Simple Description & Applications : 

Topic modeling is an unsupervised machine learning technique that's capable of scanning a set of documents, detecting word and phrase patterns within them, and automatically clustering word groups and similar expressions that best characterize a set of documents

## Neural Machine Translation Language Modelling Projects

### Neural Language Model - Text Generation

Usage - Applications : 
1) Machine Translation
2) Spell Correction
3) Speech Recognition
4) Summarization
5) Question Answering
6) Sentiment analysis
  
### Neural Machine Translation - Speech Translation

The project can be expanded to the following areas :

Usage - Applications : 
1) Text-to-speech Translation
2) Speech-to-text Translation
3) Speech-to-speech Translation

## Text Classification Projects

This sub folder contains a series of experiments on Text Data with the following objective : 

The project's goal is to process a group of documents generate features from those documents and classify each document to the correct category it belongs to. The project was dynamically designed to process large chuncks of text documents.

The Experiments are as follows : 

### Neural Bag Of Words for Sentiment Analysis
### Embedding + CNN Model for Sentiment Analysis
### n-gram CNN Model for Sentiment Analysis


Usage - The projects can be expanded and applied in the following areas : 
1) Tagging content or products using categories as a way to improve browsing or to identify related content on company website.
2) Classifying customer feedback based on its sentiment to positive, negative or neutral (sentiment analysis)
3) Classifying customer questions according what product or which part of the product architecture the question regards
4) Classifying content in offer documents according to predefined criteria to for instance quality, price, schedule, etc.
5) Classifying emails into non-spam and spam.



## Document Search using Sentence Embedding + Locality Sensitive Hashing + K Nearest Neghbour

This sub folder contains my code of a document search architecture developed using Pretrained Glove Embeddings for words, which were then used to generate embeddings for the entire document they were a part of

The architecture developed enables user to pass any text realted dataset to look out for similar documents having similar key words. The underlying working is dependent on matching documents with similar embeddings and using K nearest Neighbour to optimize search time

Usage : The project can be modified to include custom created embeddings based on business problem and be used to develop a document search application for tagging similar health records, insurance claims notes and to significantly optimize any business related activity wherein users need to look through mounds of text related data to search for similar information
