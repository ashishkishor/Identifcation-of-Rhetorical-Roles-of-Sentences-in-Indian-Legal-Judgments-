# Machine-learning
Identification of Rhetorical Roles of Sentences in Indian Legal Judgments
Probelm Statement-
We have Supreme court cases ,All cases have sentneces which belongs to either of these 7 classes:
1.Facts   
2.Ruling by Lower Court 
3.Argument 
4.Statute 
5.Precedent
6.Ratio of the decision 
7.Ruling by Present Court 

It's a multi-class classification problem

Proposed Approaches:
This Project consist of 3 models that i have used
1)Classical approach of Bag of Words -Multinomial Naive Bayes
2)Word2Vec word-embedding with BiLSTM classifier
3)Google Pre-trained word-embeddings with BiLSTM classifier

Note-For running the code all sentences should be into one text file 
and for pre-trained Embedding ,download the google-pre-trained Embeddings

Note-Result.csv contains all the sentences.
