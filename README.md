# Restaurant Review Semantic Analysis
Here are two models that both ingest restaurant reviews and predict if the review was favorable.

## Status
  
Updated both BERT and TextBlob models on 16 Mar 23

## Project Description

This is a machine learning classification model written in Python, utilizing SciKit-Learn's models to train a classification model on a dataset of restaurant reviews to predict the favorabilty of a review based on the text of the review.   

###  Files
  
The project is divided into one data files (.tsv) and two Jupyter notebooks

####  Data
  
    Restaurant_Reviews.tsv : The dataset of restaurant reviews, available from https://www.kaggle.com/datasets/jurk06/restaurant-dataset
    The dataset contains 1000 samples with one feature (the text of the review in English) and a binary target (1 = positive review, 0= negative review).  The dataset was split evenly between 500 positive reviews and 500 negative reviews.
    
####  Code
  
    BERT_reviewer.ipynb : BERT model, based on the BERT tutorial here: https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb#scrollTo=iCoyxRJ7ECTA.
    
    TextBlob_reviewer.ipynb : TextBlob model, based on the tutorial here:https://www.learnsteps.com/naive-bayesian-text-classifier-using-textblob-python/.


## Execution
Execute either notebook in Jupyter with .tsv file in the same directory.

## Results and Insights
The TextBlob Model produced results between 80%-84% accuracy, evenly split between False Positives and False Negatives (test size = 50)
The BERT Model (using DistilBERT) produced around 88% accuracy (test size = 250)
Running DistilBERT again with a test size of 50 produced accuracy closer to 92%.  the FP/FN rates were 
The BERT Model (using BERT) produced around 91% accuracy (test size = 250)

The TextBlob model was the first model I produced, and probably could have used more techniques to tokenize the words and to use TextBlob functions to cut the words down to the root.  I also expect that using a pre-trained model with BERT enabled BERT to already understand synonyms and negating words (e.g. "not good" is unfavorable sentiment vice positive sentiment), while I suspect that using TextBlob relied only the training set to look for examples of positive or negative language, and a larger dataset would probably produce better results.  

### Examples of False Positives (Unfavorable Review, Favorable Prediction)
I have been to very few places to eat that under no circumstances would I ever return to, and this tops the list. (Expect that "this tops the list" caused the model to predict this was a favorable review)
This place has a lot of promise but fails to deliver. (Expect that "a lot to promise" caused the model to predict this was a favorable review)


### Examples of False Positives (Favorable Review, Unfavorable Prediction)
Nice blanket of moz over top but i feel like this was done to cover up the subpar food. (Not sure why this is a favorable review)
Plus, it's only 8 bucks.

## Credits
All code modified and adapted by Mike Stevens (https://github.com/StevensMR) from the sources listed above
Thanks to Brett Waugh (https://github.com/WaughB) who helped mentor me on my beginning steps on my data science/machine learning journey
