# NaiveBayesTextClassifier
A Naive Bayes text classifier

Please download the data from http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html (Newsgroup Data)

The approach here is to convert each document in the training set into a bag of words model for a particular class and calculate likelihood for each term given its class.
The bags of words for each class in the training set will aid in calculation of the maximum likelihood that determines the class of a test document.
Each test document is tokenized and if the token appears in a class, the likelihood of that token for that class which is already computed will be multiplied or in this case, the log value is added. The sum total for each class is then compared and the highest sum is the class that the document belongs to. This implementation also takes care of new words in the test set and uses Laplace smoothing or add 1 smoothing to deal with them. 

Run instructions:
To run this, please install latest Anaconda distribution along with python 3.5.1 or just the NLTK is also fine.
Change the local path where the program looks for the dataset.

Results:
Total correct: 8403
Total Wrong: 1596
Accuracy: 84.03840384038403
