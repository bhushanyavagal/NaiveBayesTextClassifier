import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from math import log10

tokenizer = RegexpTokenizer(r'\w+')
stopWordList = stopwords.words('english')
dataset = ""  # Dataset path
classdir = [dirname for dirname in os.listdir(dataset)]

#divide dataset 50:50 and store the filenames in 2 different dictionaries
#trainsetAll is the dictionary of trainset of all classes
#testsetAll is the dictionary of testset of all classes
trainsetAll = {}
testsetAll = {}
for classname in classdir:
    files = [filename for filename in os.listdir(dataset + "\\" + classname)]
    testlist = files[len(files) // 2:]
    trainlist = files[:len(files) // 2]
    trainsetAll[classname] = trainlist
    testsetAll[classname] = testlist
testmainDict = {}
totalVocabSize = 0
mainDict = {}

#Tokenize each file in the dataset to form the bag of words
#Training set will have a dictionary for each class with all the tokens and frequencies in all documents
#Test set will be a dict of a class of dict of tokens for each file in the class
for classname in classdir:
    classlevelDict = {}
    classleveltestDict = {}
    for filename in os.listdir(dataset + "\\" + classname):
        if filename in trainsetAll[classname]:
            try:
                file = open(os.path.join(dataset + "\\" + classname, filename), "r", encoding='ISO-8859-1')
                doc = file.read()
                file.close()
            except IOError:
                print("Cannot open", filename)
            except:
                print("Unexpected error while reading dataset\n")
            doc = doc.lower()
            tokens = tokenizer.tokenize(doc)
            stopWordList = stopwords.words('english')
            tokens2 = [token for token in tokens if token not in stopWordList]
            for token in tokens2:
                classlevelDict[token] = classlevelDict.get(token, 0)+1
        else:
            filedict = {}
            try:
                file = open(os.path.join(dataset + "\\" + classname, filename), "r", encoding='ISO-8859-1')
                doc = file.read()
                file.close()
            except IOError:
                print("Cannot open", filename)
            except:
                print("Unexpected error while reading dataset\n")
            doc = doc.lower()
            tokens = tokenizer.tokenize(doc)
            stopWordList = stopwords.words('english')
            tokens2 = [token for token in tokens if token not in stopWordList]
            for token in tokens2:
                filedict[token] = filedict.get(token, 0)+1
            classleveltestDict[filename] = filedict
    totalVocabSize += len(classlevelDict)
    mainDict[classname] = classlevelDict
    testmainDict[classname] = classleveltestDict

#Calculate likelihood of a term for its own class for all terms in all classes
#This is the part where we train the naive bayes classifier
lhood_Dict = {}
for classname in mainDict:
    classLhoodDict = {}
    classDict = mainDict[classname]
    classSize = sum(classDict.values())
    for token in classDict:
        classLhoodDict[token] = (1 + classDict[token])/(classSize + totalVocabSize)
    lhood_Dict[classname] = classLhoodDict

#Run the Naive bayes classifier on the test set and determine maximum likelihood
numCorrect = 0
numWrong = 0
for classname in testmainDict:
    testClassDict = testmainDict[classname]
    for filename in testClassDict:
        filedict = testClassDict[filename]
        resultsDict = {}
        for classnm in lhood_Dict:
            sumLhood = 0.0
            cdict = lhood_Dict[classnm]
            ctokendict = mainDict[classnm]
            for token in filedict:
                if token in cdict:
                    sumLhood += filedict[token]*log10(cdict[token])
                else:
                    sumLhood += filedict[token]*log10(1/(sum(ctokendict.values()) + totalVocabSize))
            resultsDict[classnm] = sumLhood
        if classname == max(resultsDict, key=resultsDict.get):
            numCorrect += 1
        else:
            numWrong += 1

print("Total correct ",numCorrect)
print("Total Wrong ",numWrong)
print("Accuracy is ",(numCorrect/(numWrong+numCorrect))*100)
