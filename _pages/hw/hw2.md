---
layout: page
permalink: /homework/hw2
title: HW 2 - Language Modeling and Part of Speech Tagging
---

_TA in charge: Yilun Zhao_

Published: [TODO]

Due: [TODO]

### Introduction
In this assignment we will:
1. Go through the basics of NLTK, the most popular NLP library for Python; 
2. Develop and evaluate several language models;
3. Develop and evaluate a full part-of-speech (POS) tagger using the Viterbi algorithm; 
4. Train a BiLSTM-based POS tagger;

This document is structured in the following sequence: 
1. [Basic NLTK Tutorial](#1-basic-nltk-tutorial)
2. [Provided Files and Report](#2-provided-files-and-report)
3. [Part A: Language Models](#part-a-language-models-6-points) (2 points)
4. [Part B: Implementing Naive POS Tagging](#part-b-implementing-naive-pos-tagging) (8.5 points)
5. [Part C: Training Neural POS Tagging](#part-c-training-neural-pos-tagging) (2.5 points)
6. [Homework Submission](#homework-submission) 

Total: 13 points



### 1. Basic NLTK Tutorial
The [Natural Language Toolkit (NLTK)](https://www.nltk.org/) is the most popular NLP library for python. NLTK has already been installed on the Zoo, so you do not have to install it yourself. If you followed the “Environment Setup” section, you should be good to start using NLTK now.

Now let’s walk through some simple NLTK use cases that concern this assignment. For that, you will need to first open a python interactive shell. Just type `python` or `python3` on the command line and you should see the interactive shell start.

#### Step 1: Import the NLTK package
To use the NLTK package, include the following line at the beginning of your code (or in this case just type directly into the interactive shell):
```python       
import nltk
```

#### Step 2: Tokenization
To tokenize means to break a continuous string into tokens (usually words, but a token could also be a symbol, punctuation, or other meaningful unit). In NLTK, text can be tokenized using the word_tokenize () method. It returns a list of tokens that will be the input for many methods in NLTK.
```python
sentence = "At eight o’clock on Thursday morning on Thursday morning on Thursday
morning."
tokens = nltk.word_tokenize(sentence)
```

#### Step 3: N-grams Generation
An n-gram (in the context of this assignment) is a contiguous sequence of n tokens in a sentence. The following code returns a list of bigrams and a list of trigrams. Each n-gram is represented as a tuple in python (if you are not familiar with Python tuples, read the [Python tuple doc page](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)).

```python
bigram_tuples = list(nltk.bigrams(tokens))
trigram_tuples = list(nltk.trigrams(tokens))
``` 

We can calculate the count of each n-gram using the following code:

```python
count = {item: bigram_tuples.count(item) for item in set(bigram_tuples)}
``` 

Or we can find all the distinct n-grams that contain the word “on”:

```python
ngrams = [item for item in set(bigram_tuples) if "on" in item]
```

If you find it hard to understand the examples above, read about [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) and [dictionary comprehension](https://docs.python.org/3/tutorial/datastructures.html#dictionaries). List/dictionary comprehensions are a way of executing iterations in one line that may be very useful and convenient. Besides making coding easier, learning them will also help you understand code written by other python programmers.

**Note**: If you want to calculate the counts in a faster way, you can use a Counter object from python’s built-in collections library via the following code:

```python
from collections import Counter
count = Counter(bigram_tuples)
```

The dictionary comprehension code can take in the worst-case O(n2) time, whereas the Counter object runs in O(n). If you’re curious, think about why and how this is possible. For more, please check: [https://docs.python.org/3.6/library/collections.html#collections.Counter](https://docs.python.org/3.6/library/collections.html#collections.Counter). You can always use other methods/objects to implement instead of Counter.

#### Step 4: Default POS Tagger (Non-statistical)
The most na ̈ıve way of tagging parts-of-speech is to assign the same tag to all the tokens. This is exactly what the NLTK default tagger does. Although inaccurate and arbitrary, it sets a baseline for taggers, and can be used as a default tagger when more sophisticated methods fail.

In NLTK, it’s easy to create a default tagger by indicating the default tag in the constructor.

```python
default_tagger = nltk.DefaultTagger("NN")
tagged_sentence = default_tagger.tag(tokens)
```

Now we have our first tagger. NLTK can help if you need to understand the meaning of a tag.

```python
# Show the description of the tag "NN"
nltk.help.upenn_tagset("NN")
```

#### Step 5: Regular Expression POS Tagger (Non-statistical)
A regular expression tagger maintains a list of regular expressions paired with a tag (see the Wikipedia article for more information about regular expressions: http://en.wikipedia.org/wiki/Regular_expression). The tagger tries to match each token to one of the regular expressions in its list such that the token receives the tag that is paired with the first matching regular expression. `None` is given to a token that does not match any regular expression.

To create a Regular Expression Tagger in NLTK, we provide a list of pattern-tag pairs to the appropriate constructor. Example:

```python
patterns = [(r".*ing$", "VBG"),(r".*ed$", "VBD"),(r".*es$", "VBZ"),(r".*ed$", "VB")] regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(tokens)
```

See how many affix patterns you can come up with for your `regexp_tagger` up to a maximum of 10 rules. In what situations might this `regexp_tagger` report the wrong tags? Include your rules and some example situations this in your `HW2_Report.ipynb` file (see below).

#### Step 6: N-gram HMM Tagger (Statistical)
lthough there are many different kinds of statistical taggers, we will only work with Hidden Markov Model (HMM) taggers in this assignment.

Like every statistical tagger, n-gram taggers use a set of tagged sentences, known as the training data, to create a model that is used to tag new sentences. In NLTK, a sentence of the training data must be formatted as a list of tuples, where each tuple is a pair in word-tag format (see example below).

```python
[("The", "AT"), ("Fulton", "NP-TL"), ("County", "NN-TL")]
```

NLTK already provides corpora formatted this way. In particular, we are going to use the Brown corpus.

```python
# import the corpus from NLTK and build the training set from sentences in "news"
from nltk.corpus import brown
training = brown.tagged_sents(categories="news")
# Create Unigram, Bigram, Trigram taggers based on the training set.
unigram_tagger = nltk.UnigramTagger(training)
bigram_tagger = nltk.BigramTagger(training)
trigram_tagger = nltk.TrigramTagger(training)
```

Although we could also build 4-gram, 5-gram, etc. taggers, trigram taggers are the most popular model. This is because a trigram model is an excellent compromise between computational complexity and performance.

#### Step 7: Combination of Taggers
A tagger fails when it cannot find a best tag sequence for a given sentence. For example, one situation when an n-gram tagger will fail is when it encounters an OOV (out of vocabulary) word not seen in the training data; the tagger will tag the word as None. One way to handle tagger failure is to fall back to an alternative tagger if the primary one fails. This is called “using back off.” One can easily set a hierarchy of taggers in NLTK as follows:

```python
default_tagger = nltk.DefaultTagger("NN")
bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
```

#### Step 8: Tagging Low Frequency Words
Low frequency words are another common source of tagger failure, because an n-gram that contains a low frequency word and is found in the test data might not be found in the training data. One method to resolve this tagger failure is to group low frequency words. For example, we could substitute the token `_RARE_` for all words with frequency lower than 0.05% in the training data. Any words in the development data that were not found in the training data could then be treated instead as the token `_RARE_`, thereby allowing the algorithm to assign a tag. If we wanted to add another group, we could substitute the string `_NUMBER_` for those rare words that represent a numeral. When tagging the test data, we could substitute `_NUMBER_` for all tokens that were unseen in the training data and represent a numeral. We will use this technique later in this assignment.

### 2. Provided Files and Report

In this assignment we will be using the Brown Corpus, which is a dataset of English sentences compiled in the 1960s. We have provided this dataset to you so you don’t have to load it yourself from NLTK.

Besides data, we are also providing code for evaluating your language models and POS tagge for the assignment. In this assignment you should not create any new code files, but rather just fill in the functions in the skeleton code.

We have provided the following files:
- `data/Brown_train.txt` - Untagged Brown training data
- `data/Brown_tagged_train.txt` - Tagged Brown training data
- `data/Brown_dev.txt` - Untagged Brown development data
- `data/Brown_tagged_dev.txt` - Tagged Brown development data
- `data/Sample1.txt` - Additional sentences for part A
- `data/Sample2.txt` - More additional sentences for part A
- `data/wikicorpus_tagged_train.txt` - Tagged Wikicorpus training data
- `data/wikicorpus_dev.txt` - Untagged Wikicorpus development data
- `data/wikicorpus_tagged_dev.txt` - Tagged Wikicorpus development data
- `perplexity.py` - A script to analyze perplexity for part A
- `pos.py` - A script to analyze POS tagging accuracy for part B
- `bilstm_tagger_model.py` - A neural POS tagger model
- `solutionsA.py` - Skeleton code for part A
- `solutionsB.py` - Skeleton code for part B
- `solutionsB7.py` - A script to evaluate a Spanish tagger for B7
- `solutionsC.py` - A script to evaluate the neural tagger from for part C
- `HW2_Report.ipynb` - A python notebook template for your report
- `output/` - Directory where answers to part A, B, and C are stored

The only files that you should modify throughout the whole assignment are `solutionsA.py`, `solutionsB.py`, and `HW2_Report.ipynb`.

#### Data Files Format
The untagged data files have one sentence per line, and the tokens are separated by spaces. The tagged data files are in the same format, except that, instead of tokens separated by spaces, those files have `TOKEN/TAG` pairs separated by spaces.

#### Report
Before starting the assignment, go to `HW2_Report.ipynb` and include your name and netID in the header. Throughout the assignment, you will be asked to include specific output or comment on specific aspects of your work into `HW2_Report.ipynb`. We recommend filling the `HW2_Report.ipynb` file as you go through the assignment, as opposed to starting the report afterwards.

In this report it is not necessary to include introductions and/or explanations, other than the ones explicitly requested throughout the assignment.

The first thing to report in the `HW2_Report.ipynb` is the affix patterns you came up with for your `regexp_tagger` and some situations (with brief explanation) where the tagger didn’t work.

### Part A: Language Models
In this part of the assignment, you will be filling the `solutionsA.py` file. Open the file and notice there are several functions with a `#TODO`` comment; you will have to complete those functions. To understand the general workflow of the script, read the `main()` function but do not modify it. You also shouldn’t import any additional libraries/functions beyond what is already provided, but you also don’t have to use everything that’s provided if you can find a solution that works without something.

#### A1 Evaluating N-Gram Model Performance and Calculating Corpus Perplexity (1 point)
Use your models to find the log-probability, or score, of each sentence in the Brown training data with each n-gram model. This corresponds to implementing the `score()` function.

Make sure to accommodate the possibility that you may encounter in the sentences an n-gram that doesn’t exist in the training corpus. If you find any n-gram that was not in the training sentences, set the whole sentence log-probability to the constant `MINUS_INFINITY_SENTENCE_LOG_PROB`.

The code will output scores in three files: `output/A1.uni.txt`, `output/A1.bi.txt`, `output/A1.tri.txt`. These files simply list the log-probabilities of each sentence for each different model. Here’s what the first few lines of each file look like:

`A1.uni.txt`:
```
-178.7268354828613
-259.8586443200647
-143.330429890216
```

`A1.bi.txt`:
```
-92.10399842763805
-132.09662640739958
-90.1859108420144
```

`A1.tri.txt`:
```
-26.180045341283773
-59.85310080740878
-42.83924489495398
```

Now, you need to run our perplexity script, `perplexity.py`, on each of these files. This script will count the words of the corpus and use the log-probabilities computed by you to calculate the total perplexity of the corpus. To run the script, the command is:
```
python perplexity.py <file of scores> <file of sentences that were scored>
```

where `<file of scores>` is one of the A1 output files and `<file of sentences that were scored>` is data/Brown_train.txt. Include the perplexity of the corpus for the three different models in your HW1_Report . Here’s what our script printed when `<file>` was `output/A1.uni.txt` (truncated at four decimal places).

```
python perplexity.py output/A1.uni.txt data/Brown_train.txt
The perplexity is 1052.4865
```

#### A2: Identifying Brown Dataset Sample via Perplexity Scores (1 point)
Both `data/Sample1.txt` and `data/Sample2.txt` contain sets of sentences; one of the files is an excerpt of the Brown training dataset. Use your model to score the sentences in both files. Our code outputs the scores of each into `output/Sample1_scored.txt` and `output/Sample2_scored.txt`. Run the perplexity script on both output files and include the perplexity output of both samples in your `HW2_Report.ipynb`. 

*Use these results to make an argument for which sample belongs to the Brown dataset and which does not.*

### Part B: Implementing Naive POS Tagging
In this part of the assignment, you will be filling the `solutionsB.py` file. Open the file and notice there are several functions with a `#TODO` comment; you will have to complete those functions. To understand the general workflow of the script, read the `main()` function, but do not modify it. 

#### B1: Data Preparation - Splitting Words and Tags in the Brown Corpus (1 point)
First, you must separate the tags and words in `Brown_tagged_train.txt`. This corresponds to implementing the `split_wordtags()` function. You’ll want to store the sentences without tags in one data structure, and the tags alone in another (see instructions in the code). Make sure to add sentence start and stop symbols to both lists (of words and tags). Use the constants `START_SYMBOL` and `STOP_SYMBOL` already provided. You don’t need to write anything on `HW2_Report.ipynb` about this question.

Hint: make sure you accommodate words that themselves contain backslashes – i.e., `1/2` is encoded as `1/2/NUM` in tagged form; make sure that the token you extract is `1/2` and not `1`.

#### B2: Calculating Tag Trigram Probabilities (1 point)
Now, calculate the trigram probabilities for the tags. This corresponds to implementing the calc_trigrams () function. The code outputs your results to a file `output/B2.txt`. Here are a few lines (not contiguous) of this file for you to check your work:
```
TRIGRAM * * ADJ -5.205575150818023
TRIGRAM ADJ . X -9.996120363033047
TRIGRAM NOUN DET NOUN -1.264527106474862
TRIGRAM X . STOP -1.9292269255866192
```


After you checked your algorithm is giving the correct output, add to your `HW2_Report.ipynb` the log probabilities of the following trigrams:
```
TRIGRAM CONJ ADV ADP
TRIGRAM DET NOUN NUM
TRIGRAM NOUN PRT PRON
```

**Note**: you might wish to reuse a function you wrote in part A to make your life easier.

#### B3: Data Preprocessing for Handling Rare Words (1 point)
The next step is to implement a smoothing method. To prepare to add smoothing, replace every word that occurs five or fewer times with the token specified in the constant `RARE_SYMBOL`. This corresponds to implementing the `calc_known()` and `replace_rare()` functions.

First, you will create a list of words that occur more than five times in the training data. When tagging, any word that does not appear in this list should be replaced with the token in `RARE_SYMBOL`. You don’t need to write anything on `HW2_Report.ipynb` about this question. The code outputs the new version of the training data to `output/B3.txt`. Here are the first two lines of this file:

```
At that time highway engineers traveled rough and dirty roads to accomplish
their duties .
_RARE_ _RARE_ vehicles was a personal _RARE_ for such employees , and the matter
 of providing state transportation was felt perfectly _RARE_ .
```

**Hint**: if you use a set instead of a list to store frequently occurring words, this operation will be faster. A set is python’s implementation of a hash table and has constant time membership checking, as opposed to a list, which has linear time checking.

#### B4: Calculating Emission Probabilities in the Preprocessed Dataset (1 point)
Next, we will calculate the emission probabilities on the modified dataset. This corresponds to imple- menting the `calc_emission()` function. The code outputs your results to a file `output/B4.txt`. Here are a few lines (not contiguous) from this file for you to check your work:
```
America NOUN -10.9992
Columbia NOUN -13.5599
New ADJ -8.1884
York NOUN -10.7119
```

After you check that your algorithm is giving the correct output, add to your `HW2_Report.ipynb` the log probabilities of the following emissions (note words are case-sensitive):

```
**
Night NOUN 
Place VERB 
prime ADJ 
STOP STOP 
_RARE_ VERB
```

#### B5: Implementing Viterbi Algorithm for HMM Taggers (2 points)

Now, implement the Viterbi algorithm for HMM taggers. The Viterbi algorithm is a dynamic program- ming algorithm that has many applications. For our purposes, the Viterbi algorithm is a comparatively efficient method for finding the highest scoring tag sequence for a given sentence. Please read about the specifics about this algorithm in section 8.4 of the book.

**Note**: your book uses the term “state observation likelihood” for “emission probability” and the term “transition probability” for “trigram probability.”

Using your emission and trigram probabilities, calculate the most likely tag sequence for each sentence in `Brown_dev.txt`. This corresponds to implementing the `viterbi()` function. Your tagged sentences will be outputted to `output/B5.txt`. The first two tagged sentences should look like this:

```
He/PRON had/VERB obtained/VERB and/CONJ provisioned/VERB a/DET veteran/ADJ ship/
NOUN called/VERB the/DET Discovery/NOUN and/CONJ had/VERB recruited/VERB a/DET
crew/NOUN of/ADP twenty-one/NOUN ,/. the/DET largest/ADJ he/PRON had/VERB ever/
ADV commanded/VERB ./.
The/DET purpose/NOUN of/ADP this/DET fourth/ADJ voyage/NOUN was/VERB clear/ADJ
./.
```

Note that, while the output doesn’t have the `_RARE_` token, you still have to count unknown words as a `_RARE_` symbol to compute probabilities inside the Viterbi Algorithm.

When exploring the space of possibilities for the tags of a given word, make sure to only consider tags with emission probability greater than zero for that given word. Also, when accessing the transition probabilities of tag trigrams, use -1000 (constant `LOG_PROB_OF_ZERO` in the code) to represent the log-probability of an unseen transition.

Once you run your implementation, use the part of speech evaluation script pos.py to compare the output file with `Brown_tagged_dev.txt`. Include the accuracy of your tagger in the `HW2_Report.ipynb` file. To use the script, run the following command:

```
python pos.py output/B5.txt data/Brown_tagged_dev.txt
```

This is the result we got with our implementation of the Viterbi algorithm:
```
Percent correct tags: 93.3250
```

Do what you can to make your algorithm as efficient as possible! While we won’t give you specifics, you should think about the order in which you check the words of the trigrams, which words and/or tags you can ignore, etc. For reference, our solution runs in 11-12 seconds. This algorithm is tricky enough to implement, let alone optimize, so don’t be disheartened if you can’t get it to run at lightning speed! While we will give points [TO CHECK] for efficiency, it’ll only be a small portion of your assignment grade.

#### B6: Implementing  NLTK’s Trigram Tagger Set (1.5 point)
Finally, create an instance of NLTK’s trigram tagger set to back off to NLTK’s bigram tagger. Let the bigram tagger itself back off to NLTK’s default tagger using the tag `NOUN`. Implement this in the `nltk_tagger()` function. The code outputs your results to a file `output/B6.txt`, and this is how the first two lines of this file should look:

```
He/NOUN had/VERB obtained/VERB and/CONJ provisioned/NOUN a/DET veteran/NOUN ship
/NOUN called/VERB the/DET Discovery/NOUN and/CONJ had/VERB recruited/NOUN a/DET
crew/NOUN of/ADP twenty-one/NUM ,/. the/DET largest/ADJ he/PRON had/VERB ever/
ADV commanded/VERB ./.
The/NOUN purpose/NOUN of/ADP this/DET fourth/ADJ voyage/NOUN was/VERB clear/ADJ
./.
```

Use `pos.py` to evaluate the NLTK’s tagger accuracy and put the result in your `HW2_Report.ipynb`. This is the accuracy that we got with our implementation:

```
Percent correct tags: 88.0399
```

#### B7: POS Tagging for Spanish (1 point)
In this part, you will be using `solutionsB2.py`. To understand the general workflow of the script, read the `main()` function, but do not modify it. Note that `solutionsB2.py` uses functions from `solutionsB.py`. 

This part uses a corpus other than the Brown corpus. The training data comes from Wikicorpus, a trilingual corpus using passages from Wikipedia. We will be using the Spanish portion to train and evaluate a part- of-speech tagger.

As you may suspect, different languages may use different tagsets. The data from Wikicorpus uses the Parole tagset, which is similar to the Brown corpus’s tagset, but also has tags that are not used in English (such as `DE` for determiner, exclamatory; `Fia` for punctuation, ¿; `VS*` for semi-auxiliary verbs). For instance, the Brown corpus distinguishes between adverbs, comparative adverbs, and superlative adverbs (I run fast, but Miles runs faster, and Drago runs fastest.) The Parole tagset does not differentiate between such adverbs, but instead distinguishes between regular adverbs and negative adverbs (nunca - never). This particular tag may be helpful in Spanish since some sentences use double negatives for emphasis (“No conozco a nadie” word-for-word translates to “I do not know nobody” and actually means “I don’t know anybody”.)

You can use your `solutionsB2.py` to train and evaluate a part-of-speech tagger in Spanish, with normal trigrams. To do this, run the following (it takes my code 3 minutes to finish running):
```
python solutionsB2.py
```

This program will likely take a lot longer to run than `solutionsB.py`. Run the following command to find the accuracy:
```
python pos.py output/C5.txt data/wikicorpus_tagged_dev.txt
```

In your `HW2_Report.ipynb`, record the accuracy of your tagger for Spanish. Our implementation of the Spanish tagger achieved the following performance (to three decimal places):

```
Percent correct tags: 84.472
```

In your `HW2_Report.ipynb`, answer the following questions:

```
Question 1: The Spanish dataset takes longer to evaluate. Why do you think this is the case?

Question 2: What are aspects or features of a language that may improve tagging accuracy that are not captured by the tagged training sets?
```

### Part C: Training Neural POS Tagging
In this part of the assignment, you will train a neural network-based model for POS tagging. The implementation utilizes a neural network known as bidirectional LSTM, which is a method for sequence learning. For this assignment, we have already provided the implementation of the model and training in `bilstm_tagger_model.py``. Your task is to understand the code, answer relevant questions regarding the model's design and implementation, and then proceed to train and evaluate the model using the provided dataset.

#### C1: Understanding the Implementation (1.5 point)
Read through the code in `neural_tagger_model.py`. You should understand the general structure of the model and how it is trained. In your `HW2_Report.ipynb`, answer the following questions:

- Q1: What is the purpose of the `simplify_token` function in data preprocessing?
- Q2: What is the purpose of introducing the special PAD symbol?
- Q3: What is the initialization strategy of the word embedding layer in our implementation? (Hint: Refer to line 110)

#### C2: Training the Model (0.5 point)
In this part, you will train the model using the provided training data. To do this, run the following command:
```
python bilstm_tagger_model.py
```
The model will be trained for 5 epochs. After training, the model will be saved to `tagger.pt.model`. You need to attached this file to your submission.

#### C3: Evaluating the Model (0.5 point)
After training, you will evaluate the model using the provided development data. To do this, run the following command:
```
python solutionsC.py
```
This will output the accuracy of the model. Record the accuracy in your `HW2_Report.ipynb`. Our implementation of the neural tagger achieved the following performance (to three decimal places):

```
Test Accuracy: 93.727
```
Test accuracy greater than 90.0 will receive full credit. 


### Homework Submission
Before submission, run all of your code within `HW2_Report.ipynb` one more time. Make sure that you keep the output of each cell within `HW2_Report.ipynb`.

Please package the following four files into a single ZIP file for submission. 

```
solutionsA.py
solutionsB.py
HW2_Report.ipynb
tagger.pt.model
```

### Late Day Policy
10% off for each day. After three days, the assignment will be given a score of zero.

