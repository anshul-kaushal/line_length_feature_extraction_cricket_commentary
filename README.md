# Line and Length Feature Extractor: Cricket Commentary 

## Motivation

While browsing through the depths of LinkedIn one day, I came across an individual who was as, if not more, passionate about data and cricket as me, looking for collaborators to join his <b>Co-learning lounge</b>, to build an ambitious dashboard project with IPL, a popular Cricket league, as focus. As someone who is constantly looking for opportunities to apply the acquired skillset to solve problems, I decided to reach out to him and his team. After carefully reviewing my academic credentials and past experience, it was decided that I will work on a Natural Language Processing problem in order to extract value from text ball-ball cricket commentary and help expand their dataset and as a consequence, enhance the depth of their analysis further.

## Understanding the Problem 

### Cricket

Before we move on with examining the problem, it is important to understand what Cricket is all about. It is a popular commonwealth bat-ball sport that is played between two teams of eleven players on a field at the centre of which is a 22-yard pitch with a wicket at each end, each comprising two bails balanced on three stumps. Both the teams get an opportunity to bat while the other team bowls and whichever team scores more runs wins the match. Learn more about cricket [here](https://en.wikipedia.org/wiki/Cricket).

### Ball-ball text commentary

Various Cricket news brands like [ESPNCricInfo](https://www.espncricinfo.com/) and [Cricbuzz](https://www.cricbuzz.com/) offer live updates of a cricket match on ball-ball basis in text format. Here's an example of the same.

<img width="500" alt="Screenshot 2023-02-15 at 5 36 49 PM" src="https://user-images.githubusercontent.com/79828435/220005238-4398f280-3677-4fe3-a775-6ab353ace687.png">

The text commentary provided by ESPNCricInfo here is filled with valuable information that isn't readily accessible or useful without further processing. The purpose of the project, thus, was to extract useful features from this text data that could be added to the dataset that the analysts at Co-learning lounge were working on. The features that I decided to work on were the line and length of the ball delivered.

### Line and Length of a delivery 
The line of a delivery (ball thrown by bowler) refers to the direction in which the bowler has bowled the ball. The length of a delivery refers to how close the ball is to the batsman when it makes contact with the pitch. 

<img width="500" alt="Screenshot 2023-02-19 at 10 51 34 PM" src="https://user-images.githubusercontent.com/79828435/220005501-2ba47cb2-a03c-4b69-a2ec-159d38dd6599.png">      <img width="500" alt="Screenshot 2023-02-19 at 10 52 54 PM" src="https://user-images.githubusercontent.com/79828435/220005663-32e31f68-2453-4b90-ba7b-c137ee0403ce.png">

## Data

Access the dataset [here](https://www.kaggle.com/datasets/mhemendra/cricinfo-ipl-commentary)

The dataset contains the ball by ball commentary details of IPL 2018- 2020 scrapped from ESPNCricinfo. For the scope of this project, we're only concerened with the `long_text` column, that has 14447 entries. It's also important to note that we only have the text commentary without any labelled line and length features. Hence, <b><i>we're dealing here with a lot of unlabelled data but no labelled data whatsoever.</i></b>

## Project 

### Brief overview of strategy 

Given the complexity of the task, deep learning seemed like the right strategy to achieve substantial results. Named Entity Recognition, the process of identifying and categorizing key information in text data, seemed like the right approach to extract the line and length features from commentary. BERT is especially a great model for facilitating NER tasks, hence it was used for this project. Given that NER is a supervised task, labelled data was required to fine-tune the BERT model for this task. However, with abundance of unlabeled data, a methodology that could utilise this data while maximizing its learnings on a limited set of labeled data had to be adopted to utilise time and resources efficiently. 

BERT has two unique training approaches, namely the Masked language model and the Next sentence prediction task. Masked language model is particularly interesting since it can be adopted for self-supervised language pre-training. For masked language modelling, BERT based model takes a sentence as input and masks 15% of its words and by running it with masked words through the model, it predicts the masked words. It also learns the bidirectional representation of the sentences to make the prediction more precise. Thus, the devised strategy involved using this approach for fine-tuning the BERT model on the unlabeled data to better understand the context and the structure of these comments. Further, the representations that would be learned from this MLM task were to be used to fine-tune the model for the token classification task on a small amount of self-annotated data. 

### Data preprocessing

The following was achieved with data preprocessing:

- Extracted the `long_text` column to acquire a list of 14447 comments. 
- For the first 100 (training-set) and the last 30 comments (test-set), removed punctuation marks from and tokenized the comments.
- Wrote the tokenized comments to two separate csv files (train file and test file) along with the index and an empty column: `tag`.
- Saved the rest of the comments (14447 - 130) as unsupervised comments in a pickle file.

### Annotation 

Annotation was based on a sequence labelling technique, namely the IOB tagging, that is commonly used for NER tasks. IOB is a format for chunking that can denote the inside, outside, and beginning of a chunk. Here’s an example to understand it better:

<img width="800" alt="Screenshot 2023-02-19 at 10 57 09 PM" src="https://user-images.githubusercontent.com/79828435/220006122-07623926-a372-4803-a795-eef81fbe8bc1.png">


The comments were annotated in the two csv files (train and test) obtained from the data processing stage by manually filling in the tag column. Here’s an illustration of the same:

<img width="215" alt="Screenshot 2023-02-19 at 10 59 51 PM" src="https://user-images.githubusercontent.com/79828435/220006367-399d866c-076c-4d6d-aa20-6abc10651635.png">


### Training

Masked language modelling objective:

- Loaded the `unsupervised_comments` pickle file.
- Loaded the pretained `BERTForMaskedLM` model and `BertTokenizer`tokenizer from the `HuggingFace` library.
- Split the unsupervised comments (14317) into training (80%) and validation set (20%), after dropping the null columns.
- Prepared the dataset for ML training using the `datasets` library, directly loading the dataset from Pandas Dataframe.
- Tokenized the comments in the dataset using the aforementioned tokenizer.
- Used the `DataCollatorForLanguageModelling` data collator to form training batches.
- Trained the model for `5 epochs`.
- Saved the tokenizer vocabulary, learnt weight parameters and the config file into a directory.

Token Classification:

- Loaded the labelled comments, self-annotated, from the `ipl_comments.csv` file.
- Loaded the model and tokenizer previously trained on the unlabelled data, into `BertTokenizerFast` tokenizer and `BertForTokenClassification` model.
- Split the labelled data into training (80%) and validation set (20%).
- Prepared the tags, the tokenized (from <b>Data preprocessing</b>) sentences and wrote methods to predict and evaluate the model output.
- Trained the model for `10 epochs`.

### Evaluation

Masked language modelling objective:

The criterion used for model evaluation was perplexity. It is a way to quantify the uncertainty a langauge model has in predicting text. [Here's](https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584#:~:text=In%20the%20context%20of%20Natural,goodness%20of%20already%20written%20sentences.) an excellent yet succint article on the same.

The perplexity achieved for this task was `4.08`, which is a good score. This roughly translates to the model being confused between only 4.08 words while deciding the next word in a sequence.

Token Classification Objective:

The criterion used for model evaluation in this case was Precision, Recall and F-score. 

The precision achieved on the validation datset was 98.78%, F-score was 93.10% and Recall was 88.04%. The same for the test set equalled 93.8%, 79.8% and 86.2% respectively. 

