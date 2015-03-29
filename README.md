Overview
========

This template demonstrates how to integrate the Word2Vec implementation from deeplearning4j with PredictionIO.

The Word2Vec algorithm takes a corpus of text and computes a vector representation for each word. These representations can be subsequently used in many natural language processing applications and for further research.

Creating the project
====================

To copy the template run the following command:

```bash
> pio template get pawel-n/template-scala-parallel-word2vec <YourEngineDir>
> cd <YourEngineDir>
```

Now create a new app:

```bash
> pio app new <AppName>
[INFO] [App$] Initialized Event Store for this app ID: 2.
[INFO] [App$] Created new app:
[INFO] [App$]       Name: <AppName>
[INFO] [App$]         ID: <AppId>
[INFO] [App$] Access Key: 
<AccessKey>
```

Make sure engine.js matches your app id:
```json
...
    "appId": <AppId>
...
```

Importing the data
==================

The example data set is a list of tweets with their sentiments. We will be using just the text. First download the file:

```bash
> wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
> unzip Sentiment-Analysis-Dataset.zip
> mv Sentiment\ Analysis\ Dataset.csv data/dataset.csv
> rm Sentiment-Analysis-Dataset.zip
```

Now run the "data/import_dataset.py" script to import events:
```bash
> ./data/import_dataset.py --access_key=<AccessKey>
```

This can take a while. Feel free to make yourself a tea.

Build, train, deploy
====================

If your engine is running out of memory try to increase the limit with "--executor-memory" and "--driver-memory" options:

```bash
pio build
pio train -- --executor-memory=10GB --driver-memory=10GB
pio deploy -- --executor-memory=10GB --driver-memory=10GB
```

Querying
========

Once the engine is deployed you can query it with the "data/send_query.py" script:

```bash
> ./data/send_query.py
```

The script will ask you for a word and give you a list of similar words. The distance between two words is computed as the cosine between their vector representations.

build.sbt
=========

Due to large number of conflicts we use a custom merge strategy. It is also necessary to exclude a few dependencies of deeplearning4j-nlp.

Algorithm
=========

In this section we describe briefly the algorithm of this engine. The rest of DASE components are trivial.

As the first step we define an input preprocessor. InputHomogenization normalizes the input sentences by removing punctuation and converting words to lower case.

```scala
object PreProcessor extends SentencePreProcessor {
  override def preProcess(s: String): String =
    new InputHomogenization(s).transform()
}
```

```scala
override def train(sc: SparkContext, data: PreparedData): Model = {
    val sentences = data.sentences.collect.toSeq.asJavaCollection
    val sentenceIterator = new CollectionSentenceIterator(PreProcessor, sentences)
```

After we have normalized the sentences, the next step is to split each sentence into a list of words. Apache UIMA provides a tokenizer that will take care of this.

```scala
    val tokenizerFactory = new UimaTokenizerFactory()
```

We create a new Word2Vec object with our parameters: 

```scala
    val word2vec = new Word2Vec.Builder()
        .windowSize(params.windowSize)
        .layerSize(params.layerSize)
        .iterate(sentenceIterator)
        .tokenizerFactory(tokenizerFactory)
        .build()
```

Finally we can train it and save as our model:

```scala
    word2vec.fit()
    new Model(word2vec)
  }
```

The predict method simply calls a Word2Vec method to find the most similar words:

```scala
override def predict(model: Model, query: Query): PredictedResult = {
    val nearest = model.word2vec.wordsNearest(query.word, query.num)
    PredictedResult(nearest.asScala.toArray)
  }
```
