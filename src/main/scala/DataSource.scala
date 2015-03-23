package org.template.word2vec

import grizzled.slf4j.Logger
import io.prediction.controller._
import io.prediction.data.storage.Storage
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()

    val tweets = eventsDb
      // Find all tweets in the database.
      .find(
        appId = dsp.appId,
        entityType = Some("source"),
        eventNames = Some(List("tweet"))
      )(sc)

      // Retrieve just the text.
      .map(e => Tweet(
        e.properties.get[String]("text")
      ))

    new TrainingData(tweets)
  }
}

case class Tweet(
  text: String
) extends Serializable

class TrainingData(
  val tweets: RDD[Tweet]
) extends Serializable