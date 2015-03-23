package org.template.word2vec

import grizzled.slf4j.Logger
import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    // We use just a part of the data set for development.
    val sentences = trainingData.tweets.map(_.text).sample(false, 0.001)

    // Cache the sentences to prevent querying the database multiple times.
    sentences.cache()

    logger.info(s"Imported ${sentences.count()} sentences.")
    PreparedData(sentences)
  }
}

case class PreparedData(
  sentences: RDD[String]
) extends Serializable
