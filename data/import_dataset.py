"""
Import sample data for word2vec engine
"""

import predictionio
import argparse
import csv

def import_events(client, file):
  f = open(file, 'r')
  reader = csv.DictReader(f, delimiter=",", quotechar='"')
  count = 0

  print "Importing data..."
  for row in reader:
      client.acreate_event(
          event = "tweet",
          entity_type = "source",
          entity_id = row["SentimentSource"],
          properties = {
              "sentiment": row["Sentiment"],
              "text": row["SentimentText"]
          }
      )
      count+=1
  f.close()
  print "%s events are imported." % count

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Import sample data for recommendation engine")
  parser.add_argument('--access_key', default='invald_access_key')
  parser.add_argument('--url', default="http://localhost:7070")
  parser.add_argument('--file', default="./data/dataset.csv")

  args = parser.parse_args()
  print args

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file)
