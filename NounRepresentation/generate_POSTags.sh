#!/bin/sh

./POSTagger/ark-tweet-nlp-0.3.2/runTagger.sh --model POSTagger/ark-tweet-nlp-0.3.2/model.irc.20121211 --decoder greedy $1 > $2
