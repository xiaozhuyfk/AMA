#!/usr/bin/env bash

cd /home/hongyul/AMA; source ../.bashrc; python -m query_processor.rank_learner test webquestionstest
#cd /home/hongyul/AMA; source ../.bashrc; python -m query_processor.rank_learner model webquestionstrain
#cd /home/hongyul/AMA; source ../.bashrc; python -m query_processor.rank_learner train webquestionstrain
#cd /home/hongyul/AMA; source ../.bashrc; python -m query_processor.rank_learner model webquestionstrain $1
