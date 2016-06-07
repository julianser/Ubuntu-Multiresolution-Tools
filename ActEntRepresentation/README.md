### Description

This directory contains the files for computing the activity-entity metrics as well as recreating the activity-entity representations for the Ubuntu Dialogue Corpus described by Serban et al. (2016).



### Python Dependencies

    gensim (version 0.11 or newer)
    cPickle (version 1.71 or newer)
    CMU ARK Twitter Part-of-Speech Tagger v0.3.2

### Model Evaluation

Evaluation is straightforward using the eval_file bash script:

    eval_file.sh <model_responses> <output_file>

&lt;model_responses&gt; is a file consisting of model responses (one response per line).
&lt;output_file&gt; is the output file containing the metric results with 95% confidence intervals.



### Dataset Generation

The activity-entity representations can be created by running:

    python generate_nouns.py <dialogue_corpus> <tmp_a> <tmp_b> <coarse_representations>

where &lt;dialogue_corpus&gt; is a file of consisting of dialogues (one dialogue per line), where turns are separated by the token "__eot__" and utterances are separated by the token "__eou__".
&lt;tmp_a&gt; and &lt;tmp_b&gt; are temporary intermediate files that the script needs to store. These can safely be deleted afterwards.
&lt;coarse_representations&gt; is the output file containing of the coarse activity-entity representations.


If this is done on a new dataset, make sure that the "__eou__" and "__eot__" token counts match exactly:

    grep -roh '__eou__' <original_corpus> | wc -l
    grep -roh '__eou__' <actent_corpus> | wc -l
    grep -roh '__eot__' <original_corpus> | wc -l
    grep -roh '__eot__' <actent_corpus> | wc -l



### Files:

    activities_present_form.txt: Activities in present and present participle form (tab-separated file).
    activities_past_form.txt: Activities in past and past participle form (tab-separated file).
    append_end_of_utterance_token.py: Helper script required by "eval_file.sh".
    commands_all: Contains set of all general linux commands.
    commands_conflicts: Contains set of general linux commands which have
                        names conflicting with common use verbs.
    GroundTruth/actent_test_responses.txt: Ground truth activity-entity representations.
    eval_file.sh: Evaluation script (see instructions above).
    evaluate_actents.py: Evaluation script required by "eval_file.sh".
    generate_actents.py: Generation script (see instructions above).
    generate_POSTags.sh: Helper script used to call the POS tagger.
    ubuntu_vectors.bin: Ubuntu word embeddings required by "generate_actents.py".
    unigram2entity_dict.pkl: Python pickled dictionary containing the entities.



### References

Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation. Iulian Vlad Serban, Tim Klinger, Gerald Tesauro, Kartik Talamadupula, Bowen Zhou, Yoshua Bengio, Aaron Courville. 2016. http://arxiv.org/abs/1606.00776

The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems.  Ryan Lowe, Nissan Pow, Iulian Serban, Joelle Pineau. 2015. SIGDIAL. http://arxiv.org/abs/1506.08909.
