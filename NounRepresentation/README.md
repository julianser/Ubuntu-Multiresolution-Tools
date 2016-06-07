### Description

This directory contains the files for regenerating the noun representations for the Ubuntu Dialogue Corpus described by Serban et al. (2016).

The noun representations can be created by running:

    python generate_nouns.py <dialogue_corpus> <tmp_a> <tmp_b> <coarse_representations>

where &lt;dialogue_corpus&gt; is a file of consisting of dialogues (one dialogue per line), where turns are separated by the token "__eot__" and utterances are separated by the token "__eou__".
&lt;tmp_a&gt; and &lt;tmp_b&gt; are temporary intermediate files that the script needs to store. These can safely be deleted afterwards.
&lt;coarse_representations&gt; is the output file containing of the coarse noun representations.



### Files:

    commands_all: Contains set of all general linux commands
    commands_conflicts: Contains set of general linux commands which have
                        names conflicting with common use verbs.
    generate_nouns.py: Script which generates noun abstractions from dialogues (see instructions above).
    generate_POSTags.sh: Helper script used to call the POS tagger.



### References

Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation. Iulian Vlad Serban, Tim Klinger, Gerald Tesauro, Kartik Talamadupula, Bowen Zhou, Yoshua Bengio, Aaron Courville. 2016. http://arxiv.org/abs/1606.00776.

The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems.  Ryan Lowe, Nissan Pow, Iulian Serban, Joelle Pineau. 2015. SIGDIAL. http://arxiv.org/abs/1506.08909.
