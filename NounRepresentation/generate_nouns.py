import subprocess
import os
import argparse
import numpy
import cPickle
import copy

# Define stop-words
stopwords = "all another any anybody anyone anything both each each other either everybody everyone everything few he her hers herself him himself his I it its itself many me mine more most much myself neither no one nobody none nothing one one another other others ours ourselves several she some somebody someone something that their theirs them themselves these they this those us we what whatever which whichever who whoever whom whomever whose you your yours yourself yourselves . , ? ' - -- !"

stopwords = stopwords.lower().split()

# Load commands
all_commands = open('commands_all.txt', "r").readlines()
if len(all_commands):
    all_commands = [x.strip() for x in all_commands]

conflicting_commands = open('commands_conflicts.txt', "r").readlines()
if len(conflicting_commands):
    conflicting_commands = [x.strip() for x in conflicting_commands]

nonconflicting_commands = list(set(all_commands) - set(conflicting_commands))

# Helper function to find unique elements in list while preserving their ordering
def unique_list_elements(l):
  output = []
  for x in l:
    if x not in output:
      output.append(x)
  return output

# Helper function to flatten a list
def flatten_list(l):
    return [item for sublist in l for item in sublist]

# Helper function which returns <url> is word is a url
def word_or_url_or_path(word):
    if len(word) > 0:
        if ('http' in word) or ('www.' in word) or ('.com'  in word) or ('.net' in word) or ('.org' in word) or ('.edu' in word):
            if '.' in word:
                return '<url>'
        elif len(word) > 1:
            if word[0] == '/' and '/' in word[1:]:
                return '<path>'

    return word

# Helper function to shorten nouns
def shorten_noun(noun):
    return noun.replace('-', '').replace('\'', '').replace('+', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('_', '').replace('/', '').replace('\\','')


# Helper function to determine tense of a sentence given the POS tags for the sentence
def determine_tense_input(text_POSTags):
    future_tenses = len([word for word in text_POSTags if word == "MD"])
    present_tenses = len([word for word in text_POSTags if word in ["VBP", "VBZ","VBG"]])
    past_tenses = len([word for word in text_POSTags if word in ["VBD", "VBN"]])

    ret = ''
    if past_tenses > 0:
        ret += 'past_'
    if present_tenses > 0:
        ret += 'present_'
    if future_tenses > 0:
        ret += 'future_'

    if past_tenses+present_tenses+future_tenses == 0:
        ret += 'no_'

    ret += 'tenses'
    return ret

# Constants used in Ubuntu
command_placeholder = 'something'
end_of_turn = '__eot__'
end_of_utterance = '__eou__'


# Given a turn and, optionally, its POS tags outputs the time tenses and nouns of the turn
def process_full_turn(turn, current_turn_POSTags = None):
    # First, replace all unambigious commands with the command_placeholder token
    words_excluding_commands = copy.deepcopy(turn.replace('**unknown**', '')).split()

    for wordidx, word in enumerate(words_excluding_commands):
        if word.lower() in nonconflicting_commands:
            words_excluding_commands[wordidx] = command_placeholder


    preprocessed_turn = ' '.join(words_excluding_commands)

    found_tenses_all_utterances = []
    found_nouns_all_utterances = []

    # If POS tags are given, find nouns inside each utterance
    if current_turn_POSTags is not None:
        assert preprocessed_turn in current_turn_POSTags.replace('  ', ' ').replace('  ', ' ')
        current_turn_POSTags_list = current_turn_POSTags.split('\t')[1].split(' ')

        retokenized_turn = current_turn_POSTags.split('\t')[0].replace('  ', ' ').replace('  ', ' ').split(' ')

        if len(retokenized_turn[-1]) == 0:
            del retokenized_turn[-1]
        if len(retokenized_turn[0]) == 0:
            del retokenized_turn[0]

        if len(retokenized_turn) != len(current_turn_POSTags_list):
            assert False

        retokenized_utterances = (' '.join(retokenized_turn)).split(end_of_utterance)

        if retokenized_utterances:
            if retokenized_utterances[0] == '':
                del retokenized_utterances[0]
        if retokenized_utterances:
            if retokenized_utterances[-1] == '':
                del retokenized_utterances[-1]
        if retokenized_utterances:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]
        if retokenized_utterances:
            if retokenized_utterances[-1] == f' {end_of_turn}':
                del retokenized_utterances[-1]
        if retokenized_utterances:
            if retokenized_utterances[-1] == '':
                del retokenized_utterances[-1]
        if retokenized_utterances:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]
        if retokenized_utterances:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]
        if retokenized_utterances:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]

        curr_postag_idx = 0

        for retokenized_utterance in retokenized_utterances:
            retokenized_utterance_list = retokenized_utterance.split(' ')

            if len(retokenized_utterance_list[-1]) == 0:
                del retokenized_utterance_list[-1]
            if len(retokenized_utterance_list) == 0:
                assert False
            if len(retokenized_utterance_list[0]) == 0:
                del retokenized_utterance_list[0]

            retokenized_utterance_list += [end_of_utterance]

            if len(retokenized_utterance_list) <= 1:
                found_tenses_all_utterances.append('no_tenses')
                found_nouns_all_utterances.append(['no_nouns'])
                continue

            assert len(retokenized_utterance_list) > 1

            utterance_postags_list = [
                current_turn_POSTags_list[i]
                for i in range(
                    curr_postag_idx,
                    curr_postag_idx + len(retokenized_utterance_list),
                )
            ]
            curr_postag_idx += len(retokenized_utterance_list)

            # Compute time tense
            tenses_token = determine_tense_input(utterance_postags_list)
            found_tenses_all_utterances.append(tenses_token)

            # Find nouns
            utterance_nouns_list = []

            for POSTag_idx, POSTag in enumerate(utterance_postags_list):
                potential_noun = retokenized_utterance_list[POSTag_idx].lower()
                potential_url_path = word_or_url_or_path(potential_noun)
                # Add to noun list if it's a path or url
                if potential_noun != potential_url_path:
                    utterance_nouns_list.append(potential_url_path)
                elif len(POSTag) > 1: # Add to noun list if it has the noun POS tag
                    if POSTag[:2] == 'NN':
                        if potential_noun != command_placeholder:
                            if potential_noun != end_of_turn:
                                if potential_noun != end_of_utterance:
                                    if potential_noun not in stopwords:
                                        utterance_nouns_list.append(shorten_noun(potential_noun))

            if not utterance_nouns_list:
                utterance_nouns_list = ['no_nouns']

            found_nouns_all_utterances.append(utterance_nouns_list)

    utterance_count = len(preprocessed_turn.split(end_of_utterance))-1
    while len(found_tenses_all_utterances) < utterance_count:
        found_tenses_all_utterances.append('no_tenses')
        found_nouns_all_utterances.append(['no_nouns'])

    assert len(found_tenses_all_utterances) == len(found_nouns_all_utterances)

    return preprocessed_turn, found_tenses_all_utterances, found_nouns_all_utterances



# Preprocesses the dialogues and computes POS tags using POS tagger
def process_dialogues(dialogue_input_file, dialogue_output_file, postags_output_file):
    print 'Generating POS tags for: ', dialogue_input_file

    text = open(dialogue_input_file, 'r').readlines()
    dialogues_counted=len(text)

    processed_dialogues = ''

    for i in range(0,dialogues_counted):
        if i % 5000 == 0:
            print '     i: ' + str(i) + ' / ' + str(dialogues_counted)

        current_dialogue = text[i].strip()
        if current_dialogue.split()[-1] == end_of_turn:
            current_dialogue = ' '.join(current_dialogue.split()[0:-1])

        turns = current_dialogue.split(end_of_turn)

        assert len(turns) > 0 # Make sure dialogue is not empty!

        processed_dialogue = ''

        for turn in turns:
            # Process turn  to get time tenses and nouns
            processed_turn , _, _ = process_full_turn(turn )

            processed_dialogue += processed_turn  + ' ' + end_of_turn


        processed_dialogues += processed_dialogue.replace('  ', ' ') + '\n'

    # Fix up dialogues for the POS Tagger
    processed_dialogues = processed_dialogues.replace(end_of_utterance, ' ' + end_of_utterance + ' ')
    processed_dialogues = processed_dialogues.replace(end_of_turn, ' ' + end_of_turn + ' ')
    processed_dialogues = processed_dialogues.replace(end_of_utterance, ' $$$###$$$ ')
    processed_dialogues = processed_dialogues.replace(end_of_turn, ' $$$### \n ')
    processed_dialogues = processed_dialogues.replace('**unknown**', ' ' + command_placeholder + ' ')

    processed_dialogues = processed_dialogues.replace('  ', ' ')
    processed_dialogues = processed_dialogues.replace('  ', ' ')
    processed_dialogues = processed_dialogues.replace('  ', ' ')
    processed_dialogues = processed_dialogues.replace('  ', ' ')
    processed_dialogues = processed_dialogues.replace('  ', ' ')
    processed_dialogues = processed_dialogues.replace('  ', ' ')
    processed_dialogues = processed_dialogues.replace(' \n', '\n')
    processed_dialogues = processed_dialogues.replace('\n ', '\n')

    out_file = open(dialogue_output_file, 'w')
    out_file.write(processed_dialogues)
    out_file.close()

    # Run POS Tagger: ark-tweet-nlp-0.3.2
    print 'Running POS Tagger: ark-tweet-nlp-0.3.2...'
    args = ['sh', 'generate_POSTags.sh', dialogue_output_file, postags_output_file]
    subprocess.call(args) 


# Convert the dialogues into dialogue noun representations
def process_nouns(dialogue_input_file, pos_input_file, dialogue_output_file):
    print 'Processing dialogue noun representations for: ', dialogue_input_file

    turns_counted = 0
    words_counted = 0

    turns_with_nouns = 0
    nouns_counted = 0

    text = open(dialogue_input_file, 'r').readlines()
    dialogues_counted=len(text)

    POSTags = open(pos_input_file, 'r').readlines()
    POSTags_line_index = 0

    if len(POSTags[-1]) == 0:
        del POSTags[-1]

    processed_dialogue_nouns = ''

    for i in range(0,dialogues_counted):
        if i % 5000 == 0:
            print '     i: ' + str(i) + ' / ' + str(dialogues_counted)

        processed_dialogue_noun = ''

        current_dialogue = text[i].strip()
        if current_dialogue.split()[-1] == end_of_turn:
            current_dialogue = ' '.join(current_dialogue.split()[0:-1])

        turns = current_dialogue.split(end_of_turn)

        assert len(turns) > 0 # Make sure dialogue is not empty!

        add_end_of_turn_to_noun_representation = False
        if len(current_dialogue) > len(end_of_turn):
            if current_dialogue[len(current_dialogue)-len(end_of_turn):len(current_dialogue)] == end_of_turn:
                add_end_of_turn_to_noun_representation = True

        for turn in turns:
            turns_counted += 1


            # Extract appropriate POS tags for turn
            while len(POSTags[POSTags_line_index].strip().replace(' ', '').replace('\t', '')) == 0:
                POSTags_line_index += 1
                
            current_turn_POSTags = POSTags[POSTags_line_index].strip()
            current_turn_POSTags = current_turn_POSTags.replace('$$$###$$$', ' ' + end_of_utterance + ' ')
            current_turn_POSTags = current_turn_POSTags.replace('$$$###', ' ' + end_of_turn + ' ')

            POSTags_line_index += 1

            # Process turn to get time tenses and nouns
            processed_turn, found_tenses_all_utterances, found_nouns_all_utterances = process_full_turn(turn, current_turn_POSTags)

            # Generate noun representation given time tenses and nouns
            processed_noun = ''
            for utt_idx in range(len(found_tenses_all_utterances)):
                processed_noun += found_tenses_all_utterances[utt_idx] + ' ' + ' '.join(unique_list_elements(found_nouns_all_utterances[utt_idx])) + ' ' + end_of_utterance + ' '

            processed_noun = processed_noun.replace('  ', ' ')

            # Verify that generated noun representation is valid
            assert len(turn.split(end_of_turn)) == len(processed_noun.split(end_of_turn))

            assert len(turn.split(end_of_utterance)) == len(processed_noun.split(end_of_utterance))

            processed_noun += end_of_turn + ' '

            processed_dialogue_noun += processed_noun

            # Compute statistics for words
            words_counted += len(turn.split())

            # Compute statistics for nouns
            if len(set(flatten_list(found_nouns_all_utterances))) > 0:
                turns_with_nouns += 1
                nouns_counted += len(set(flatten_list(found_nouns_all_utterances)))



        # If necessary, delete last __eot__ token
        if not add_end_of_turn_to_noun_representation:
            if len(processed_dialogue_noun) > len(end_of_turn):
                processed_dialogue_noun = processed_dialogue_noun[0:len(processed_dialogue_noun)-(len(end_of_turn)+2)]

        processed_dialogue_nouns += processed_dialogue_noun.replace('  ', ' ') + '\n'

    print 'Statistics:'
    print '     nouns_counted / words_counted', float(nouns_counted)/float(words_counted)
    print '     turns_with_nouns / turns_counted', float(turns_with_nouns)/float(turns_counted)

    processed_dialogue_nouns = processed_dialogue_nouns.replace('  ', ' ')
    processed_dialogue_nouns = processed_dialogue_nouns.replace('  ', ' ')
    processed_dialogue_nouns = processed_dialogue_nouns.replace('  ', ' ')
    processed_dialogue_nouns = processed_dialogue_nouns.replace(' \n', '\n')
    processed_dialogue_nouns = processed_dialogue_nouns.replace('\n ', '\n')

    out_file = open(dialogue_output_file, 'w')
    out_file.write(processed_dialogue_nouns)
    out_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dialogues", type=str, default="", help="Input dialogues")
    parser.add_argument("preprocessed_dialogues", type=str, default="", help="Intermediate file used for saving preprocessed dialogues")
    parser.add_argument("POSTags", type=str, default="", help="Intermediate file for saving POS tags")

    parser.add_argument("output_nouns", type=str, default="", help="Output file containing dialogue noun representations")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    process_dialogues(args.input_dialogues, args.preprocessed_dialogues, args.POSTags)
    process_nouns(args.input_dialogues, args.POSTags, args.output_nouns)
