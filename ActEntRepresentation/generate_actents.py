import subprocess
import os
import argparse
import numpy
import cPickle
import copy


# Helper function to find all words with one character difference
alphabet = 'abcdefghijklmnopqrstuvwxyz'
def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

# Load in Word2Vec embeddings trained on Ubuntu Dialogue Corpus
from gensim.models import Word2Vec
w2v = Word2Vec.load_word2vec_format('ubuntu_vectors.bin', binary=True)

# Load word -> entity dictionaries
unigram2entity = cPickle.load(open('unigram2entity_dict.pkl', 'r'))

# Load activity types (and their synonyms)
activities_past2present = {}
activities_present_dict = {}
activities_past_dict = {}

activities_present_list = []

for line in open('activities_present_form.txt', 'r'):
   words = line.strip().lower().replace('-', '').split()
   first_word = words[0]
   activities_present_list.append(first_word)
   for word in words:
       activities_present_dict[word] = first_word

for line_idx, line in enumerate(open('activities_past_form.txt', 'r').readlines()):
    words = line.strip().lower().replace('-', '').split()
    if len(words) > 0:
        first_word = words[0]
        for word in words:
            activities_past_dict[word] = first_word
            activities_past2present[word] = activities_present_list[line_idx]


# Find misspelled activity verbs using Word2Vec
# We define a misspelled activity word as a word which:
# 1) is one character away from the original (correct) activity word,
# 2) is among the top 10 closest words to the original (correct) activity word w.r.t. Word2Vec word embedding cosine similarity,
# 2) has a Word2Vec word embedding cosine similarity above 0.5 to the original (correct) activity word,
# 3) is contained among the 20K most frequent words.
#
# This procedure has very high precision.
for activity_key in activities_present_dict:
   if activity_key in w2v:
      similar_words = w2v.most_similar(activity_key)
      for similar_word in similar_words:
         if (similar_word[0] not in activities_present_dict
             and similar_word[0] not in activities_past_dict):
            if similar_word[0] in edits1(activity_key):
                if w2v.similarity(similar_word[0], activity_key) > 0.5:
                    activities_present_dict[similar_word[0]] = activities_present_dict[activity_key]


for activity_key in activities_past_dict:
   if activity_key in w2v:
      similar_words = w2v.most_similar(activity_key)
      for similar_word in similar_words:
         if (similar_word[0] not in activities_present_dict
             and similar_word[0] not in activities_past_dict):
            if similar_word[0] in edits1(activity_key):
                if w2v.similarity(similar_word[0], activity_key) > 0.5:
                    activities_past_dict[similar_word[0]] = activities_past_dict[activity_key]

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

# Helper function to map word -> entity
def word_to_entity(word):
   if len(word) > 0:
      if word in unigram2entity:
          return unigram2entity[word]
      if word.title() in unigram2entity:
          return unigram2entity[word.title()]
      if word.lower() in unigram2entity:
          return unigram2entity[word.lower()]
      if word.upper() in unigram2entity:
          return unigram2entity[word.upper()]
      if word.replace('.', '').lower() in unigram2entity:
          return unigram2entity[word.replace('.', '').lower()]
      if ('http' in word) or ('www.' in word) or ('.com'  in word) or ('.net' in word) or ('.org' in word) or ('.edu' in word):
          if '.' in word:
              return '<url>'

      if len(word) > 1:
         if word[0] == '/' and '/' in word[1:]:
            return '<path>'

      if '-' in word:
         first_subword = word.split('-')[0]
         if len(first_subword) >= 4:
             return word_to_entity(first_subword)
      elif word[0] != '\\' and word[-1] != '\\':
         if word[0] != '/' and word[-1] != '/':
            if '/' in word:
                first_subword = word.split('/')[0]
                if len(first_subword) >= 4:
                    return word_to_entity(first_subword)
            elif '\\' in word:
                first_subword = word.split('\\')[0]
                if len(first_subword) >= 4:
                    return word_to_entity(first_subword)

      if len(word) > 3:
         if word[-1].isdigit():
            return word_to_entity(word[:-1])


   return None

# Helper function to determine tense of an utterance given its POS tags
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

# Constants
entity_placeholder = '__ent__'
command_placeholder = '__cmd__'

end_of_turn = '__eot__'
end_of_utterance = '__eou__'


assert word_to_entity(entity_placeholder) is None
assert word_to_entity(command_placeholder) is None



# Given a turn (and optionally, its POS tags) output the time tenses, activities, entities and commands related to the turn
def process_full_turn(turn, current_turn_POSTags = None):
    # First find all entities
    words = turn.split()
    words_with_replaced_entities = copy.deepcopy(turn).split()


    found_entities = []
    found_entities_all_utterances = []
    found_entities_utterance = []
    n_words = len(words)
    for word_idx in range(n_words):
        if words[word_idx] == end_of_utterance:
            found_entities_all_utterances.append(copy.deepcopy(found_entities_utterance))
            found_entities_utterance = []

        if not words[word_idx] == end_of_turn and not words[word_idx] == end_of_utterance:
            if word_idx < n_words-1:
                if not words[word_idx+1] == end_of_turn and not words[word_idx+1] == end_of_utterance:
                    compound_word = words[word_idx] + '_' + words[word_idx+1]
                    r = word_to_entity(compound_word)
                    if r:
                        found_entities.append(r)
                        found_entities_utterance.append(r + '_entity')
                        words_with_replaced_entities[word_idx] = entity_placeholder
                        words_with_replaced_entities[word_idx+1] = entity_placeholder
                        continue
            
            r = word_to_entity(words[word_idx])
            if r:
                found_entities.append(r)
                found_entities_utterance.append(r + '_entity')
                words_with_replaced_entities[word_idx] = entity_placeholder
                continue

    # Remove repetition of entity_placeholders
    word_idx = 0
    while word_idx < len(words_with_replaced_entities)-1:
        word_idx += 1
        if words_with_replaced_entities[word_idx] == entity_placeholder:
            if words_with_replaced_entities[word_idx-1] == entity_placeholder:
                del words_with_replaced_entities[word_idx]
                word_idx -= 1

    # Find set of all commands in turn
    word_is_unambiguous_command = [False]*len(words_with_replaced_entities)
    for word_idx, word in enumerate(words_with_replaced_entities):
        if word in nonconflicting_commands:
            word_is_unambiguous_command[word_idx] = True

    command_was_added = True
    while command_was_added:
        command_was_added = False
        for word_idx, word in enumerate(words_with_replaced_entities):
            if not (word_is_unambiguous_command[word_idx]) and (not word == end_of_utterance) and (not word == '!') and not (word == ','):
                if word != entity_placeholder:
                    neighbour_commands = 0
                    max_neighbour_commands = 0
                    if word_idx > 0:
                        max_neighbour_commands += 1
                        if word_is_unambiguous_command[word_idx-1]:
                            neighbour_commands += 1
                    if word_idx < len(words_with_replaced_entities) - 1:
                        max_neighbour_commands += 1
                        if word_is_unambiguous_command[word_idx+1]:
                            neighbour_commands += 1

                    # Word is surrounded by unambiguous command
                    if neighbour_commands == max_neighbour_commands:
                        word_is_unambiguous_command[word_idx] = True
                        command_was_added = True

                    # Word is neighbour to one command
                    elif neighbour_commands > 0:
                        if word in all_commands:
                            word_is_unambiguous_command[word_idx] = True
                            command_was_added = True
                        elif '-' in word:
                            word_is_unambiguous_command[word_idx] = True
                            command_was_added = True

    # Go through every pair of quotation marks.
    # If one token inside is a command, then all words are assumed to be commands.
    quotation_start = -1
    quotation_end = -1
    quotation_is_command = False
    for word_idx, word in enumerate(words_with_replaced_entities):
        # Check if word is quotation mark
        if word == '\'' or word == '"':
            if quotation_start < 0:
                quotation_start = word_idx
            else:
                quotation_end = word_idx
                if (quotation_end - quotation_start) > 1 and (quotation_end - quotation_start) < 6:
                    if quotation_is_command:
                        for idx in range(quotation_start, quotation_end+1):
                            word_is_unambiguous_command[idx] = True

                quotation_start = -1
                quotation_end = -1
                quotaton_is_command = False

        if word_is_unambiguous_command[word_idx]:
            if quotation_start > 0:
                quotation_is_command = True

        if (word == end_of_utterance) or (word == end_of_turn) or ('!' in word) or ('?' in word) or ('.' in word) or (word == '--') or (word == '-') or (word == ':'):
            quotation_start = -1
            quotation_end = -1
            quotaton_is_command = False

    # Store all commands found
    found_commands = []
    found_commands_all_utterances = [False]
    for word_idx, word in enumerate(words_with_replaced_entities):
        if word == end_of_utterance:
            found_commands_all_utterances.append(False)

        if not word == end_of_turn and not word == end_of_utterance:
            if word_is_unambiguous_command[word_idx]:
                if not word == '"' and not word == '\'' and not word == ']' and not word == '[' and not word == '",' and not word == '\',' and not word == '<' and not word == '>' and not word == '-' and not word == '--' and not word == '->' and not word == ',':
                    found_commands.append(word)
                    found_commands_all_utterances[-1] = True

    del found_commands_all_utterances[-1]

    # Replace all commands found with command placeholder
    words_with_replaced_commands = copy.deepcopy(words_with_replaced_entities)
    for word_idx, word in enumerate(words_with_replaced_commands):
        if not word == end_of_turn and not word == end_of_utterance:
            if word_is_unambiguous_command[word_idx]:
                words_with_replaced_commands[word_idx] = command_placeholder

    # Remove repetition of command_placeholders
    word_idx = 0
    while word_idx < len(words_with_replaced_commands)-1:
        word_idx += 1
        if words_with_replaced_commands[word_idx] == command_placeholder:
            if words_with_replaced_commands[word_idx-1] == command_placeholder:
                del words_with_replaced_commands[word_idx]
                word_idx -= 1

    preprocessed_turn = ' '.join(words_with_replaced_commands)

    found_tenses_all_utterances = []
    found_activities_all_utterances = []

    # If POS tags are given, find activities inside each utterance
    if not current_turn_POSTags == None:

        assert preprocessed_turn.replace(entity_placeholder, 'something').replace(command_placeholder, 'something').replace('**unknown**', 'something').replace('<url>', 'somewhere').replace('<path>', 'somewhere') in current_turn_POSTags.replace('  ', ' ').replace('  ', ' ')
        current_turn_POSTags_list = current_turn_POSTags.split('\t')[1].split(' ')
        
        retokenized_turn = current_turn_POSTags.split('\t')[0].replace('  ', ' ').replace('  ', ' ').split(' ')

        if len(retokenized_turn[-1]) == 0:
            del retokenized_turn[-1]
        if len(retokenized_turn[0]) == 0:
            del retokenized_turn[0]

        if len(retokenized_turn) != len(current_turn_POSTags_list):
            assert False

        retokenized_utterances = (' '.join(retokenized_turn)).split(end_of_utterance)

        if len(retokenized_utterances) > 0:
            if retokenized_utterances[0] == '':
                del retokenized_utterances[0]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == '':
                del retokenized_utterances[-1]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == ' ' + end_of_turn:
                del retokenized_utterances[-1]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == '':
                del retokenized_utterances[-1]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]
        if len(retokenized_utterances) > 0:
            if retokenized_utterances[-1] == ' ':
                del retokenized_utterances[-1]

        curr_postag_idx = 0

        for retokenized_utterance in retokenized_utterances:
            retokenized_utterance_list = retokenized_utterance.split(' ')

            if len(retokenized_utterance_list[-1]) == 0:
                del retokenized_utterance_list[-1]
            if len(retokenized_utterance_list) == 0:
                assert True==False
            if len(retokenized_utterance_list[0]) == 0:
                del retokenized_utterance_list[0]

            retokenized_utterance_list += [end_of_utterance]

            if len(retokenized_utterance_list) <= 1:
                found_tenses_all_utterances.append('no_tenses')
                found_activities_all_utterances.append(['none_activity'])
                continue

            assert len(retokenized_utterance_list) > 1

            utterance_postags_list = []
            for i in range(curr_postag_idx, curr_postag_idx+len(retokenized_utterance_list)):
                utterance_postags_list.append(current_turn_POSTags_list[i])

            curr_postag_idx += len(retokenized_utterance_list)

            # Compute time tense
            tenses_token = determine_tense_input(utterance_postags_list)
            found_tenses_all_utterances.append(tenses_token)

            # Find activities
            utterance_verb_list = []
            utterance_two_verb_list = []
            utterance_potential_verb_list = []

            for POSTag_idx, POSTag in enumerate(utterance_postags_list):
                if len(POSTag) > 1:
                    if POSTag[0:2] == 'VB':
                        utterance_verb_list.append(retokenized_utterance_list[POSTag_idx])

                    if POSTag_idx > 0:
                        if utterance_postags_list[POSTag_idx-1][0:2] == 'VB':
                            utterance_two_verb_list.append(retokenized_utterance_list[POSTag_idx-1]+retokenized_utterance_list[POSTag_idx])

                    elif POSTag[0:2] == 'NN':
                        utterance_potential_verb_list.append(retokenized_utterance_list[POSTag_idx])


            utterance_activity_list = []
            if len(utterance_verb_list) > 0 or len(utterance_two_verb_list) > 0:
                for verb in utterance_verb_list:
                    verb = verb.lower().replace('-', '').replace('_', '')

                    if verb in activities_present_dict:
                        utterance_activity_list.append(activities_present_dict[verb] + '_activity')
                    elif verb in activities_past_dict:
                        utterance_activity_list.append(activities_past2present[activities_past_dict[verb]] + '_activity')
                    else:
                        if '/' in verb:
                            for subverb in verb.split('/'):
                                if subverb in activities_present_dict:
                                    utterance_activity_list.append(activities_present_dict[subverb] + '_activity')
                                elif subverb in activities_past_dict:
                                    utterance_activity_list.append(activities_past2present[activities_past_dict[subverb]] + '_activity')

                for verb in utterance_two_verb_list:
                    verb = verb.lower().replace('-', '').replace('_', '')
                    if verb in activities_present_dict:
                        utterance_activity_list.append(activities_present_dict[verb] + '_activity')
                    elif verb in activities_past_dict:
                        utterance_activity_list.append(activities_past2present[activities_past_dict[verb]] + '_activity')


            else:
                if len(utterance_potential_verb_list) > 0:
                    if not utterance_potential_verb_list[0] == 'something':
                        verb = utterance_potential_verb_list[0]
                        if verb in activities_present_dict:
                            utterance_activity_list.append(activities_present_dict[verb] + '_activity')
                        elif verb in activities_past_dict:
                            utterance_activity_list.append(activities_past2present[activities_past_dict[verb]] + '_activity')

            if len(utterance_activity_list) == 0:
                utterance_activity_list = ['none_activity']

            found_activities_all_utterances.append(utterance_activity_list)

    while len(found_tenses_all_utterances) < len(found_entities_all_utterances):
        found_tenses_all_utterances.append('no_tenses')
        found_activities_all_utterances.append(['none_activity'])

    assert len(found_tenses_all_utterances) == len(found_activities_all_utterances)

    assert len(found_tenses_all_utterances) == len(found_entities_all_utterances)
    assert len(found_tenses_all_utterances) == len(found_commands_all_utterances)

    return preprocessed_turn, found_tenses_all_utterances, found_activities_all_utterances, found_entities_all_utterances, found_commands_all_utterances



# Preprocesses the dialogues and computes POS tags using POS tagger
def process_dialogues(dialogue_input_file, dialogue_output_file, postags_output_file):
    print 'Generating POS tags for: ', dialogue_input_file

    text = open(dialogue_input_file, 'r').readlines()
    dialogues_counted=len(text)

    processed_dialogues = ''

    for i in range(0,dialogues_counted):
        if i % 5000 == 0:
            print '     i: ' + str(i) + ' / ' + str(dialogues_counted)

        turns = text[i].strip().split(end_of_turn)

        assert len(turns) > 0 # Make sure dialogue is not empty!

        processed_dialogue = ''

        for turn in turns:
            # Process turn to get time tenses, activities, entities and commands
            processed_turn, found_tenses_all_utterances, found_activities_all_utterances, found_entities_all_utterances, found_commands_all_utterances = process_full_turn(turn)

            processed_dialogue += processed_turn + ' ' + end_of_turn


        processed_dialogues += processed_dialogue.replace('  ', ' ') + '\n'

    # Fix up dialogues for the POS Tagger
    processed_dialogues = processed_dialogues.replace(end_of_utterance, ' ' + end_of_utterance + ' ')
    processed_dialogues = processed_dialogues.replace(end_of_turn, ' ' + end_of_turn + ' ')
    processed_dialogues = processed_dialogues.replace(end_of_utterance, ' $$$###$$$ ')
    processed_dialogues = processed_dialogues.replace(end_of_turn, ' $$$### \n ')
    processed_dialogues = processed_dialogues.replace(command_placeholder, ' something ')
    processed_dialogues = processed_dialogues.replace(entity_placeholder, ' something ')
    processed_dialogues = processed_dialogues.replace('**unknown**', ' something ')
    processed_dialogues = processed_dialogues.replace('<path>', ' somewhere ')
    processed_dialogues = processed_dialogues.replace('<url>', ' somewhere ')

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



# Convert the dialogues into dialogue activity-entity representations
def process_actents(dialogue_input_file, pos_input_file, dialogue_output_file):
    print 'Processing dialogue activity-entity representations for: ', dialogue_input_file

    turns_counted = 0
    words_counted = 0

    turns_with_entities = 0
    entities_counted = 0

    turns_with_commands = 0
    commands_counted = 0

    turns_with_activities = 0
    activities_counted = 0

    activity_frequency = {}

    text = open(dialogue_input_file, 'r').readlines()
    dialogues_counted=len(text)

    POSTags = open(pos_input_file, 'r').readlines()
    POSTags_line_index = 0

    if len(POSTags[-1]) == 0:
        del POSTags[-1]

    processed_dialogue_actents = ''

    for i in range(0,dialogues_counted):
        if i % 5000 == 0:
            print '     i: ' + str(i) + ' / ' + str(dialogues_counted)

        processed_dialogue_act = ''

        current_dialogue = text[i].strip()

        turns = current_dialogue.split(end_of_turn)

        assert len(turns) > 0 # Make sure dialogue is not empty!

        add_end_of_turn_to_act = False
        if len(current_dialogue) > len(end_of_turn):
            if current_dialogue[len(current_dialogue)-len(end_of_turn):len(current_dialogue)] == end_of_turn:
                add_end_of_turn_to_act = True

        for turn in turns:
            turns_counted += 1


            # Extract appropriate POS tags for turn
            while len(POSTags[POSTags_line_index].strip().replace(' ', '').replace('\t', '')) == 0:
                POSTags_line_index += 1
                
            current_turn_POSTags = POSTags[POSTags_line_index].strip()
            current_turn_POSTags = current_turn_POSTags.replace('$$$###$$$', ' ' + end_of_utterance + ' ')
            current_turn_POSTags = current_turn_POSTags.replace('$$$###', ' ' + end_of_turn + ' ')

            POSTags_line_index += 1

            # Process turn to get time tenses, activities, entities and commands
            processed_turn, found_tenses_all_utterances, found_activities_all_utterances, found_entities_all_utterances, found_commands_all_utterances = process_full_turn(turn, current_turn_POSTags)

            # Generate activity-entity representation given time tenses, activities, entities and commands
            processed_actents = ''
            for utt_idx in range(len(found_tenses_all_utterances)):
                processed_actents += found_tenses_all_utterances[utt_idx] + ' ' + ' '.join(unique_list_elements(found_activities_all_utterances[utt_idx]))

                if len(found_entities_all_utterances[utt_idx]) > 0:
                    processed_actents += ' ' + ' '.join(unique_list_elements(found_entities_all_utterances[utt_idx]))

                if found_commands_all_utterances[utt_idx] > 0:
                    processed_actents += ' cmd ' + end_of_utterance + ' '
                else:
                    processed_actents += ' no_cmd ' + end_of_utterance + ' '



            # Verify that generated representation is valid
            assert len(turn.split(end_of_turn)) == len(processed_actents.split(end_of_turn))
            assert len(turn.split(end_of_utterance)) == len(processed_actents.split(end_of_utterance))

            processed_actents += end_of_turn + ' '

            # Print turn + activity-entity representation
            #print ''
            #print 'turn: ', processed_turn + ' ' + end_of_turn
            #print 'activity-entity representation: ', processed_actents

            processed_dialogue_act += processed_actents

            # Compute statistics for words
            words_counted += len(turn.split())

            # Compute statistics for commands
            if True in found_commands_all_utterances:
                turns_with_commands += 1
                commands_counted += 1

            # Compute statistics for entities
            if len(set(flatten_list(found_entities_all_utterances))) > 0:
                turns_with_entities += 1
                entities_counted += len(set(flatten_list(found_entities_all_utterances)))

            # Compute statistics for activities
            if len(set(flatten_list(found_activities_all_utterances))) > 0:
                turns_with_activities += 1
                activities_counted += len(set(flatten_list(found_activities_all_utterances)))

                for activity in set(flatten_list(found_activities_all_utterances)):
                    if not activity in activity_frequency:
                        activity_frequency[activity] = 0
                        
                    activity_frequency[activity] += 1

        # If necessary, delete last __eot__ token
        if not add_end_of_turn_to_act:
            if len(processed_dialogue_act) > len(end_of_turn):
                processed_dialogue_act = processed_dialogue_act[0:len(processed_dialogue_act)-(len(end_of_turn)+2)]

        processed_dialogue_actents += processed_dialogue_act.replace('  ', ' ') + '\n'

    print 'Statistics:'
    print '     entities_counted / words_counted', float(entities_counted)/float(words_counted)
    print '     turns_with_entities / turns_counted', float(turns_with_entities)/float(turns_counted)

    print '     commands_counted / words_counted', float(commands_counted)/float(words_counted)
    print '     turns_with_commands / turns_counted', float(turns_with_commands)/float(turns_counted)

    print '     activities_counted / words_counted', float(activities_counted)/float(words_counted)
    print '     turns_with_activities / turns_counted', float(turns_with_activities)/float(turns_counted)

    print '     activities_counted / turns_counted', float(activities_counted)/float(turns_counted)

    processed_dialogue_actents = processed_dialogue_actents.replace('  ', ' ')
    processed_dialogue_actents = processed_dialogue_actents.replace('  ', ' ')
    processed_dialogue_actents = processed_dialogue_actents.replace('  ', ' ')
    processed_dialogue_actents = processed_dialogue_actents.replace(' \n', '\n')
    processed_dialogue_actents = processed_dialogue_actents.replace('\n ', '\n')

    out_file = open(dialogue_output_file, 'w')
    out_file.write(processed_dialogue_actents)
    out_file.close()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dialogues", type=str, default="", help="Input dialogues")
    parser.add_argument("preprocessed_dialogues", type=str, default="", help="Intermediate file used for saving preprocessed dialogues")
    parser.add_argument("POSTags", type=str, default="", help="Intermediate file for saving POS tags")

    parser.add_argument("output_actents", type=str, default="", help="Output file containing activity-entity representations")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    process_dialogues(args.input_dialogues, args.preprocessed_dialogues, args.POSTags)
    process_actents(args.input_dialogues, args.POSTags, args.output_actents)

