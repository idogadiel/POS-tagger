import tagger

print('start')

annotated_corpus = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(annotated_corpus)

untamed_sentence = 'Brian sat down for dinner. He sat down in the chair. He sat down at the table. He looked at his white plate. He looked at his silver fork. He looked at his silver spoon. His dad said, "Pass me your plate, Brian." His dad put white rice on the plate. His dad put yellow corn on the plate. His dad put green peas on the plate. He gave the plate back to Brian. "This looks delicious," Brian said. "It is delicious," his dad said. Brian wondered why corn was yellow. He wondered why peas were green. He wondered if there were yellow peas and green corn.'
untamed_sentence = untamed_sentence.replace('.', ' .')
sentence = untamed_sentence.split()

baseline_tagged_sentence = tagger.baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts)
hmm_result = tagger.hmm_tag_sentence(sentence, A, B)
print(baseline_tagged_sentence)
print(hmm_result)

tagged_sentence = [('American', 'ADJ'),
                   ('forces', 'NOUN'),
                   ('killed', 'VERB'),
                   ('Shaikh', 'PROPN'),
                   ('Abdullah', 'PROPN')]
tagger.joint_prob(tagged_sentence, A, B)

print('end')
