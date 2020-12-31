import tagger

print('start')
annotated_corpus = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(annotated_corpus)

sentence = ['my',
            'back',
            'really',
            'hurt']

tagged_sentence = [('American', 'ADJ'),
                   ('forces', 'NOUN'),
                   ('killed', 'VERB'),
                   ('Shaikh', 'PROPN'),
                   ('Abdullah', 'PROPN')]

# baseline_tagged_sentence = tagger.baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts)
#
hmm_result = tagger.hmm_tag_sentence(sentence, A, B)
print(hmm_result)
tagger.joint_prob(tagged_sentence, A, B)
print('end')
