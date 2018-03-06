#!/usr/bin/env python

import os
import codecs
import json
import time

from tqdm import *

from helpers import lemmatized_sentence_corpus

def get_restaurant_ids(businesses_filepath):
    restaurant_ids = set()

    # open the businesses file
    with codecs.open(businesses_filepath, encoding='utf_8') as f:

        # iterate through each line (json record) in the file
        for business_json in f:
            # convert the json record to a Python dict
            business = json.loads(business_json)

            # if this business is not a restaurant, skip to the next one
            if u'Restaurants' not in business[u'categories']:
                continue

            # add the restaurant business id to our restaurant_ids set
            restaurant_ids.add(business[u'business_id'])

    # turn restaurant_ids into a frozenset, as we don't need to change it anymore
    return frozenset(restaurant_ids)

def write_review_file(review_txt_filepath, review_json_filepath, restaurant_ids):
    review_count = 0

    # create & open a new file in write mode
    with codecs.open(review_txt_filepath, 'w', encoding='utf_8') as review_txt_file:

        # open the existing review json file
        with codecs.open(review_json_filepath, encoding='utf_8') as review_json_file:

            # loop through all reviews in the existing file and convert to dict
            for review_json in review_json_file:
                review = json.loads(review_json)

                # if this review is not about a restaurant, skip to the next one
                if review[u'business_id'] not in restaurant_ids:
                    continue

                # write the restaurant review as a line in the new file
                # escape newline characters in the original review text
                review_txt_file.write(review[u'text'].replace('\n', '\\n') + '\n')
                review_count += 1

    return review_count

def write_unigram_sents(unigram_sentences_filepath, review_txt_filepath, spacy_model):
    n_lines = 3221419
    sentence_count = 0
    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in tqdm(lemmatized_sentence_corpus(review_txt_filepath, spacy_model), total=n_lines):
            f.write(sentence + '\n')
            sentence_count += 1
            
    return sentence_count

def write_sents(sentences_filepath, sentences, phrase_model=None):
    sentence_count = 0
    with codecs.open(sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in sentences:
            sentence = u' '.join(model[sentence])
            f.write(sentence + '\n')
            sentence_count += 1

    return sentence_count
