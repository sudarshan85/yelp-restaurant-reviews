#!/usr/bin/env python

import os
import codecs
import json
import time

from spacy.lang.en.stop_words import STOP_WORDS

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space


def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')

            
def lemmatized_sentence_corpus(filename, spacy_model):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in spacy_model.pipe(line_review(filename), batch_size=10000, n_threads=8):        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])

            
def full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)  

        
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
    sentence_count = 0
    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus(review_txt_filepath, spacy_model):
            f.write(sentence + '\n')
            sentence_count += 1
            
    return sentence_count


def write_sents(sentences_filepath, sentences, model):
    sentence_count = 0
    with codecs.open(sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in sentences:
            sentence = u' '.join(model[sentence])
            f.write(sentence + '\n')
            sentence_count += 1

    return sentence_count


def write_trigram_review(trigram_reviews_filepath, review_txt_filepath, bigram_model, trigram_model, spacy_model):
    review_count = 0
    with codecs.open(trigram_reviews_filepath, 'w', encoding='utf_8') as f:
        for parsed_review in spacy_model.pipe(line_review(review_txt_filepath), batch_size=10000, n_threads=8):

            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_review if not punct_space(token)]

            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]

            # remove any remaining stopwords
            trigram_review = [term for term in trigram_review if term not in STOP_WORDS]

            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')
            review_count += 1    
    
    return review_count        