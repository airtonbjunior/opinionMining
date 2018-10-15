# -*- coding: utf-8 -*-

#########################################################################
############## Semeval - Sentiment Analysis in Twitter  #################
#########################################################################

####
#### Authors: Pedro Paulo Balage Filho e Lucas Avanço
#### Version: 2.0
#### Date: 26/03/14
####

# Python 3 compatibility
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

#### Provides a rule-based classifier used in Semeval's TwitterHybridClassifier ###
class RulesClassifier(object):

    # Classifies the tweet_message using rules. These looks for emoticons,  basically.
    # The message myst be pre-processed in the format (w,tag)
    def classify(self, tweet_tokens):

        positive_patterns = []
        negative_patterns = []

        # emoticons are substituted by codes in the pre-process step
        pos_patterns = ['&happy',
                        '&laugh',
                        '&wink',
                        '&heart',
                        '&highfive',
                        '&angel',
                        '&tong',
                       ]

        neg_patterns = ['&sad',
                        '&annoyed',
                        '&seallips',
                        '&devil',
                       ]

        # how many positive and negative emoticons are in the message?
        matches_pos = [token for token,tag in tweet_tokens if token in pos_patterns]
        matches_neg = [token for token,tag in tweet_tokens if token in neg_patterns]

        # return (positive_score , negative_score). Number of emoticons for
        # each sentiment
        return ( len(matches_pos),-1*len(matches_neg) )