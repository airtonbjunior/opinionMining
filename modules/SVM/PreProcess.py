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

from subprocess import Popen, PIPE

try:
    # python 2.x
    from string import punctuation, letters
except:
    # python 3.x
    from string import punctuation, ascii_letters


import re

# Some elements from http://en.wikipedia.org/wiki/List_of_emoticons
emoticons = { ':-)'   : '&happy',
              ':)'    : '&happy',
              ':o)'   : '&happy',
              ':]'    : '&happy',
              ':3'    : '&happy',
              ':c)'   : '&happy',
              ':>'    : '&happy',
              '=]'    : '&happy',
              '8)'    : '&happy',
              '=)'    : '&happy',
              ':}'    : '&happy',
              ':^)'   : '&happy',
              ':-))'  : '&happy',
              '|;-)'  : '&happy',
              ":'-)"  : '&happy',
              ":')"   : '&happy',
              '\o/'   : '&happy',
              '*\\0/*': '&happy',
              ':-D'   : '&laugh',
              ':D'    : '&laugh',
              '8-D'   : '&laugh',
              '8D'    : '&laugh',
              'x-D'   : '&laugh',
              'xD'    : '&laugh',
              'X-D'   : '&laugh',
              'XD'    : '&laugh',
              '=-D'   : '&laugh',
              '=D'    : '&laugh',
              '=-3'   : '&laugh',
              '=3'    : '&laugh',
              'B^D'   : '&laugh',
              '>:['   : '&sad',
              ':-('   : '&sad',
              ':('    : '&sad',
              ':-c'   : '&sad',
              ':c'    : '&sad',
              ':-<'   : '&sad',
              ':<'    : '&sad',
              ':-['   : '&sad',
              ':['    : '&sad',
              ':{'    : '&sad',
              ':-||'  : '&sad',
              ':@'    : '&sad',
              ":'-("  : '&sad',
              ":'("   : '&sad',
              'D:<'   : '&sad',
              'D:'    : '&sad',
              'D8'    : '&sad',
              'D;'    : '&sad',
              'D='    : '&sad',
              'DX'    : '&sad',
              'v.v'   : '&sad',
              "D-':"  : '&sad',
              '(>_<)' : '&sad',
              ':|'    : '&sad',
              '>:O'   : '&surprise',
              ':-O'   : '&surprise',
              ':-o'   : '&surprise',
              ':O'    : '&surprise',
              '°o°'   : '&surprise',
              ':O'    : '&surprise',
              'o_O'   : '&surprise',
              'o_0'   : '&surprise',
              'o.O'   : '&surprise',
              '8-0'   : '&surprise',
              '|-O'   : '&surprise',
              ';-)'   : '&wink',
              ';)'    : '&wink',
              '*-)'   : '&wink',
              '*)'    : '&wink',
              ';-]'   : '&wink',
              ';]'    : '&wink',
              ';D'    : '&wink',
              ';^)'   : '&wink',
              ':-,'   : '&wink',
              '>:P'   : '&tong',
              ':-P'   : '&tong',
              ':P'    : '&tong',
              'X-P'   : '&tong',
              'x-p'   : '&tong',
              'xp'    : '&tong',
              'XP'    : '&tong',
              ':-p'   : '&tong',
              ':p'    : '&tong',
              '=p'    : '&tong',
              ':-Þ'   : '&tong',
              ':Þ'    : '&tong',
              ':-b'   : '&tong',
              ':b'    : '&tong',
              ':-&'   : '&tong',
              ':&'    : '&tong',
              '>:\\'  : '&annoyed',
              '>:/'   : '&annoyed',
              ':-/'   : '&annoyed',
              ':-.'   : '&annoyed',
              ':/'    : '&annoyed',
              ':\\'   : '&annoyed',
              '=/'    : '&annoyed',
              '=\\'   : '&annoyed',
              ':L'    : '&annoyed',
              '=L'    : '&annoyed',
              ':S'    : '&annoyed',
              '>.<'   : '&annoyed',
              ':-|'   : '&annoyed',
              '<:-|'  : '&annoyed',
              ':-X'   : '&seallips',
              ':X'    : '&seallips',
              ':-#'   : '&seallips',
              ':#'    : '&seallips',
              'O:-)'  : '&angel',
              '0:-3'  : '&angel',
              '0:3'   : '&angel',
              '0:-)'  : '&angel',
              '0:)'   : '&angel',
              '0;^)'  : '&angel',
              '>:)'   : '&devil',
              '>;)'   : '&devil',
              '>:-)'  : '&devil',
              '}:-)'  : '&devil',
              '}:)'   : '&devil',
              '3:-)'  : '&devil',
              '3:)'   : '&devil',
              'o/\o'  : '&highfive',
              '^5'    : '&highfive',
              '>_>^'  : '&highfive',
              '^<_<'  : '&highfive',
              '<3'    : '&heart'
          }


### Provides a pre-process for tweet messages.
### Replace emoticons, hash, mentions and urls for codes
### Correct long seguences of letters and punctuations
### Apply the nltk part-of_speech tagger to the message
def pre_process(tweet_messages):

    ark_tweet_cmd = ['Tools/ark-tweet-nlp/runTagger.sh', '--input-format', 'text', '--output-format', 'conll', '--no-confidence', '--quiet']

    tweets = '\n'.join(tweet_messages)
    tweets = tweets.encode("ascii","ignore")
    # Run the tagger and get the output
    p = Popen(ark_tweet_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate(input=tweets)
    stdout = stdout.decode('utf8','ignore')
    ark_tweet_output = stdout.strip() + '\n'

    # Check the return code.
    if p.returncode != 0:
        print ('Tools/ark-tweet-nlp/runTagger.sh command failed! Details: %s\n%s' % (stderr,ark_tweet_output))
        return None

    tweet_tokens_list = list()
    tweet_tokens = list()
    lines = ark_tweet_output.split('\n')
    for line in lines:
        values = re.split(r'[ \t]',line)
        values = [t for t in values if len(t) != 0]
        if len(values) == 0:
            tweet_tokens_list.append(tweet_tokens)
            tweet_tokens = list()
            continue
        try:
            # Word and POS
            tweet_tokens.append( (values[0],values[1]) )
        except:
            print ('Error reading art tweet tagger output line: ' + line)

    for tweet_tokens in tweet_tokens_list:
        for index in range(len(tweet_tokens)):
            token = tweet_tokens[index][0]
            tag = tweet_tokens[index][1]

            # substitute mentions
            if tag == '@':
                tweet_tokens[index] = ('&mention',tag)

            # substitute urls
            if tag == 'U':
                tweet_tokens[index] = ('&url',tag)

            # substitute emoticions
            if tag == 'E':
                tweet_tokens[index] = (emoticons.get(token,'_'),tag)

    # return the tweet in the format [(word,tag),...]
    return tweet_tokens_list