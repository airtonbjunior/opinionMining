import time

from variables import *
from functions import *

# log time
start = time.time()


if __name__ == "__main__":
    getDictionary()

    # Dictionaries used on test
    # v1: LIU Pos/Neg and AFFIN, hashtags Pos/Neg, Emoticons Pos/Neg, invertWords

#    functions_to_evaluate = [
#        "polaritySum(x)", #baseline
#        "if_then_else(hasEmoticons(x), emoticonsPolaritySum(removeLinks(x)), polaritySum(removeEllipsis(removeLinks(lemmingText(removeAllPonctuation(replaceNegatingWords(x)))))))",
#        "protectedDiv(cos(emoticonsPolaritySum(removeAllPonctuation(removeLinks(removeStopWords(stemmingText(stemmingText(removeAllPonctuation(removeLinks(x))))))))), invertSignal(if_then_else(hasEmoticons(removeStopWords(removeLinks(x))), negativeEmoticons(stemmingText(removeLinks(stemmingText(x)))), negativeWordsQuantity(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeEllipsis(x)))))))))))",
#        "add(if_then_else(hasHashtag(removeAllPonctuation(removeLinks(removeStopWords(replaceNegatingWords(removeStopWords(removeStopWords(removeEllipsis(stemmingText(stemmingText(removeEllipsis(x))))))))))), 0.5759342653620698, protectedDiv(invertSignal(negativeWordsQuantity(removeAllPonctuation(removeLinks(removeEllipsis(x))))), 0.5759342653620698)), polaritySum(removeEllipsis(replaceNegatingWords(replaceNegatingWords(removeLinks(removeEllipsis(removeAllPonctuation(removeStopWords(removeLinks(x))))))))))",
#        "add(polaritySum(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeStopWords(removeAllPonctuation(replaceNegatingWords(removeAllPonctuation(removeStopWords(replaceNegatingWords(x)))))))))), sin(if_then_else(hasEmoticons(stemmingText(stemmingText(x))), if_then_else(hasEmoticons(removeAllPonctuation(x)), positiveHashtags(stemmingText(removeEllipsis(removeLinks(x)))), polaritySum(removeEllipsis(removeEllipsis(removeLinks(removeAllPonctuation(stemmingText(removeAllPonctuation(x)))))))), polaritySum(removeEllipsis(removeEllipsis(stemmingText(x)))))))",
#        "add(emoticonsPolaritySum(removeLinks(removeEllipsis(stemmingText(stemmingText(removeLinks(stemmingText(removeStopWords(x)))))))), polaritySum(removeLinks(replaceNegatingWords(removeAllPonctuation(removeStopWords(replaceNegatingWords(x)))))))",
#        "polaritySum(removeLinks(replaceNegatingWords(removeStopWords(removeLinks(replaceNegatingWords(removeStopWords(removeEllipsis(replaceNegatingWords(removeEllipsis(replaceNegatingWords(removeAllPonctuation(removeAllPonctuation(replaceBoosterWords(x))))))))))))))",
#        "polaritySum(removeLinks(replaceNegatingWords(removeAllPonctuation(replaceNegatingWords(removeAllPonctuation(replaceNegatingWords(replaceBoosterWords(x))))))))",
#        "polaritySum(removeAllPonctuation(removeEllipsis(removeStopWords(removeEllipsis(removeAllPonctuation(replaceNegatingWords(replaceNegatingWords(replaceNegatingWords(replaceBoosterWords(x))))))))))",
#        "polaritySum(replaceNegatingWords(removeAllPonctuation(removeStopWords(removeStopWords(boostUpperCase(removeStopWords(boostUpperCase(removeStopWords(boostUpperCase(replaceNegatingWords(replaceBoosterWords(x))))))))))))",
#        "cos(add(sub(protectedLog(add(if_then_else(hasEmoticons(boostUpperCase(x)), emoticonsPolaritySum(removeStopWords(boostUpperCase(removeEllipsis(x)))), if_then_else(hasEmoticons(boostUpperCase(x)), 0.7550559511929338, positiveHashtags(removeLinks(removeAllPonctuation(removeLinks(removeAllPonctuation(replaceNegatingWords(x)))))))), add(if_then_else(hasEmoticons(boostUpperCase(x)), emoticonsPolaritySum(removeStopWords(boostUpperCase(x))), if_then_else(hasEmoticons(stemmingText(replaceBoosterWords(replaceNegatingWords(removeLinks(removeAllPonctuation(removeLinks(x))))))), if_then_else(hasEmoticons(boostUpperCase(x)), 0.7550559511929338, negativeEmoticons(removeLinks(x))), positiveHashtags(stemmingText(boostUpperCase(removeLinks(x)))))), sub(positiveHashtags(replaceNegatingWords(removeLinks(stemmingText(replaceBoosterWords(replaceNegatingWords(removeLinks(removeAllPonctuation(removeLinks(x))))))))), sub(if_then_else(hasEmoticons(boostUpperCase(x)), if_then_else(hasEmoticons(boostUpperCase(x)), positiveWordsQuantity(removeLinks(removeAllPonctuation(removeLinks(removeAllPonctuation(x))))), cos(positiveHashtags(boostUpperCase(removeLinks(x))))), positiveHashtags(stemmingText(boostUpperCase(removeLinks(boostUpperCase(x)))))), polaritySum(replaceNegatingWords(replaceBoosterWords(boostUpperCase(boostUpperCase(removeAllPonctuation(replaceNegatingWords(replaceNegatingWords(x))))))))))))), hashtagPolaritySum(replaceNegatingWords(replaceNegatingWords(removeLinks(removeAllPonctuation(removeLinks(x))))))), 0.7550559511929338))",
#        "polaritySum(removeStopWords(removeAllPonctuation(replaceBoosterWords(replaceBoosterWords(replaceNegatingWords(x))))))",
#        "polaritySum(removeStopWords(removeEllipsis(removeEllipsis(removeAllPonctuation(replaceBoosterWords(replaceNegatingWords(x)))))))",
#        "mul(0.5326220427886663, polaritySum(removeStopWords(removeStopWords(removeAllPonctuation(removeStopWords(removeStopWords(removeAllPonctuation(removeStopWords(removeAllPonctuation(replaceBoosterWords(replaceNegatingWords(x))))))))))))",
#        "cos(if_then_else(hasEmoticons(removeStopWords(removeAllPonctuation(x))), if_then_else(hasEmoticons(removeAllPonctuation(removeAllPonctuation(x))), negativeWordsQuantity(replaceBoosterWords(removeAllPonctuation(x))), add(negativeWordsQuantity(removeStopWords(removeAllPonctuation(removeStopWords(removeAllPonctuation(stemmingText(replaceBoosterWords(stemmingText(removeAllPonctuation(x))))))))), invertSignal(-0.9744494532008883))), add(negativeWordsQuantity(removeAllPonctuation(removeAllPonctuation(removeEllipsis(x)))), invertSignal(-0.9744494532008883))))",
#        "polaritySum(boostUpperCase(replaceNegatingWords(removeAllPonctuation(boostUpperCase(removeStopWords(boostUpperCase(removeStopWords(replaceBoosterWords(boostUpperCase(replaceNegatingWords(removeStopWords(replaceBoosterWords(replaceNegatingWords(x))))))))))))))"
#    ]

    # Start the 30 models (LIU + AFFIN considering the real polarities)
    function_to_evaluate = [
        "polaritySum(removeAllPonctuation(removeEllipsis(removeEllipsis(replaceBoosterWords(removeLinks(removeEllipsis(removeStopWords(removeStopWords(replaceBoosterWords(removeAllPonctuation(replaceNegatingWords(x))))))))))))",
        "polaritySum(replaceBoosterWords(removeStopWords(replaceBoosterWords(removeAllPonctuation(removeLinks(replaceBoosterWords(removeStopWords(replaceBoosterWords(replaceNegatingWords(removeLinks(boostUpperCase(x))))))))))))",
        "polaritySum(removeLinks(boostUpperCase(removeLinks(boostUpperCase(replaceBoosterWords(removeLinks(removeAllPonctuation(removeStopWords(replaceBoosterWords(removeAllPonctuation(replaceNegatingWords(replaceNegatingWords(x)))))))))))))",
        "polaritySum(removeLinks(replaceBoosterWords(removeEllipsis(replaceBoosterWords(removeLinks(removeStopWords(replaceNegatingWords(replaceBoosterWords(replaceBoosterWords(removeStopWords(replaceNegatingWords(replaceBoosterWords(replaceBoosterWords(removeAllPonctuation(replaceNegatingWords(x))))))))))))))))",
        "polaritySum(removeStopWords(replaceBoosterWords(removeStopWords(replaceBoosterWords(removeLinks(boostUpperCase(replaceBoosterWords(removeAllPonctuation(replaceNegatingWords(x))))))))))",
        "polaritySum(boostUpperCase(removeStopWords(removeAllPonctuation(replaceBoosterWords(removeAllPonctuation(removeAllPonctuation(removeLinks(removeAllPonctuation(removeLinks(removeAllPonctuation(removeAllPonctuation(removeEllipsis(replaceNegatingWords(removeLinks(x)))))))))))))))",
    ]   "polaritySum(removeLinks(boostUpperCase(replaceBoosterWords(removeLinks(boostUpperCase(replaceBoosterWords(removeStopWords(removeEllipsis(replaceBoosterWords(removeStopWords(removeEllipsis(replaceBoosterWords(removeAllPonctuation(replaceNegatingWords(x)))))))))))))))"

    for function_to_evaluate in functions_to_evaluate:
        evaluateMessages("tweets2013", function_to_evaluate)
        evaluateMessages("tweets2014", function_to_evaluate)
        evaluateMessages("sms", function_to_evaluate)
        evaluateMessages("livejournal", function_to_evaluate)
        evaluateMessages("sarcasm", function_to_evaluate)
        evaluateMessages("all", function_to_evaluate)

    resultsAnalysis()


# log time
end = time.time()
print("\n\n[Script ends after " + str(format(end - start, '.3g')) + " seconds]")




# SANDBOX
# other functions tested
    # Baseline
    # [TEST v1]:  [53.85 all] [28.41 sarcasm] [62.00 liveJournal] [48.84 sms] [50.82 tweets2014] [54.54 tweets2013]
    #function_to_evaluate = "polaritySum(x)"

    # Modified Baseline
    # [TEST v1]:  [58.76 all] [41.68 sarcasm] [64.16 liveJournal] [53.34 sms] [57.03 tweets2014] [59.75 tweets2013]
    #function_to_evaluate = "if_then_else(hasEmoticons(x), emoticonsPolaritySum(removeLinks(x)), polaritySum(removeEllipsis(removeLinks(lemmingText(removeAllPonctuation(replaceNegatingWords(x)))))))"

    # [TRAIN]: F1 0.5994326849864792 [2067 correct evaluations] [1103 positives, 964 negatives and 0 neutrals]
    # [TEST v1]:  [53.21 all] [43.52 sarcasm] [57.19 liveJournal] [41.92 sms] [55.13 tweets2014] [55.74 tweets2013]
    #function_to_evaluate = "protectedDiv(cos(emoticonsPolaritySum(removeAllPonctuation(removeLinks(removeStopWords(stemmingText(stemmingText(removeAllPonctuation(removeLinks(x))))))))), invertSignal(if_then_else(hasEmoticons(removeStopWords(removeLinks(x))), negativeEmoticons(stemmingText(removeLinks(stemmingText(x)))), negativeWordsQuantity(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeEllipsis(x)))))))))))"

    # [TRAIN]: F1 0.6369658710978381 [2520 correct evaluations] [896 positives, 1004 negatives and 620 neutrals]
    # [TEST v1]:  [57.01 all] [45.05 sarcasm] [62.72 liveJournal] [52.31 sms] [54.2 tweets2014] [58.09 tweets2013]
    #function_to_evaluate = "add(if_then_else(hasHashtag(removeAllPonctuation(removeLinks(removeStopWords(replaceNegatingWords(removeStopWords(removeStopWords(removeEllipsis(stemmingText(stemmingText(removeEllipsis(x))))))))))), 0.5759342653620698, protectedDiv(invertSignal(negativeWordsQuantity(removeAllPonctuation(removeLinks(removeEllipsis(x))))), 0.5759342653620698)), polaritySum(removeEllipsis(replaceNegatingWords(replaceNegatingWords(removeLinks(removeEllipsis(removeAllPonctuation(removeStopWords(removeLinks(x))))))))))"

    # [TRAIN]: F1 0.6232731810549077 [2444 correct evaluations] [1016 positives, 831 negatives and 597 neutrals]
    # [TEST v1]:  [58.78 all] [40.19 sarcasm] [66.21 liveJournal] [55.52 sms] [56.41 tweets2014] [58.42 tweets2013]
    #function_to_evaluate = "add(polaritySum(removeAllPonctuation(removeAllPonctuation(removeAllPonctuation(removeStopWords(removeAllPonctuation(replaceNegatingWords(removeAllPonctuation(removeStopWords(replaceNegatingWords(x)))))))))), sin(if_then_else(hasEmoticons(stemmingText(stemmingText(x))), if_then_else(hasEmoticons(removeAllPonctuation(x)), positiveHashtags(stemmingText(removeEllipsis(removeLinks(x)))), polaritySum(removeEllipsis(removeEllipsis(removeLinks(removeAllPonctuation(stemmingText(removeAllPonctuation(x)))))))), polaritySum(removeEllipsis(removeEllipsis(stemmingText(x)))))))"

    # [TRAIN]: F1 0.6298756475178069 [2470 correct evaluations] [1018 positives, 763 negatives and 689 neutrals]
    # [TEST]:  [59.07 all] [35.64 sarcasm] [64.15 liveJournal] [54.94 sms] [57.22 tweets2014] [59.82 tweets2013]
    #function_to_evaluate = "add(emoticonsPolaritySum(removeLinks(removeEllipsis(stemmingText(stemmingText(removeLinks(stemmingText(removeStopWords(x)))))))), polaritySum(removeLinks(replaceNegatingWords(removeAllPonctuation(removeStopWords(replaceNegatingWords(x)))))))"

    # [TRAIN]: F1 0.6161622143264465 [2425 correct evaluations] [977 positives, 752 negatives and 696 neutrals]
    # [TEST v1]: [58.69 all] [35.44 sarcasm] [64.39 liveJournal] [55.5 sms] [56.59 tweets2014] [59.01 tweets2013]
    #function_to_evaluate = "polaritySum(removeLinks(replaceNegatingWords(removeStopWords(removeLinks(replaceNegatingWords(removeStopWords(removeEllipsis(replaceNegatingWords(removeEllipsis(replaceNegatingWords(removeAllPonctuation(removeAllPonctuation(replaceBoosterWords(x))))))))))))))"

    # [TRAIN]: F1 0.6124793970758737 [2418 correct evaluations] [984 positives, 738 negatives and 696 neutrals]
    # [TEST v1]: [58.79 all] [35.44 sarcasm] [64.39 liveJournal] [55.44 sms] [56.77 tweets2014] [59.15 tweets2013]
    #function_to_evaluate = "polaritySum(removeLinks(replaceNegatingWords(removeAllPonctuation(replaceNegatingWords(removeAllPonctuation(replaceNegatingWords(replaceBoosterWords(x))))))))"
    
    # [TRAIN]: F1 0.6124793970758737 [2418 correct evaluations] [984 positives, 738 negatives and 696 neutrals]
    # [TEST v1]: [58.75 all] [37.18 sarcasm] [64.39 liveJournal] [55.56 sms] [56.59 tweets2014] [59.06 tweets2013]
    #function_to_evaluate = "polaritySum(removeAllPonctuation(removeEllipsis(removeStopWords(removeEllipsis(removeAllPonctuation(replaceNegatingWords(replaceNegatingWords(replaceNegatingWords(replaceBoosterWords(x))))))))))"

    # [TRAIN]: F1 0.6165577280275001 [2426 correct evaluations] [978 positives, 753 negatives and 695 neutrals]
    # [TEST]: [58.83 all] [35.44 sarcasm] [65.07 liveJournal] [55.59 sms] [56.63 tweets2014] [58.98 tweets2013]
    #function_to_evaluate = "polaritySum(replaceNegatingWords(removeAllPonctuation(removeStopWords(removeStopWords(boostUpperCase(removeStopWords(boostUpperCase(removeStopWords(boostUpperCase(replaceNegatingWords(replaceBoosterWords(x))))))))))))"

    # [TRAIN]: F1 0.6265620378586217 [2181 correct evaluations] [1034 positives, 1147 negatives and 0 neutrals]
    # [TEST]: [53.28 all] [50.33 sarcasm] [59.75 liveJournal] [48.00 sms] [50.4 tweets2014] [53.98 tweets2013]
    #function_to_evaluate = "cos(add(sub(protectedLog(add(if_then_else(hasEmoticons(boostUpperCase(x)), emoticonsPolaritySum(removeStopWords(boostUpperCase(removeEllipsis(x)))), if_then_else(hasEmoticons(boostUpperCase(x)), 0.7550559511929338, positiveHashtags(removeLinks(removeAllPonctuation(removeLinks(removeAllPonctuation(replaceNegatingWords(x)))))))), add(if_then_else(hasEmoticons(boostUpperCase(x)), emoticonsPolaritySum(removeStopWords(boostUpperCase(x))), if_then_else(hasEmoticons(stemmingText(replaceBoosterWords(replaceNegatingWords(removeLinks(removeAllPonctuation(removeLinks(x))))))), if_then_else(hasEmoticons(boostUpperCase(x)), 0.7550559511929338, negativeEmoticons(removeLinks(x))), positiveHashtags(stemmingText(boostUpperCase(removeLinks(x)))))), sub(positiveHashtags(replaceNegatingWords(removeLinks(stemmingText(replaceBoosterWords(replaceNegatingWords(removeLinks(removeAllPonctuation(removeLinks(x))))))))), sub(if_then_else(hasEmoticons(boostUpperCase(x)), if_then_else(hasEmoticons(boostUpperCase(x)), positiveWordsQuantity(removeLinks(removeAllPonctuation(removeLinks(removeAllPonctuation(x))))), cos(positiveHashtags(boostUpperCase(removeLinks(x))))), positiveHashtags(stemmingText(boostUpperCase(removeLinks(boostUpperCase(x)))))), polaritySum(replaceNegatingWords(replaceBoosterWords(boostUpperCase(boostUpperCase(removeAllPonctuation(replaceNegatingWords(replaceNegatingWords(x))))))))))))), hashtagPolaritySum(replaceNegatingWords(replaceNegatingWords(removeLinks(removeAllPonctuation(removeLinks(x))))))), 0.7550559511929338))"

    # [TRAIN]: F1 0.6175298075276515 [2429 correct evaluations] [978 positives, 755 negatives and 696 neutrals]
    # [TEST]: [58.8 all] [37.18 sarcasm] [64.56 liveJournal] [55.69 sms] [56.59 tweets2014] [59.06 tweets2013]
    #function_to_evaluate = "polaritySum(removeStopWords(removeAllPonctuation(replaceBoosterWords(replaceBoosterWords(replaceNegatingWords(x))))))"

    #function_to_evaluate = "add(invertSignal(negativeWordsQuantity(x)), sin(polaritySum(x)))"
    #function_to_evaluate = "sub(mul(sub(polaritySum(removeStopWords(stemmingText(x))), sin(-0.3012931295024437)), protectedLog(-0.21199248533470838)), protectedLog(protectedLog(sub(polaritySum(stemmingText(stemmingText(stemmingText(removeStopWords(removeStopWords(removeStopWords(stemmingText(removeStopWords(stemmingText(removeStopWords(x))))))))))), sub(hashtagPolaritySum(removeStopWords(removeStopWords(removeStopWords(x)))), cos(protectedLog(polaritySum(stemmingText(removeStopWords(removeStopWords(x)))))))))))"
    #function_to_evaluate = "mul(add(add(polaritySum(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(x)))))))))))), positiveEmoticons(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(x)))))))), mul(sub(sin(-0.7500287440821918), protectedDiv(negativeEmoticons(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(x))))))), protectedSqrt(protectedDiv(protectedLog(0.30225574066002103), cos(0.3289974105155071))))), protectedDiv(sin(negativeWordsQuantity(repeatInputString(repeatInputString(repeatInputString(repeatInputString(x)))))), add(protectedSqrt(cos(mul(hashtagPolaritySum(x), -1.1631941015415768))), -0.27062630818833844)))), mul(protectedDiv(protectedLog(-0.9481590665673725), negativeEmoticons(x)), exp(add(-0.28621032356521914, -0.21595094634073808))))"
    #function_to_evaluate = "add(emoticonsPolaritySum(repeatInputString(repeatInputString(repeatInputString(repeatInputString(repeatInputString(x)))))), polaritySum(repeatInputString(repeatInputString(repeatInputString(repeatInputString(x))))))"
    #function_to_evaluate = "if_then_else(hasEmoticons(removeEllipsis(x)), sub(cos(protectedSqrt(sub(sin(protectedDiv(polaritySum(replaceNegatingWords(x)), hashtagPolaritySum(x))), hashtagPolaritySum(removeAllPonctuation(removeAllPonctuation(x)))))), protectedLog(emoticonsPolaritySum(removeEllipsis(removeLinks(stemmingText(removeEllipsis(removeEllipsis(x)))))))), polaritySum(removeAllPonctuation(removeLinks(removeAllPonctuation(removeStopWords(removeLinks(removeLinks(removeAllPonctuation(replaceNegatingWords(x))))))))))"
    #function_to_evaluate = "add(if_then_else(hasEmoticons(removeEllipsis(removeEllipsis(x))), sub(emoticonsPolaritySum(x), polaritySum(removeAllPonctuation(removeStopWords(replaceNegatingWords(x))))), polaritySum(stemmingText(removeEllipsis(removeStopWords(x))))), polaritySum(removeAllPonctuation(removeLinks(replaceNegatingWords(x)))))"
    #function_to_evaluate = "if_then_else(hasEmoticons(stemmingText(x)), emoticonsPolaritySum(removeEllipsis(removeStopWords(removeEllipsis(stemmingText(x))))), polaritySum(removeAllPonctuation(removeEllipsis(removeStopWords(removeAllPonctuation(removeLinks(replaceNegatingWords(removeAllPonctuation(removeLinks(removeAllPonctuation(removeEllipsis(x))))))))))))"
    #function_to_evaluate = "add(polaritySum(removeStopWords(removeEllipsis(removeAllPonctuation(removeAllPonctuation(removeLinks(removeAllPonctuation(replaceNegatingWords(removeEllipsis(removeStopWords(replaceNegatingWords(removeEllipsis(x)))))))))))), sub(mul(exp(if_then_else(True, -1.6617453540075866, -0.1240560689118353)), negativeWordsQuantity(replaceNegatingWords(removeAllPonctuation(replaceNegatingWords(x))))), negativeWordsQuantity(removeAllPonctuation(stemmingText(x)))))"
    #function_to_evaluate = "if_then_else(hasEmoticons(stemmingText(x)), emoticonsPolaritySum(removeEllipsis(removeStopWords(removeEllipsis(stemmingText(x))))), polaritySum(removeAllPonctuation(removeStopWords(replaceNegatingWords(removeEllipsis(removeEllipsis(x)))))))"
    #function_to_evaluate = "if_then_else(hasEmoticons(replaceNegatingWords(removeLinks(removeStopWords(removeLinks(x))))), emoticonsPolaritySum(x), polaritySum(removeAllPonctuation(removeAllPonctuation(removeEllipsis(removeEllipsis(removeEllipsis(replaceNegatingWords(replaceNegatingWords(removeStopWords(removeAllPonctuation(replaceNegatingWords(x))))))))))))"
    #function_to_evaluate = "polaritySum(removeAllPonctuation(removeEllipsis(removeStopWords(removeEllipsis(removeAllPonctuation(replaceNegatingWords(removeLinks(removeLinks(x)))))))))"
    #function_to_evaluate = "sub(polaritySum(replaceNegatingWords(removeAllPonctuation(removeAllPonctuation(replaceNegatingWords(removeStopWords(x)))))), sin(sin(sub(protectedDiv(negativeWordsQuantity(replaceNegatingWords(replaceNegatingWords(x))), exp(sin(hashtagPolaritySum(removeLinks(removeEllipsis(x)))))), emoticonsPolaritySum(removeStopWords(replaceNegatingWords(x)))))))"
    #function_to_evaluate = "if_then_else(hasEmoticons(x), protectedLog(invertSignal(emoticonsPolaritySum(x))), polaritySum(removeStopWords(replaceNegatingWords(removeAllPonctuation(replaceNegatingWords(replaceNegatingWords(x)))))))"
    #function_to_evaluate = "sub(add(polaritySum(replaceNegatingWords(removeStopWords(removeAllPonctuation(stemmingText(stemmingText(removeAllPonctuation(x))))))), add(add(emoticonsPolaritySum(x), emoticonsPolaritySum(x)), invertSignal(negativeWordsQuantity(removeLinks(removeLinks(removeEllipsis(removeLinks(removeStopWords(removeEllipsis(x)))))))))), invertSignal(add(polaritySum(replaceNegatingWords(removeStopWords(replaceNegatingWords(removeStopWords(replaceNegatingWords(replaceNegatingWords(replaceNegatingWords(removeStopWords(replaceNegatingWords(removeLinks(removeAllPonctuation(x)))))))))))), add(add(add(polaritySum(x), add(polaritySum(removeLinks(removeEllipsis(replaceNegatingWords(removeEllipsis(replaceNegatingWords(removeStopWords(removeEllipsis(x)))))))), add(polaritySum(replaceNegatingWords(replaceNegatingWords(removeStopWords(removeAllPonctuation(stemmingText(removeStopWords(x))))))), emoticonsPolaritySum(x)))), polaritySum(replaceNegatingWords(removeAllPonctuation(removeStopWords(replaceNegatingWords(replaceNegatingWords(replaceNegatingWords(x)))))))), polaritySum(replaceNegatingWords(removeAllPonctuation(removeStopWords(replaceNegatingWords(replaceNegatingWords(replaceNegatingWords(x)))))))))))"