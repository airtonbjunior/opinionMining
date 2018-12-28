import time
import jamspell

import variables

# log time
start = time.time()

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('/home/airton/en.bin')


#with open(variables.TRAIN_WORDS, 'r') as f:
with open(variables.TEST_WORDS, 'r') as f:
    for line in f:
    	#if(line.strip() != corrector.FixFragment(line.strip())):
    	with open(variables.TEST_WORDS_SPELLCHECK, 'a') as fw:
    		fw.write(corrector.FixFragment(line.strip()) + "\n")
    		#print("[CORRECTED] " + line.strip() + " -> " + corrector.FixFragment(line.strip()))
    	#print(line.strip() + " -> " + corrector.FixFragment(line.strip()))

#TRAIN_WORDS_SPELLCHECK

#print(corrector.FixFragment('I am the begt spell cherken!'))

end = time.time()
print("[spell checker function ended][" + str(format(end - start, '.3g')) + " seconds]\n")