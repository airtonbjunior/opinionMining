""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


auxFunctions.py
	Aux Functions used by the Genetic Programming Algorithm

"""
import variables

def saveBestValues(individual, accuracy, precision, recall, f1):
    if accuracy > variables.BEST['accuracy']:
        variables.BEST['accuracy'] = accuracy     

    if precision['positive'] > variables.BEST['precision']['positive']:
        variables.BEST['precision']['positive'] = precision['positive']

    if precision['negative'] > variables.BEST['precision']['negative']:
        variables.BEST['precision']['negative'] = precision['negative']
    
    if precision['neutral'] > variables.BEST['precision']['neutral']:
        variables.BEST['precision']['neutral'] = precision['neutral']

    if precision['avg'] > variables.BEST['precision']['avg']:
        variables.BEST['precision']['avg'] = precision['avg']
        variables.BEST['precision']['avg_function'] = str(individual)        

    if recall['positive'] > variables.BEST['recall']['positive']:
        variables.BEST['recall']['positive'] = recall['positive']

    if recall['negative'] > variables.BEST['recall']['negative']:
        variables.BEST['recall']['negative'] = recall['negative']

    if recall['neutral'] > variables.BEST['recall']['neutral']:
        variables.BEST['recall']['neutral'] = recall['neutral']

    if recall['avg'] > variables.BEST['recall']['avg']:
        variables.BEST['recall']['avg'] = recall['avg']
        variables.BEST['recall']['avg_function'] = str(individual)        

    if f1['positive'] > variables.BEST['f1']['positive']:
        variables.BEST['f1']['positive'] = f1['positive']
   
    if f1['negative'] > variables.BEST['f1']['negative']:
        variables.BEST['f1']['negative'] = f1['negative']
   
    if f1['neutral'] > variables.BEST['f1']['neutral']:
        variables.BEST['f1']['neutral'] = f1['neutral']         
    
    if f1['avg'] > variables.BEST['f1']['avg']:
        variables.BEST['f1']['avg'] = f1['avg']
        variables.BEST['f1']['avg_function'] = str(individual)           

    if f1['avg_pn'] > variables.BEST['f1']['avg_pn']:
        variables.BEST['f1']['avg_pn'] = f1['avg_pn']