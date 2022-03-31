#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import math

#select accession, institution_name, create_date, sigma_algo_name, answer_compared_to_nlp
#from data.all_analyzed
#where aidoc_site='utsw'
#and analysis_finish_time_utc between '2022-03-01' and '2022-03-28''
original = pd.read_csv('UTSW-March22.csv') # to compare later with df

df = pd.read_csv('UTSW-March22.csv')#all changes will apply on df


# In[2]:




#creating 'AI_Algorithm' with the names of algorithms that Shuli preferes.

conitions_Algo = [
    (df['sigma_algo_name'] == 'hyperdense'), 
    (df['sigma_algo_name'] == 'pe'),
    (df['sigma_algo_name'] == 'ipe'),
    (df['sigma_algo_name'] == 'hypodense')

    ]
values_Algo = ['ICH','PE', 'iPE', 'CSF']
df['AI_Algorithm'] = np.select(conitions_Algo, values_Algo)




# In[3]:


#creating 'results' variable, which holds the number of TP,FN,FP,TN per each algo
results=df.groupby(['AI_Algorithm', 'answer_compared_to_nlp']).size().reset_index().pivot(columns='AI_Algorithm', index='answer_compared_to_nlp', values=0) 


#creating 'df_before' variable, which holds the values of metrics of sens', spec' and PPV, per each algo. 
pathology = ["ICH", "PE", "iPE","CSF" ]
data = []
for p in pathology:
    sens  = round(results[p]['TP']/(results[p]['TP']+results[p]['FN'])*100,1)
    #spec = round(results[p]['TN']/(results[p]['TN']+results[p]['FP'])*100,1)
    #ppv= round(results[p]['TP']/(results[p]['TP']+results[p]['FP'])*100,1)

    data.append([sens])

df_before = pd.DataFrame(data, columns=['sensitivity before our change'])
df_before = df_before.rename(index={0: 'ICH',1: 'PE',2: 'iPE',3: 'CSF'})  


# In[4]:


#######The core of the model#######
AI_est_sens ={"ICH":0.8,"PE":0.9,"iPE":0.63,"CSF":0.75} #these numbers can change of course
Target_AI={"ICH":0.92,"PE":0.9,"iPE":0.8,"CSF":0.85} 

FN_before={} # The initial number of FNs in each algo 
TP_before={} # Number of TPs in each algo
FN_after={} # Number of FNs after our change in each algo 
AI_FN_after={} # Number of FNs by AI after our change in each algo
NLP_FP_after={} # Number of NLP FP after our change in each algo
FN_per_omit={} # % of FNS to drop in each algo 
sens_before_site_clean ={}# sensitivity in each algo before site cleans discrepancies. This is sanity check and should be equal to df_after var which calculates it upon the table 'results after (external)'
sens_after_site_clean={} #sensitivity in each algo after site cleans discrepancies 
for i in pathology:
    FN_before[i]= results[i]["FN"]
    TP_before[i]= results[i]["TP"]
    FN_after[i]=  round(FN_before[i]*(1/Target_AI[i]-1)/(1/AI_est_sens[i]-1),1)
    AI_FN_after[i]= round(FN_after[i]*(1-AI_est_sens[i]),1)
    NLP_FP_after[i]= round(FN_after[i]-AI_FN_after[i],1)
    FN_per_omit[i]=round((100*(FN_before[i]-FN_after[i])/FN_before[i]),0)
    sens_before_site_clean[i]=round(100*(TP_before[i]/(TP_before[i]+AI_FN_after[i]+NLP_FP_after[i])),1)
    sens_after_site_clean[i]=round(100*(TP_before[i]/(TP_before[i]+AI_FN_after[i])),1)


# In[5]:


#Changing randomly of 'FN' and turning them to 'TN'. 
#The percentage of FNs per each algo that is changed to 'TN' is determined by the variable ending with FN_per_omit for each algo.

for i in pathology:
    name=   f'dfupdate_{i}'
    name = df[(df['answer_compared_to_nlp']=="FN") & (df['AI_Algorithm']==i)].sample(frac=FN_per_omit[i]/100)
    name.answer_compared_to_nlp="TN"
    df.update(name)
    
   
 # creating two variables - 'AI result' (based on answer compared to NLP) 
# 'NLP result' (based on answer compared to NLP)
conitions_AI = [
    (df['answer_compared_to_nlp'] == 'TP'), 
    (df['answer_compared_to_nlp'] == 'FN'),
    (df['answer_compared_to_nlp'] == 'FP'),
    (df['answer_compared_to_nlp'] == 'TN')
    ]
values_AI = ['positive','negative', 'positive', 'negative']
df['AI result'] = np.select(conitions_AI, values_AI)


conitions_NLP = [
    (df['answer_compared_to_nlp'] == 'TP'), 
    (df['answer_compared_to_nlp'] == 'FN'),
    (df['answer_compared_to_nlp'] == 'FP'),
    (df['answer_compared_to_nlp'] == 'TN')
    ]
values_NLP = ['positive','positive', 'negative', 'negative']
df['NLP result'] = np.select(conitions_NLP, values_NLP)


#cleaning of the model
#omitting empty values in an 'answer_compared_to_nlp'.
df = df[df.answer_compared_to_nlp.notnull()]

# omitting results of other algos which are not the 4 algos we are interested in - ICH, PE, iPE, CSF
df = df[df.AI_Algorithm!='0']
 


# In[6]:


# creating 'df_combined' dataframe, which holds the values of 1.df_before - which is sensitivity, per each algo, before the change ('FN' to 'TN'), 2. df_after - sensitivity after the change, and 3. disc - sensitivity after cleaning discrepancies 4. prod - product request - should be similar to disc variable 

results=df.groupby(['AI_Algorithm', 'answer_compared_to_nlp']).size().reset_index().pivot(columns='AI_Algorithm', index='answer_compared_to_nlp', values=0)

pathology = ["ICH", "PE", "iPE","CSF" ]
data = []
for p in pathology:
    sens  = round(results[p]['TP']/(results[p]['TP']+results[p]['FN'])*100,1)
    #spec = round(results[p]['TN']/(results[p]['TN']+results[p]['FP'])*100,1)
    #ppv= round(results[p]['TP']/(results[p]['TP']+results[p]['FP'])*100,1)

    
    data.append([sens])

df_after = pd.DataFrame(data, columns=['sensitivity after change (estimated)'])
df_after = df_after.rename(index={0: 'ICH',1: 'PE',2: 'iPE',3: 'CSF'} )  
df_combined=df_before
df_combined = df_combined.join(df_after["sensitivity after change (estimated)"])
disc=pd.DataFrame.from_dict(sens_after_site_clean,orient ='index')
disc = disc.rename(columns={0: 'sensitivity after change and after site cleans discrepancies (estimated)'})
df_combined = df_combined.join(disc["sensitivity after change and after site cleans discrepancies (estimated)"])

prod = pd.DataFrame([92.0,90.0,80.0,85.0], columns=['Production request'])
prod = prod.rename(index={0: 'ICH',1: 'PE',2: 'iPE',3: 'CSF'} ) 
df_combined = df_combined.join(prod["Production request"])


# In[7]:


# Organizing the results ('df') before exporting to excel
df=df.rename(columns = {'accession' : 'Accession number ','AI_Algorithm' : 'AI algorithm', 'create_date' : 'Date created', 'institution_name':'Institution name' })
df=df.sort_values(by='Date created')
df= df.drop(['sigma_algo_name'], axis=1)


# In[8]:


# determining the name of the file
file_name = 'utsw_output.xlsx'
  
# saving the excel

with pd.ExcelWriter('utsw_output.xlsx') as writer:  
    original.to_excel(writer, sheet_name='original excel by query')
    df_combined.to_excel(writer, sheet_name='sensitivity in different phases')
    df.to_excel(writer, sheet_name='results after (external)')


# In[ ]:




