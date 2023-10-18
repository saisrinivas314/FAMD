import pandas as pd
import numpy as np
import os

names = pd.read_csv('Dataset/sha256_family.csv')
names = names.values
features_set = {
    "feature": 1,
    "permission": 2,
    "activity": 3,
    "service_receiver": 3,
    "provider": 3,
    "service": 3,
    "intent": 4,
    "api_call": 5,
    "real_permission": 6,
    "call": 7,
    "url": 8
}

def extract_features(file_content):
    features_occurrences = {el: 0 for el in range(1, 9)}
    featuresList = []
    for line in file_content:
        line_set = line.split('::')[0]
        temp_var = features_set.get(line_set, None)
        if(temp_var is not None):
            features_occurrences[temp_var] += 1
            featuresList.append(line_set)
    features = []
    for i in range(1, 9):
        features.append(features_occurrences[i])
    return featuresList

def getMalware(name):
    malware_type = 'Benign'
    for i in range(len(names)):
        file = names[i,0]
        malware = names[i,1]
        if file == name:
            malware_type = malware
            break
    return malware_type

path = 'Dataset/drebin'
X = ''
count = 0
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        file = root+"/"+directory[j]
        malware = getMalware(directory[j])
        if malware != 'Benign':
            with open(file, 'r') as fileData:
                file_content = fileData.read().splitlines()
                sample = extract_features(file_content)
                data = ''
                if len(sample) > 30:
                    for i in range(0,30):
                        data+=sample[i]+","
                    data+=malware    
                    X+=data+"\n"    
                    print(file+" "+malware)
                    count = count + 1
            fileData.close()



count1 = 0
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        file = root+"/"+directory[j]
        malware = getMalware(directory[j])
        if count1 > count:
            break
        if malware == 'Benign':
            with open(file, 'r') as fileData:
                file_content = fileData.read().splitlines()
                sample = extract_features(file_content)
                data = ''
                if len(sample) > 30:
                    for i in range(0,30):
                        data+=sample[i]+","
                    data+=malware    
                    X+=data+"\n"    
                    print(file+" "+malware+" "+str(count)+" "+str(count1))
                    count1 = count1 + 1
            fileData.close()

f = open("dataset.csv", "w")
f.write(X)
f.close()

            
