import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from skfeature.function.information_theoretical_based import FCBF
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

label_name = ['AccuTrack' 'Adrd' 'Adsms' 'Ansca' 'Antares' 'Anti' 'Anudow' 'Arspam'
 'BaseBridge' 'BeanBot' 'Benign' 'Bgserv' 'Biige' 'Booster' 'Boxer'
 'CellShark' 'Ceshark' 'Coogos' 'Copycat' 'Cosha' 'CrWind' 'Dabom'
 'Dogowar' 'DroidDream' 'DroidKungFu' 'DroidRooter' 'DroidSheep'
 'EICAR-Test-File' 'EWalls' 'ExploitLinuxLotoor' 'FaceNiff' 'FakeDoc'
 'FakeInstaller' 'FakeRun' 'FakeTimer' 'Fakelogo' 'Fakengry' 'Fakeview'
 'FarMap' 'Fatakr' 'Fauxcopy' 'Fidall' 'FinSpy' 'Fjcon' 'Flexispy'
 'FoCobers' 'Fujacks' 'GGtrack' 'Gamex' 'Gapev' 'Gappusin' 'Gasms'
 'Geinimi' 'Generic' 'GinMaster' 'GlodEagl' 'Glodream' 'Gmuse' 'Hamob'
 'Hispo' 'Iconosys' 'Imlog' 'JS/Exploit-DynSrc' 'JSmsHider' 'Jifake'
 'Kidlogger' 'Kiser' 'Kmin' 'Koomer' 'Ksapp' 'Lemon' 'LifeMon' 'Lypro'
 'MMarketPay' 'MTracker' 'Mania' 'Maxit' 'MobileTx' 'Mobilespy'
 'Mobinauten' 'Mobsquz' 'Nandrobox' 'Nickspy' 'NickyRCP' 'Nyleaker'
 'Opfake' 'PJApps' 'PdaSpy' 'Penetho' 'Pirater' 'Pirates' 'Placms'
 'Plankton' 'Proreso' 'QPlus' 'RATC' 'Raden' 'RootSmart' 'SMSBomber'
 'SMSSend' 'SMSZombie' 'SMSreg' 'SafeKidZone' 'Saiva' 'Sakezon' 'Sdisp'
 'SeaWeth' 'SendPay' 'SerBG' 'SheriDroid' 'SmsSpy' 'SmsWatcher' 'Smspacem'
 'Spy.ImLog' 'SpyBubble' 'SpyHasb' 'SpyMob' 'SpyPhone' 'Spyoo' 'Spyset'
 'Stealer' 'Stealthcell' 'Steek' 'Tesbo' 'TheftAware' 'TigerBot'
 'Trackplus' 'TrojanSMS.Denofow' 'TrojanSMS.Hippo' 'TrojanSMS.Stealer'
 'Typstu' 'UpdtKiller' 'Updtbot' 'Vdloader' 'Vidro' 'Whapsni' 'Xsider'
 'YcChar' 'Yzhc' 'Zitmo' 'Zsone']

label_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]

le = LabelEncoder()


dataset  = pd.read_csv("dataset.csv", dtype='str')
#dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))
temp = dataset['label']
dataset = dataset.values
#Y = dataset[:,dataset.shape[1]-1]

Y = []
for i in range(len(temp)):
    if temp[i] == 'Benign':
        Y.append(0)
    else:
        Y.append(1)
Y = np.asarray(Y)
X = []
k = 0
with open("dataset.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        if k > 0:
            words = ''
            arr = line.split(",")
            for i in range(len(arr)-1):
                words+=arr[i]+" "
            X.append(words)
        k = k + 1    
X = np.asarray(X)
print(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(X)
print(Y)



tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), decode_error='replace')
tfidf = tfidf_vectorizer.fit_transform(X).toarray()        
df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
print(str(df))
print(df.shape)
df1 = df.values
X = df1[:, 0:df1.shape[1]]
X = np.asarray(X)
print(X)

idx = FCBF.fcbf(X,Y, n_selected_features=60)
features = X[:, idx[0:60]]
print(features)
print(features.shape)
print(Y.shape)

X = X[0:200]
Y = Y[0:200]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

cls = CatBoostClassifier(iterations=50, learning_rate=0.1, custom_loss=['AUC', 'Accuracy'])
cls.fit(X,Y)
predict = cls.predict(X_test)
predict = predict.flatten()
y_test = y_test.flatten()
pred = []
for i in range(len(predict)):
    pred.append(int(predict[i]))
predict = np.asarray(pred)
print(predict)
print(y_test)
a = accuracy_score(y_test,predict)*100
print(a)
p = precision_score(y_test, predict,average='macro') * 100
r = recall_score(y_test, predict,average='macro') * 100
f = f1_score(y_test, predict,average='macro') * 100

print(p)
print(r)
print(f)








