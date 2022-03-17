import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
xgb.set_config(verbosity=0)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('sapsiparisleri.csv', skiprows=1, delimiter=';', encoding="utf8")
df.columns = ["docentry", "docdate", "cardcode", "cardname", "itemcode", "dscription", "miktar", "shiptocode", "address"]

df['carditem'] = df.cardcode + ', ' + df.itemcode
# df['card_item'] = df['carditem'].str.split(',')
df['docdate'] = pd.to_datetime(df['docdate'])
musteri_son_satin_alim = df[(df.docdate < pd.Timestamp(2021,12,3)) & (df.docdate >= pd.Timestamp(2021,7,30))].reset_index(drop=True) # Tarihler => Yıl,Ay,Gün
musteri_siradaki_alim = df[(df.docdate < pd.Timestamp(2022,3,3)) & (df.docdate >= pd.Timestamp(2021,12,3))].reset_index(drop=True) # Tarihler => Yıl,Ay,Gün

musteri = pd.DataFrame(df['carditem'].unique())
musteri.columns = ['carditem']

siradaki_ilk_alim =musteri_siradaki_alim.groupby("carditem")["docdate"].min().reset_index()
siradaki_ilk_alim.columns = ['carditem', 'min_alim_tarihi']

son_alim = musteri_son_satin_alim.groupby("carditem")["docdate"].max().reset_index()
son_alim.columns = ['carditem','max_alim_tarihi']

alim_tarihleri = son_alim.merge(siradaki_ilk_alim, on='carditem', how='left')
alim_tarihleri['sonraki_alim_gunu'] = ((alim_tarihleri['min_alim_tarihi']) - (alim_tarihleri['max_alim_tarihi'])).dt.days
df_musteri = pd.DataFrame(musteri, columns= ["carditem"])
musteri = df_musteri.merge(alim_tarihleri[['carditem','sonraki_alim_gunu']], on='carditem', how='left')

gunluk_siparis = musteri_son_satin_alim[['carditem','docdate']]
gunluk_siparis['fatura_gunu'] = musteri_son_satin_alim['docdate']
gunluk_siparis = gunluk_siparis.sort_values(['carditem','docdate'])
gunluk_siparis = gunluk_siparis.drop_duplicates(subset=['carditem','docdate'],keep='first')

gunluk_siparis['oncekiFaturaTarihi'] = gunluk_siparis.groupby("carditem")['docdate'].shift(1)
gunluk_siparis['2_oceki_fatura_tarihi'] = gunluk_siparis.groupby("carditem")['docdate'].shift(2)
gunluk_siparis['3_onceki_fatura_tarihi'] = gunluk_siparis.groupby("carditem")['docdate'].shift(3)

gunluk_siparis['gun_farki'] = (gunluk_siparis['docdate'] - gunluk_siparis['oncekiFaturaTarihi']).dt.days
gunluk_siparis['gun_farki_2'] = (gunluk_siparis['docdate'] - gunluk_siparis['2_oceki_fatura_tarihi']).dt.days
gunluk_siparis['gun_farki_3'] = (gunluk_siparis['docdate'] - gunluk_siparis['3_onceki_fatura_tarihi']).dt.days

gun_farki = gunluk_siparis.groupby("carditem").agg({'gun_farki': ['mean','std']}).reset_index()
gun_farki.columns = ['carditem', 'gun_farki_ortalamasi','gun_farki_standart_sapmasi']

son_gunluk_siparis = gunluk_siparis.drop_duplicates(subset=['carditem'],keep='last')

son_gunluk_siparis = son_gunluk_siparis.dropna()
son_gunluk_siparis = son_gunluk_siparis.merge(gun_farki, on='carditem')
musteri = musteri.merge(son_gunluk_siparis[['carditem','gun_farki','gun_farki_2','gun_farki_3','gun_farki_ortalamasi','gun_farki_standart_sapmasi']], on='carditem')

sinif = musteri.copy()
sinif = pd.get_dummies(sinif)

sinif['sonraki_alim_gunu_araligi'] = 2
sinif.loc[sinif["sonraki_alim_gunu"]>20,['sonraki_alim_gunu_araligi']] = 1
sinif.loc[sinif["sonraki_alim_gunu"]>50,['sonraki_alim_gunu_araligi']] = 0

# corr = sinif[sinif.columns].corr()
# plt.figure(figsize = (30,20)) # plt.subplots = plt.figure
# sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f")
# plt.show()

sinif = sinif.drop('sonraki_alim_gunu',axis=1)
X, y = sinif.drop('sonraki_alim_gunu_araligi',axis=1), sinif["sonraki_alim_gunu_araligi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))

for ismi,model in models:
    kfold = KFold(n_splits=2, random_state=22, shuffle=True)
    cv_sonuc = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(ismi, cv_sonuc)

xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
print('egitim_setinde_XGB_siniflandiricinin_dogrulugu: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('test_setinde_XGB_siniflandiricinin_dogrulugu: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(), param_grid = param_test1, scoring='accuracy', n_jobs=-1, cv=2)
gsearch1.fit(X_train,y_train)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))

print(musteri)


# -------------------------------------------------------------------------------------------------------------------


# df['carditemdate'] = df.cardcode + ', ' + df.docdate
# musteri = pd.DataFrame(df['carditemdate'].unique())
# musteri.columns = ['carditemdate']



