import pandas as pd
data = pd.read_csv('golf-dataset.csv')
def counter(data,colname,label,target):
    temp = (data[colname] == label) & (data['Play Golf'] == target)
    return len(data[temp])
split_data = [80,70,60]
for i in split_data:
    prediksi = [] #menyimpan hasil prediksi nilai
    probabilitas = {0:{},1:{}} #menyimpan nilai probabilitas
    train_len = int((i*len(data))/100) #menghitung banyaknya nilai training
    #split training dan testing
    train_X = data.iloc[:train_len,:] #mengambil data sebanyak train_len
    test_X = data.iloc[train_len+1:,:-1] #mendapatkan nilai data test
    test_Y = data.iloc[train_len+1:,-1] #mengambil nilai class
    #data,colname,label,target
    count_yes = counter(train_X,'Play Golf','Yes','Yes') #total  class yes pada data
    count_no = counter(train_X, 'Play Golf','No','No') #total class no pada data
    prob_yes = count_yes/len(train_X) #nilai probabilitas pada class yes
    prob_no = count_no/len(train_X) #nilai probabilitas pada class no

    # print(count_yes)
    # print(count_no)
    #training model
    for j in train_X.columns[:-1]:
        # nilai probabilitas pada setiap atribut terhadap class
        probabilitas[1][j] = {}
        probabilitas[0][j] = {}
        for k in train_X[j].unique():
            count_k_yes = counter(train_X, j, k, 'Yes')
            count_k_no = counter(train_X, j, k, 'No')
            probabilitas[1][j][k] = count_k_yes/count_yes
            probabilitas[0][j][k] = count_k_no/count_no

    #test model
    for baris in range(0,len(test_X)):
        hasil_yes = prob_yes
        hasil_no = prob_no
        for kolom in test_X.columns:
            hasil_yes *= probabilitas[1][kolom][test_X[kolom].iloc[baris]]
            hasil_no  *= probabilitas[0][kolom][test_X[kolom].iloc[baris]]
        if hasil_yes > hasil_no:
            prediksi.append('Yes')
        else:
            prediksi.append('No')


    #confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    for j in range(0,len(prediksi)):
        if prediksi[j] == 'Yes':
            if test_Y.iloc[j] == 'Yes':
                tp+=1
            else:
                fp+=1
        else:
            if test_Y.iloc[j] == 'No':
                tn+=1
            else:
                fn+=1
    print('Akurasi dengan data training '+str(i)+ '% :' ,(tp+tn)/len(test_Y)*100,'%')
