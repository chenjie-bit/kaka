import pickle
import csv 


if __name__=="__main__":
    with open("final_result.pkl","rb")as f:
        data=pickle.load(f)
    print(data)
    header = ['method','8','18','14','15','overall','reject']
    da = []
    for one in data.keys():
        precision = data[one]['precision']
        overall = data[one]['overall']
        reject = data[one]['reject']
        data_lst = ['8','18','14','15']
        list_keys = [k for k,v in precision.items()]
        for i in data_lst:
            if i not in list_keys:
                precision[i] = 0
        #print(precision)
        data_8 = precision['8']
        data_18 = precision['18']
        data_14 = precision['14']
        data_15 = precision['15']
        #print(type(data_8))
        add = [data_8,data_18,data_14,data_15, overall, reject]
        fin = [one] + add
        da.append(fin)

    file_name = '结果/'+'result' + '.csv'
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(da)
