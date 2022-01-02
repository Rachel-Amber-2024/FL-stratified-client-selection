import numpy as np
import pandas as pd
import pickle
from math import floor
from numpy.random import choice


def get_num_cnt(args, list_dls_train):
    labels = []
    for dl in list_dls_train:
        labels_temp = []
        for data in dl:
            labels_temp += data[1].tolist()
        labels.append(labels_temp)

    num_cnt = []
    for label_ in labels:
        cnt = []
        total = len(label_)
        for num in range(10):
            cnt.append(label_.count(num))
        num_cnt.append(cnt)

    with open(f"dataset/data_partition_result/{args.dataset}_{args.partition}.pkl", "wb") as output:
        pickle.dump(num_cnt, output)
    print("Data partition result successfully saved!")

    # print num_cnt
    print("num_cnt table: ")
    num_cnt_table = pd.DataFrame(num_cnt, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # print 100 rows completely
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(num_cnt_table)


def stratify_clients(args):
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    partition_result_path = f"dataset/data_partition_result/{args.dataset}_{args.partition}.pkl"
    print("@@@ Start reading data_partition_result file：", partition_result_path, " @@@")

    m_data = []
    data = []

    with open(partition_result_path, 'rb') as f:
        while True:
            try:
                row_data = pickle.load(f)
                for m in row_data:
                    m_data.append(m)
            except EOFError:
                break

    # zero-mean normalizationof data
    for d in m_data:
        da = []
        avg = np.mean(d)
        std = np.std(d, ddof=1)
        for i in d:
            da.append((i - avg) / std)
        data.append(da)
    data = np.array(data)

    # The principal components analysis(PCA) of data dimension reduction
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

    # Prototype Based Clustering: KMeans
    model = KMeans(n_clusters=args.strata_num)
    model.fit(data)
    pred_y = model.predict(data)
    pred_y= list(pred_y)
    result = []
    # put indexes into result
    for num in range(args.strata_num):
        one_type = []
        for index, value in enumerate(pred_y):
            if value==num:
                one_type.append(index)
        result.append(one_type)
    print(result)
    save_path = f'dataset/stratify_result/{args.dataset}_{args.partition}.pkl'
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as output:
        pickle.dump(result, output)

    # print silhouette_score
    s_score = metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean')
    print("strata_num：", args.strata_num, " silhouette_score：", s_score, "\n")

    return result


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def sample_clients_without_allocation(chosen_p, choice_num):
    n_clients = len(chosen_p[0])
    strata_num = len(chosen_p)

    sampled_clients = np.zeros(len(chosen_p) * choice_num, dtype=int)

    for k in range(strata_num):
        c = choice(n_clients, choice_num, replace=False, p=chosen_p[k])
        for n_th, one_choice in enumerate(c):
            sampled_clients[k * choice_num + n_th] = int(one_choice)

    return sampled_clients


def sample_clients_with_allocation(chosen_p, allocation_number):
    n_clients = len(chosen_p[0])

    sampled_clients = []

    for i, n in enumerate(allocation_number):
        if n == 0:
            pass
        else:
            c = choice(n_clients, n, replace=False, p=chosen_p[i])
            for n_th, one_choice in enumerate(c):
                sampled_clients.append(int(one_choice))

    return sampled_clients


def cal_allocation_number(partition_result, stratify_result, sample_ratio):
    cohesion_list = []
    for row_strata in stratify_result:
        dist = np.zeros(len(row_strata))

        for j in range(len(row_strata)):
            for k in range(len(row_strata)):
                if k == j:
                    pass
                else:
                    dist[j] += np.sqrt(np.sum(np.square(np.array(partition_result[j]) - np.array(partition_result[k]))))

        dist /= len(row_strata)

        cohesion_list.append(dist)

    avg_cohesion = np.zeros(len(cohesion_list))

    for i, strata_cohesion in enumerate(cohesion_list):
        avg_cohesion[i] = sum(strata_cohesion) / len(strata_cohesion)

    allocation_number = np.zeros(len(avg_cohesion))
    for i, strata_coh in enumerate(avg_cohesion):
        weight = strata_coh / sum(avg_cohesion)
        allocation_number[i] = floor(sample_ratio * 100 * weight)

    allocation_number = allocation_number.astype(int)

    zero_num = (allocation_number == 0).sum()
    i = 0
    while np.sum(allocation_number) < sample_ratio * 100:
        if allocation_number[i] == 0:
            allocation_number[i] += max(1, int(round((sample_ratio * 100 - np.sum(allocation_number)) / zero_num)))
        i += 1

    return allocation_number
