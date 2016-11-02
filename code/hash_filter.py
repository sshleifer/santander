from __future__ import print_function
import datetime
from collections import defaultdict
import numpy as np
import operator
import pandas as pd

from code.constants import TRAIN_PATH, TEST_PATH, product_names, hash_cols


def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def is_valid_submission(path):
    df = pd.read_csv(path)
    df.columns.tolist() == ['ncodpers', 'added_products']
    df.shape[0] == 929616


def reverse_sort(product_counts):
    return sorted(product_counts.items(), key=operator.itemgetter(1), reverse=True)


def get_hash(arr):
    '''Tuple of demographic variables to use for group'''
    (fecha_dato, client, ind_empleado, pais_residencia, sexo, age,
        fecha_alta, ind_nuevo, antiguedad,
        indrel, ult_fec_cli_1t, indrel_1mes,
        tiprel_1mes, indresi, indext,
        conyuemp, canal_entrada, indfall,
        tipodom, cod_prov, nomprov,
        ind_actividad_cliente, renta, segmento) = arr[:24]
    return (pais_residencia, sexo, age, ind_nuevo, segmento,
            ind_empleado, ind_actividad_cliente, indresi)


class HashFilter(object):

    last_date = '2016-05-28'

    def __init__(self, train_path=TRAIN_PATH, test_path=TEST_PATH):
        print('Preparing arrays...')
        self.customers = dict()  # customerss we've seen
        self.hash_to_product = defaultdict(lambda: defaultdict(int))
        self.product_counts = defaultdict(int)
        self.test_path = test_path

        # Validation variables
        self.customers_valid = dict()
        self.hash_to_product_valid = defaultdict(lambda: defaultdict(int))
        self.product_counts_valid = defaultdict(int)
        self.validation_set = []

        self.map7 = 0.0

        with open(train_path) as f:
            self.train(f)
        self.validation()
        # self.generate_submission(test_path)

    def train(self, f):
        n_lines = 0
        f.readline()
        while 1:
            line = f.readline()[:-1]
            n_lines += 1

            if line == '':
                break

            tmp1 = line.split("\"")
            arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
            arr = [a.strip() for a in arr]
            assert len(arr) == 48, 'Error: !!!!{}'.format(line)
            client = arr[1]
            hash = get_hash(arr)
            part = arr[24:]

            # Normal part

            for i in range(24):
                if part[i] == '1':
                    if client in self.customers:
                        if self.customers[client][i] == '0':
                            self.hash_to_product[hash][i] += 1
                            self.product_counts[i] += 1
                    else:
                        self.hash_to_product[hash][i] += 1
                        self.product_counts[i] += 1
            self.customers[client] = part

            # Validation part
            if arr[0] == self.last_date:
                self.validation_set.append(arr)
            else:
                for i in range(24):
                    if part[i] == '1':
                        if client in self.customers_valid:
                            if self.customers_valid[client][i] == '0':
                                self.hash_to_product_valid[hash][i] += 1
                                self.product_counts_valid[i] += 1
                        else:
                            self.hash_to_product_valid[hash][i] += 1
                            self.product_counts_valid[i] += 1
                self.customers_valid[client] = part

            if n_lines % 1000000 == 0:
                print('Process {} lines ...'.format(n_lines))
                # break
        print('Sort best arrays...')
        print('Hashes num: ', len(self.hash_to_product))
        print('Valid part: ', len(self.validation_set))

        self.hash_to_product = {b: reverse_sort(self.hash_to_product[b]) for b, v in self.hash_to_product.items()}
        self.product_counts = reverse_sort(self.product_counts)

        # Valid
        print(self.hash_to_product_valid)
        self.hash_to_product_valid = {b: reverse_sort(self.hash_to_product_valid[b]) for b in self.hash_to_product_valid}
        self.product_counts_valid = reverse_sort(self.product_counts_valid)

    def validation(self):
        for arr1 in self.validation_set:
            client = arr1[1]
            hash = get_hash(arr1)
            predicted = self.predict_valid(hash, client)

            # Find real
            real = []
            arr2 = arr1[24:]

            for i in range(len(arr2)):
                if arr2[i] == '1':
                    if client in self.customers_valid:
                        if self.customers_valid[client][i] == '0':
                            real.append(i)
                    else:
                        real.append(i)
            self.map7 += apk(real, predicted)

        self.map7 /= max(len(self.validation_set), 1)
        print('Predicted score: {}'.format(self.map7))

    def predict_valid(self, hash, client):
        '''Find 7 products from a users group that they havent used, if not enough in group use overall counts'''
        product_recommendations = self.hash_to_product_valid.get(hash, []) + self.product_counts_valid
        used_products = self.customers_valid.get(client, [])
        predicted = []
        for product, _ in product_recommendations:
            if len(predicted) == 7:
                return predicted
            elif product in predicted or product in used_products:
                continue
            else:
                predicted.append(product)
        return predicted

    def predict(self, hash, client):
        '''Find 7 products from a users group that they havent used, if not enough in group use overall counts'''
        product_recommendations = self.hash_to_product.get(hash, []) + self.product_counts
        used_products = self.customers.get(client, [])
        predicted = []
        for product, _ in product_recommendations:
            if len(predicted) == 7:
                return predicted
            elif product in predicted or product in used_products:
                continue
            else:
                predicted.append(product)
        return predicted

    def generate_submission(self, test_path=None):
        '''write a csv file with predictions based on data at <test_path>'''
        test_path = test_path or self.test_path
        print('Generate submission...')
        tstamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        sub_file = 'submissions/submission_{}_{}.csv'.format(self.map7, tstamp)
        out = open(sub_file, "w")
        f = open(test_path)
        f.readline()
        n_lines = 0
        count_empty = 0
        out.write("client,added_products\n")

        while 1:
            line = f.readline()[:-1]
            n_lines += 1

            if line == '':
                break

            tmp1 = line.split("\"")
            arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
            arr = [a.strip() for a in arr]
            client = arr[1]

            out.write(client + ',')

            hash = get_hash(arr)
            predicted = self.predict(hash, client)
            for p in predicted:
                out.write(product_names[p] + ' ')

            if n_lines % 1000000 == 0:
                print('Read {} lines ...'.format(n_lines))
                # break

        out.write("\n")

        print('Total cases:', str(n_lines))
        print('Empty cases:', str(count_empty))
        out.close()
        f.close()


def take_7(cluster_recs, overall_recs):
    chosen = cluster_recs.index
    need = 7 - len(chosen)
    return chosen.append(overall_recs.drop(chosen).index[:need])


class SamFilter(object):
    '''Collaborative Filtering based on hash_cols'''
    id_col = 'ncodpers'

    def __init__(self, df_train, hash_cols=hash_cols):
        '''Record customer usage, group usage, '''
        self.customer_usage = df_train.groupby(self.id_col)[product_names].sum()
        self.overall_usage = df_train[product_names].sum().sort_values(ascending=False)
        self.cluster_usage = df_train.groupby(hash_cols)[product_names].sum()
        clist = {}
        for k, v in self.cluster_usage.iterrows():
            recs = v[v > 0].sort_values(ascending=False).index
            clist[k] = recs.append(self.overall_usage.index.drop(recs))
        self.clist = clist

    def _predict(self, row):
        '''Find non-used products that were popular in the cluster'''
        id, cluster_id = row[self.id_col], tuple(row[hash_cols].values)
        start = self.clist.get(cluster_id, self.overall_usage.index)
        if id in self.customer_usage.index:
            used_products = self.customer_usage.loc[row[self.id_col]].loc[lambda x: x > 0].index
        else:
            return start[:7]
            used_products = pd.Index([])
        return start.drop(used_products)[:7]

    def predict_each_row(self, test_data):
        '''Make a prediction for each row in test_data'''
        return (pd.Series({v[self.id_col]: ' '.join(self._predict(v)) for k, v in test_data.iterrows()})
                .rename_axis('ncodpers').rename('added_products'))

    def _apk(self, preds, truth, id):
        pred_list = preds.split()
        used_products = self.customer_usage.loc[id].loc[lambda x: x > 0].index

        truth = truth.drop(used_products, errors='ignore')
        return apk(pred_list, truth.tolist())

    def make_validation_set(self, df):
        truth = (df.groupby('ncodpers')[product_names].sum())
        return pd.Series({code: row[row > 0].index for code, row in truth.iterrows()})

    def score(self, pred_df, truth_df):
        map7 = 0
        jnd = pred_df.to_frame(name='added_products').join(truth_df).dropna()
        for i, v in jnd.iterrows():
            map7 += self._apk(v['added_products'], v['truth'], i)
        map7 /= max(len(pred_df), 1)
        return map7

if __name__ == '__main__':
    hf = SamFilter()
    hf.generate_submission()
