from __future__ import print_function
import datetime
# import numpy as np
import pandas as pd

from code.constants import product_names, hash_cols
from code.helpers import apk
from tqdm import tqdm


DTYPES = {'age': str,
          'antiguedad': str,
          'conyuemp': str,
          'ind_actividad_cliente': str,
          'ind_empleado': str,
          'ind_nuevo': str,
          'indrel_1mes': str,
          'indresi': str,
          'pais_residencia': str,
          'segmento': str,
          'sexo': str}

TA2 = pd.read_pickle('inputs/valid_products.pkl')


class SamFilter(object):
    '''Collaborative Filtering based on hash_cols'''
    id_col = 'ncodpers'

    def __init__(self, df_train, hash_cols=hash_cols, min_cluster_size=2.):
        '''Record customer usage, group usage, '''
        self.hash_cols = hash_cols
        self.customer_usage = (df_train[df_train.fecha_dato == df_train.fecha_dato.max()]
                               .groupby(self.id_col)[product_names].sum())
        self.overall_usage = df_train[product_names].sum().sort_values(ascending=False)
        gb = df_train.groupby(self.hash_cols)
        too_small = gb.size().loc[lambda x: x < min_cluster_size].index
        self.cluster_usage = gb[product_names].sum().drop(too_small)
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
        res = {}
        for k, v in tqdm(test_data.iterrows()):
            res[v[self.id_col]] = ' '.join(self._predict(v))
        return (pd.Series(res).rename_axis('ncodpers').rename('added_products'))

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

    def pos_success_score(sf, df_valid):
        preds = sf.predict_each_row(df_valid)
        pred_df = preds.apply(lambda x: x.split()).apply(pd.Series).rename_axis('product', 1).stack()
        p2 = pred_df.reset_index('product', drop=True).to_frame(name='product').assign(predicted=1).reset_index()
        ncods = p2.ncodpers.unique()
        return TA2[TA2.ncodpers.isin(ncods)].merge(p2, how='left').predicted.fillna(0).mean()

if __name__ == '__main__':
    df = pd.read_csv('inputs/my_train.csv', dtype=DTYPES)
    df_test = pd.read_pickle('inputs/test_df.pkl')
    sf = SamFilter(df)
    tstamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    preds = sf.predict_each_row(df_test)
    sub_file = 'submissions/sam_{}.csv'.format(tstamp)
    preds.to_csv(sub_file)
