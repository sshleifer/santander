from __future__ import print_function
import datetime
import numpy as np
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
SECOND_TO_LAST_DATE = '2016-04-28'
nprods = 24


class SamFilter(object):
    '''Collaborative Filtering based on hash_cols'''
    id_col = 'ncodpers'

    def __init__(self, df_train, hash_cols=hash_cols, min_cluster_size=2., max_train_date=None):
        '''Record customer usage, group usage, '''
        self.hash_cols = hash_cols
        self.customer_usage = (df_train[df_train.fecha_dato == df_train.fecha_dato.max()]
                               .groupby(self.id_col)[product_names].sum())
        self.overall_usage = df_train[product_names].sum().sort_values(ascending=False)
        gb = df_train.groupby(self.hash_cols)
        too_small = gb.size().loc[lambda x: x < min_cluster_size].index
        self.cluster_usage = gb[product_names].sum().drop(too_small)
        self.cldf = self.cluster_usage.reset_index().reset_index().rename(columns={'index': 'cluster'}).assign(cluster=lambda x: x.cluster.astype(str))

        self.turbo = pd.read_csv('submissions/submission_turbo.csv')
        self.fit(df_train, max_train_date=max_train_date)

    def fit(self, tr_df, max_train_date):
        if max_train_date is not None:
            tr_df = df[df.fecha_dato < max_train_date]
        self.usage_df = self._build_usage_df(tr_df)
        self.recommendations = self.filter_to_top7(self.usage_df, tr_df)

    def _build_usage_df(self, df_tr_last_day):
        hash_map = df_tr_last_day.set_index('ncodpers')[hash_cols]
        cl2 = hash_map.reset_index().merge(self.cldf, how='left').set_index('ncodpers')
        clmt = (cl2.set_index('cluster', append=True)[product_names]
                .rename_axis('product', 1).stack().to_frame(name='cluster_usage'))
        overall_usage_df = self.overall_usage.rename_axis('product').to_frame(name='overall_usage')
        return (
            self.customer_usage.rename_axis('product', 1).stack().to_frame(name='usage').reset_index()
            .merge(clmt.reset_index(), how='left')
            .merge(overall_usage_df.reset_index(), on='product', how='left')
        )

    def filter_to_top7(self, usage_df, df_tr_last_day):
        id = self.id_col
        ncods = df_tr_last_day[id].nunique()
        poss_df = df_tr_last_day.set_index(id)[product_names].rename_axis('product', 1).stack().to_frame(name='used_last').reset_index()
        scale_factor = usage_df.overall_usage.min() + .1
        usage_df['score'] = (usage_df.cluster_usage.fillna(0) * scale_factor) + usage_df.overall_usage
        sf2 = usage_df.score.max() + 1
        usage_df['score'] = usage_df['score'] - (sf2 * np.clip(usage_df.usage, 0, 1))
        assert usage_df.score.notnull().all()
        assert ncods * poss_df['product'].nunique() == len(poss_df)
        ucand = usage_df.sort_values([id, 'score'], ascending=False)
        indices = [x for x in range(len(ucand)) if x % nprods <= 6]
        return ucand.iloc[indices]

    def predict_each_row(self, test_data):
        '''Make a prediction for each row in test_data'''
        gb = self.recommendations.groupby('ncodpers')
        res = {}
        for k, v in tqdm(test_data.set_index(self.id_col).iterrows()):
            try:
                grp = gb.get_group(k)
                res[k] = ' '.join(grp['product'].values)
            except KeyError:
                res[k] = self.turbo.loc[k, 'added_products']
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

    def pos_success_score(self, df_valid):
        preds = self.predict_each_row(df_valid)
        pred_df = preds.apply(lambda x: x.split()).apply(pd.Series).rename_axis('product', 1).stack()
        p2 = pred_df.reset_index('product', drop=True).to_frame(name='product').assign(predicted=1).reset_index()
        ncods = p2.ncodpers.unique()
        return TA2[TA2.ncodpers.isin(ncods)].merge(p2, how='left').predicted.fillna(0).mean()

if __name__ == '__main__':
    df = pd.read_csv('inputs/my_train.csv', dtype=DTYPES)
    df_test = pd.read_pickle('inputs/test_df.pkl')
    self = SamFilter(df)
    tstamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    preds = self.predict_each_row(df_test)
    sub_file = 'submissions/sam_{}.csv'.format(tstamp)
    preds.to_csv(sub_file)
