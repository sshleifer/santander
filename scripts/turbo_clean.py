from __future__ import print_function

import datetime
import os
from collections import defaultdict
import operator

product_names = [
    'ind_ahor_fin_ult1',
    'ind_aval_fin_ult1',
    'ind_cco_fin_ult1',
    'ind_cder_fin_ult1',
    'ind_cno_fin_ult1',
    'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1',
    'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1',
    'ind_deme_fin_ult1',
    'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1',
    'ind_fond_fin_ult1',
    'ind_hip_fin_ult1',
    'ind_plan_fin_ult1',
    'ind_pres_fin_ult1',
    'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1',
    'ind_valo_fin_ult1',
    'ind_viv_fin_ult1',
    'ind_nomina_ult1',
    'ind_nom_pens_ult1',
    'ind_recibo_ult1'
]

TRAIN_PATH = '/Users/shleifer/flow/santander/inputs/train_ver2.csv'
TEST_PATH = '/Users/shleifer/flow/santander/inputs/test_ver2.csv'


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


def reverse_sort(overallbest):
    return sorted(overallbest.items(), key=operator.itemgetter(1), reverse=True)


def get_hash(arr, type = 0):
    '''Tuple of demographic variables to use for group'''
    (fecha_dato, client, ind_empleado,
    pais_residencia, sexo, age,
    fecha_alta, ind_nuevo, antiguedad,
    indrel, ult_fec_cli_1t, indrel_1mes,
    tiprel_1mes, indresi, indext,
    conyuemp, canal_entrada, indfall,
    tipodom, cod_prov, nomprov,
    ind_actividad_cliente, renta, segmento) = arr[:24]

    if type == 0:
        return (pais_residencia, sexo, age, ind_nuevo, segmento, ind_empleado, ind_actividad_cliente, indresi)
    else:
        return (sexo, age, segmento)

class HashFilter(object):

    def __init__(self, train_path=TRAIN_PATH, test_path=TEST_PATH):
        print('Preparing arrays...')
        self.customers = dict()  # customerss we've seen
        self.best = defaultdict(lambda: defaultdict(int))
        self.overallbest = defaultdict(int)

        # Validation variables
        self.customers_valid = dict()
        self.best_valid = defaultdict(lambda: defaultdict(int))
        self.overallbest_valid = defaultdict(int)
        self.valid_part = []

        self.map7 = 0.0

        with open(train_path) as f:
            self.read_and_set(f)
        self.validation()
        #self.generate_submission(test_path)

    def read_and_set(self, f):
        total = 0
        f.readline()
        while 1:
            line = f.readline()[:-1]
            total += 1

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
                            self.best[hash][i] += 1
                            self.overallbest[i] += 1
                    else:
                        self.best[hash][i] += 1
                        self.overallbest[i] += 1
            self.customers[client] = part

            # Validation part
            if arr[0] != '2016-05-28':
                for i in range(24):
                    if part[i] == '1':
                        if client in self.customers_valid:
                            if self.customers_valid[client][i] == '0':
                                self.best_valid[hash][i] += 1
                                self.overallbest_valid[i] += 1
                        else:
                            self.best_valid[hash][i] += 1
                            self.overallbest_valid[i] += 1
                self.customers_valid[client] = part
            else:
                self.valid_part.append(arr)

            if total % 1000000 == 0:
                print('Process {} lines ...'.format(total))
                # break
        print('Sort best arrays...')
        print('Hashes num: ', len(self.best))
        print('Valid part: ', len(self.valid_part))

        self.best = {b: reverse_sort(self.best[b]) for b,v in self.best.items()}
        self.overallbest = reverse_sort(self.overallbest)

        # Valid
        print(self.best_valid)
        self.best_valid = {b: reverse_sort(self.best_valid[b]) for b in self.best_valid}
        self.overallbest_valid = reverse_sort(self.overallbest_valid)


    def validation(self):
        for arr1 in self.valid_part:
            client = arr1[1]
            hash = get_hash(arr1)

            if hash in self.best_valid:
                product_array = self.best_valid[hash]
            else:
                product_array = self.overallbest_valid

            predicted = []
            for a in product_array:
                if client in self.customers_valid:
                    if self.customers_valid[client][a[0]] == '1':
                        continue
                if a[0] not in predicted:
                    predicted.append(a[0])
                    if len(predicted) == 7:
                        break

            if len(predicted) < 7:
                for a in self.overallbest_valid:
                    # If user is not new
                    if client in self.customers_valid:
                        if self.customers_valid[client][a[0]] == '1':
                            continue
                    if a[0] not in predicted:
                        predicted.append(a[0])
                        if len(predicted) == 7:
                            break

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

        self.map7 /= max(len(self.valid_part), 1)
        print('Predicted score: {}'.format(self.map7))


    def generate_submission(self, test_path=None):
        test_path = test_path or self.test_path
        print('Generate submission...')
        sub_file = os.path.join('submission_' + str(self.map7) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
        out = open(sub_file, "w")
        f = open(test_path)
        f.readline()
        total = 0
        count_empty = 0
        out.write("client,added_products\n")

        while 1:
            line = f.readline()[:-1]
            total += 1

            if line == '':
                break

            tmp1 = line.split("\"")
            arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
            arr = [a.strip() for a in arr]
            client = arr[1]
            hash = get_hash(arr)

            out.write(client + ',')
            # If class exists output for class
            if hash in self.best:
                arr = self.best[hash]
            else:
                arr = self.overallbest
                count_empty += 1

            predicted = []
            for a in arr:
                # If user is not new
                if client in self.customers:
                    if self.customers[client][a[0]] == '1':
                        continue
                if a[0] not in predicted:
                    predicted.append(a[0])
                    if len(predicted) == 7:
                        break
            if len(predicted) < 7:
                for a in self.overallbest:
                    # If user is not new
                    if client in self.customers:
                        if self.customers[client][a[0]] == '1':
                            continue
                    if a[0] not in predicted:
                        predicted.append(a[0])
                        if len(predicted) == 7:
                            break

            for p in predicted:
                out.write(product_names[p] + ' ')

            if total % 1000000 == 0:
                print('Read {} lines ...'.format(total))
                # break

        out.write("\n")

        print('Total cases:', str(total))
        print('Empty cases:', str(count_empty))
        out.close()
        f.close()
