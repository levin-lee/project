import pandas as pd
import time
import pickle as pk


class Filter:
    def __init__(self, data):
        self.data = data

    def main(self):
        data = self.data.groupby(['ACC_NBR']).apply(
            lambda x: x.sort_values(by=['CONTENT_NAME'], ascending=False).set_index(['CONTENT_NAME']).T.to_dict('int')[
                'SCORE']).reset_index().rename(columns={0: 'ITEM'})
        return data.set_index('ACC_NBR').T.to_dict('int')['ITEM']


if __name__ == '__main__':
    start = time.time()

    file = pd.ExcelFile('target.xlsx')
    datas = pd.read_excel(file, sheet_name='Sheet1', encoding='utf-8')
    result = Filter(datas).main()
    with open('filter.txt', 'wb') as f:
        pk.dump(result, f)

    end = time.time()
    print(end - start)
