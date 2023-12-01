import pandas as pd
from collections import Counter
import time
import pickle as pk


class Recommend:
    def __init__(self, data, rules, filters, TopN):
        self.data = data
        self.Rule = rules
        self.Filter = filters
        self.TopN = TopN

    # ************************计算过程************************
    def main(self):
        # 根据用户分组计算推荐得分
        return self.data.groupby(['ACC_NBR']).apply(self.Recommendation).reset_index().rename(columns={0: 'RECOMMEND'})

    # *****************余弦距离判断，TopN推荐*****************
    def Recommendation(self, data):
        self.index = 0
        self.result = {}

        # 0  映射辅助函数
        def transform(data):
            print(data)
            if self.index == 0:
                # 0.0 groupby会默认对第一个元素计算两边，消除这个影响
                self.index += 1
            else:
                # 0.1 获取内容对应的规则字典
                rule_dict = self.Rule[data['CONTENT_NAME'].values[0]]
                # 0.2 将规则字典的value值*内容评分
                result_update = dict(map(lambda x: (x[0], x[1] * data['SCORE'].values[0]), rule_dict.items()))
                print(result_update)
                # 0.3 将结果添加进结果字典，如果重复取最大value值
                r, r_update = Counter(self.result), Counter(result_update)
                self.result = dict(r | r_update)

        # 1  根据规则映射距离、评分加权结果
        data.groupby('CONTENT_NAME').apply(transform).reset_index().rename(columns={0: 'RECOMMEND'})
        # 2  根据过滤剔除历史记录
        filter_dict = self.Filter[data['ACC_NBR'].values[0]]
        self.result = dict(filter(lambda x: x[0] not in filter_dict.keys(), self.result.items()))
        # 3  排序返回TopN结果
        return dict(sorted(self.result.items(), key=lambda x: x[1], reverse=True)[0:self.TopN])


if __name__ == '__main__':
    start = time.time()

    TopN = 5

    file = pd.ExcelFile('target.xlsx')
    datas = pd.read_excel(file, sheet_name='Sheet1', encoding='utf-8')
    with open('rule.txt', 'rb') as f:
        rules = pk.load(f)
    with open('filter.txt', 'rb') as f:
        filters = pk.load(f)
    result = Recommend(datas, rules, filters, TopN).main()
    pd.set_option('display.max_columns', None)
    print(result)
    result.to_csv('result.csv', encoding='utf_8_sig', index=0)

    end = time.time()
    print(end - start)
