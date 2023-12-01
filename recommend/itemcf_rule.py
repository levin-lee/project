import pandas as pd
import numpy as np
import itertools as it
import time
import pickle as pk


#   让列表可以哈希，做字典key
class MyList(list):
    def __hash__(self):
        return hash(self[0])


#   规则过滤类
class Rule:
    def __init__(self, data):
        self.data = data
        self.user_item = None
        self.Rule = {}

    # ************************计算过程************************
    def main(self):
        # 1  用户-物品透视表，去均值化（用户方向）
        self.user_item = pd.pivot_table(self.data, index='ACC_NBR', columns='CONTENT_NAME', values='SCORE')
        self.user_item = self.user_item.groupby(['ACC_NBR']).apply(self.Amend)
        # 2  生成关联规则，计算相似度（修正余弦）
        Rule = self.Association()
        Rule['SIMILARITY'] = Rule['RULE'].map(self.Similarity)
        # 3  规则再加工并返回
        return self.Reprocess_rule(Rule)

    # **********************两两关联规则**********************
    def Association(self):
        # 1  用户-产品表，剔除单一产品用户
        datas = self.data.groupby(['ACC_NBR']).apply(lambda x: x['CONTENT_NAME'].tolist()).reset_index().rename(
            columns={0: 'ITEM'})
        datas = datas[datas['ITEM'].map(lambda x: len(x)) > 1]
        # 2  两两规则，计算用户规则数
        datas['ITEM'] = datas['ITEM'].map(lambda x: [i for i in it.combinations(x, 2)])
        datas['COUNT'] = datas['ITEM'].map(lambda x: len(x))

        # 3  分组计算，生成规则链条并合并
        def merge(data):
            data = (np.concatenate(np.array(data['ITEM'])))
            return pd.Series((data.tolist()))

        datas = datas.groupby(['COUNT']).apply(merge).reset_index().rename(columns={0: 'RULE'}).drop(
            ['COUNT', 'level_1'], axis=1)
        # 4  分组计算个数，排序、去重
        datas['RULE'] = datas['RULE'].map(lambda x: MyList(sorted(x)))
        return datas.groupby(['RULE']).count().reset_index()

    # ***********************去均值化修正**********************
    def Amend(self, data):
        # 1  获取id，行列转置
        acc_nbr = data.index.values[0]
        data = data.T
        # 2  获取列数据
        data = data[acc_nbr]
        # 3  计算均值，返回修正数据
        mean = data.mean(axis=0, skipna=True)
        return data.map(lambda x: x - mean if x != 0 else x)

    # ***********************相似度计算************************
    def Similarity(self, rule_list):
        # 1  插入规则字典
        for rule in rule_list:
            rule_v = self.user_item[rule].copy(deep=True)
            if rule not in self.Rule.keys():
                rule_v.fillna(0, inplace=True)
                self.Rule[rule] = rule_v
        # 2  返回相似度
        return np.corrcoef(self.Rule[rule_list[0]], self.Rule[rule_list[1]])[0, 1]

    # *******************再加工，生成字典**********************
    def Reprocess_rule(self, rule):
        # 1  补0
        rule.fillna(0, inplace=True)
        # 2  再构造
        rule['L'] = rule['RULE'].map(lambda x: x[0])
        rule['R'] = rule['RULE'].map(lambda x: x[1])
        rule.drop(['RULE'], axis=1, inplace=True)
        # 3  镜像后union
        ruleT = rule[['R', 'L', 'SIMILARITY']].rename(columns={'L': 'R', 'R': 'L'})
        rule = pd.concat([rule, ruleT], sort=False)
        # 4  输出字典
        rule_dict = rule.groupby('L').apply(
            lambda x: x.set_index('R').T.to_dict('int')['SIMILARITY']).reset_index().rename(columns={0: 'RULE'})
        return rule_dict.set_index('L').T.to_dict('int')['RULE']


if __name__ == '__main__':
    start = time.time()

    file = pd.ExcelFile('target.xlsx')
    datas = pd.read_excel(file, sheet_name='Sheet1', encoding='utf-8')
    result = Rule(datas).main()
    for k, v in result.items():
        print(k, ' : ', v)
    with open('rule.txt', 'wb') as f:
        pk.dump(result, f)

    end = time.time()
    print(end - start)
