from urllib.request import quote
import requests
import time
from lxml import etree
import pandas as pd
import re

def get_cookie(cookie):
    cookies = {}
    for item in cookie.split(";"):
        cookies[item.split("=")[0]] = item.split("=")[1]
    return cookies

def get_html(url):
    cookie = '_T_WM=700ac34f93304aaa7d0e4c5f8e1e365d; SCF=Ai8pGZKJnqDhx69nDRTuvWQ9at4YdUNFzz5uuriBzA1_5ge9jQo7dftkmJ3PWhV_ojUCiQx_V1DBbYd-alv-cPQ.; SUB=_2A25PZS5MDeRhGeRL6VcX-S_IzT6IHXVsqbIErDV6PUJbktB-LXj_kW1NU1A7DJy6eRCDLyRB8J3_z7D5Gd9Fa0FG; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W54Gz_a5z4B7GarHIUeE5WH5JpX5K-hUgL.Fozfeo-c1K2XSoz2dJLoI7U0Cgizq-p0; ALF=1653140253'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36',
    }
    response = requests.get(url, headers=headers).content.decode()
    html = bytes(bytearray(response, encoding='utf-8'))
    html = etree.HTML(html)
    return html

# 正则匹表达提取数字
def get_num(template):
    rule =  re.compile(r'(\d.+?\d*)')
    slotList = rule.findall(template)
    if slotList == []:
        return ''
    else:
        return slotList[0]

# 获取最终数据
def get_data(url):
    html = get_html(url)
    title = html.xpath('//div[@class="journal_right"]/a/text()')
    label = html.xpath('//span[@class="label"]/text()')
    data_list = html.xpath('//span[@class="factor"]/text()')
    effect = []
    quote = []
    search = []
    send = []
    for n in range(len(data_list)):
        if n % 4 == 0:
            effect.append(get_num(data_list[n]))
        elif n % 4 == 1:
            quote.append(get_num(data_list[n]))
        elif n % 4 == 2:
            search.append(get_num(data_list[n]))
        elif n % 4 == 3:
            send.append(get_num(data_list[n]))
    return title, effect, quote, search, send


if __name__ == '__main__':
    start = time.time()

    # 正则网址
    target = '中国科技核心期刊'
    target_code = quote(target, encoding='utf-8')
    for i in range(3):
        url = 'https://xueshu.baidu.com/usercenter/journal/navigation?query=&language=1&journal_db=4&journal_name=' + target_code + '&page={}'.format(i + 1)
        # 获取数据
        results = pd.DataFrame(data=None, columns=['名称', '影响因子', '被引用量', '搜索指数', '发文量'])
        results['名称'], results['影响因子'], results['被引用量'], results['搜索指数'], results['发文量'] = get_data(url)
        results.to_excel(r'C:\job\Machinelearning\20220418_2280_cf\result{}.xlsx'.format(i), encoding='utf_8_sig', index=0)

    end = time.time()
    print(end - start)


