# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
from baike.items import Crop
import re

class CropSpider(scrapy.Spider):
    name = 'crop'
    allowed_domains = ['baike.baidu.com']
    file_object = open('fenlei.txt', 'r', encoding='utf-8').read()
    wordList = file_object.split()  # 获取词表
    start_urls = []
    for i in wordList:  ##生成url列表
        cur = "https://baike.baidu.com/item/"
        cur = cur + str(i)
        start_urls.append(cur)

    # def __init__(self, type):
    #     self.type = type

    def parse(self, response):
        sketch = response.text
        exc = BeautifulSoup(sketch, "html.parser")
        crop_context = Crop()
        name = response.xpath(
            "/html/body[@class='wiki-lemma normal']/div[@class='body-wrapper']/div[@class='content-wrapper']/div[@class='content']/div[@class='main-content']/dl[@class='lemmaWgt-lemmaTitle lemmaWgt-lemmaTitle-']/dd[@class='lemmaWgt-lemmaTitle-title']/h1").extract_first()
        pattern = re.compile('<\w*>(.*)</\w*>')
        n = pattern.findall(name)
        p = exc.find_all(name='div', attrs='para')
        for i in range(len(p)):
            crop_context['context'] = p[i].get_text()
            crop_context['name'] = n[0]
            yield crop_context
            pass