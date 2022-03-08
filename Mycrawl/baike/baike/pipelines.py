# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import os


class BaikePipeline(object):
    def open_spider(self, spider):
        pass

def process_item(self, item, spider):
    res = dict(item)
    line = res['context']
    # typee = res['typee']
    name = res['name']
    path = 'F:/python/project/Agriculture/croptext/' + name + '.txt'
    file = open(path, 'a', encoding='utf8')
    l = line.replace('\t', '').replace('\n', '').replace('\r', '').replace(' ', '')
    file.write(l + '\n')
    file.close()

    def close_spider(self, spider):
        pass
