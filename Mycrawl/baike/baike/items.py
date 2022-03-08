# -*- coding: utf-8 -*-

# Define here the model for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BaikeItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class Crop(scrapy.Item):
    context = scrapy.Field()
    name = scrapy.Field()
    pass
