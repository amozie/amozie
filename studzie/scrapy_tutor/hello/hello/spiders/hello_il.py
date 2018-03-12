# -*- coding: utf-8 -*-
import scrapy
from hello.items import HelloItem
from scrapy.loader import ItemLoader


class ExampleSpider(scrapy.Spider):
    name = 'hello_il'
    allowed_domains = ['www.abuquant.com']
    start_urls = ['http://www.abuquant.com/article/']

    def parse(self, response):
        l = ItemLoader(item=HelloItem(), response=response)
        l.add_css('text', 'h2.entry-title a::text')
        l.add_css('text', 'h2.entry-title a::attr(href)')
        return l.load_item()