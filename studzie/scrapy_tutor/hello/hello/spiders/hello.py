# -*- coding: utf-8 -*-
import scrapy
from hello.items import HelloItem


class ExampleSpider(scrapy.Spider):
    name = 'hello'
    allowed_domains = ['www.abuquant.com']
    start_urls = ['http://www.abuquant.com/article/']

    def parse(self, response):
        for sel in response.css('div.post-desc'):
            subsel = sel.css('div.post-title h2.entry-title a')
            item = HelloItem()
            item['title'] = subsel.xpath('text()').extract()
            item['text'] = sel.css('div.post-excerpt::text').extract()
            item['link'] = subsel.xpath('@href').extract()
            yield item
        for url in response.css('div.pages a::attr(href)').extract():
            yield scrapy.Request(url=url, callback=self.parse)