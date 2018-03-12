# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from hello.items import HelloItem


class HelloCrawlSpider(CrawlSpider):
    name = 'hello_crawl'
    allowed_domains = ['www.abuquant.com']
    start_urls = ['http://www.abuquant.com/article']

    rules = (
        Rule(LinkExtractor(allow=r'^http://www.abuquant.com/article$'), callback='parse_item', follow=False),
        Rule(LinkExtractor(allow=r'article/page'), callback='parse_item', follow=False),
    )

    def parse_item(self, response):
        for sel in response.css('div.post-desc'):
            subsel = sel.css('div.post-title h2.entry-title a')
            item = HelloItem()
            item['title'] = subsel.xpath('text()').extract()
            item['text'] = sel.css('div.post-excerpt::text').extract()
            item['link'] = subsel.xpath('@href').extract()
            yield item
