#coding=utf-8

import urllib2
import re
import os
# import sys
import shutil
from bs4 import BeautifulSoup
# reload(sys)
# sys.setdefaultencoding('utf-8')


def pullImages(productId) :
	url1 = 'https://item.jd.com/' + str(productId) + '.html#none'
	str1 = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv2136&productId='+str(productId)
	str2 = '&score=0&sortType=5&page='
	str3 = '&pageSize=10&isShadowSku=0&fold=1'

	#find the title
	req = urllib2.urlopen(url1)
	buf = req.read()
	soup = BeautifulSoup(buf, "html.parser")
	div_title = soup.find_all("div", {'class':'sku-name'})
	title = div_title[0].string
	if title == None :
		return
	else :
		title = title.strip()
	try :
		print title
	except UnicodeEncodeError :
		pass

	#防止标题出现字符'/'，导致文件夹创建失败
	title = title.replace('/', '#')
	saveDir = 'imgs/' + title
	if os.path.isdir(saveDir) :
		#shutil.rmtree('imgs')
		#os.mkdir('imgs')
      #return
		pass
	else :
		os.mkdir(saveDir)

	#爬取商品图片
	spec_list = soup.find('div', {'id':'spec-list'})
	tmp = spec_list.find_all('img')
	i = 0
	for item in tmp :
		img_src = str(item.get('src'))
		img_src = 'http:'+img_src.replace('54x54', '450x450').replace('n5', 'n1')
		req = urllib2.urlopen(img_src)
		buf = req.read()
		f = open(saveDir + '/' + str(i) + '.jpg', 'wb')
		f.write(buf)
		# print img_src
		i = i + 1
	# print ''

	url2 = str1 + str2 + str(0) + str3;
	f = open(saveDir + '/' + 'data.txt', 'wb')
	f.write(url1 + '\t' + url2)

	pageCount = 3
	k = 0
	while k < pageCount :
		url2 = str1 + str2 + str(k) + str3;

		#爬取评论区图片
		try :
			req = urllib2.urlopen(url2)
		except urllib2.URLError:
			pass
		buf = req.read()
		soup = BeautifulSoup(buf, "html.parser")
		img_list = soup.find_all(name='img')

		for url in img_list :
			f = open(saveDir + '/' + str(i) + '.jpg', 'wb')
			img_url = url.get('src')
			if 'http:' in img_url :
				pass
			else :
				img_url = 'http:' + img_url
			print img_url
			req = urllib2.urlopen(img_url)
			buf = req.read()
			f.write(buf)
			i = i + 1
		if i < 400 and k == pageCount - 1:
			pageCount += 1
		k += 1

	return;


pullImages('2892813')
