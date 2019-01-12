import requests,chardet,psutil
r = requests.get('https://www.douban.com/')
print(r.status_code)
r = requests.get('https://www.douban.com/search', params={'q': 'python', 'cat': '1001'})
print(r.url)
print(chardet.detect(b'Hello, world!'))
data = '离离原上草，一岁一枯荣'.encode('gbk')
print(chardet.detect(data))
data = '最新の主要ニュース'.encode('euc-jp')
print(chardet.detect(data))
print(psutil.cpu_count()) # CPU逻辑数量