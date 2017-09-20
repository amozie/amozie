import http.client

conn = http.client.HTTPConnection("localhost:8080")

payload = "{\r\n\"systemId\":\"kamisama\",\r\n\"curTime\":\"1505873549\",\r\n\"sign\":\"083f6629393f4d368411ed49a8c273ea\",\r\n\"docInfoList\":[\r\n{\r\n\"docExId\":\"postman0300\",\r\n\"creatorId\":\"e@e.e\",\r\n\"docName\":\"postman0300\",\r\n\"signatorList\":[\r\n\t{\r\n\t\t\"email\":\"testcustomid\"\r\n\t},\r\n\t{\r\n\t\t\"email\":\"e@e.e\"\r\n\t}\r\n],\r\n\"signType\":\"001\",\r\n\"fileName\":\"postman0300.pdf\",\r\n\"fileData\":\"JVBERi0xLjcKJcKzx9gNCjEgMCBvYmoNPDwvTmFtZXMgPDwvRGVzdHMgNCAwIFI+PiAvT3V0bGluZXMgNSAwIFIgL1BhZ2VzIDIgMCBSIC9UeXBlIC9DYXRhbG9nPj4NZW5kb2JqDTMgMCBvYmoNPDwvQXV0aG9yICgpIC9Db21tZW50cyAoKSAvQ29tcGFueSAoKSAvQ3JlYXRpb25EYXRlIChEOjIwMTcwNzE5MTgxNzA1KzEwJzE3JykgL0NyZWF0b3IgKCkgL0tleXdvcmRzICgpIC9Nb2REYXRlIChEOjIwMTcwNzE5MTgxNzA1KzEwJzE3JykgL1Byb2R1Y2VyICgpIC9Tb3VyY2VNb2RpZmllZCAoRDoyMDE3MDcxOTE4MTcwNSsxMCcxNycpIC9TdWJqZWN0ICgpIC9UaXRsZSAoKSAvVHJhcHBlZCBmYWxzZT4+DWVuZG9iag02IDAgb2JqDTw8L0NvbnRlbnRzIDcgMCBSIC9NZWRpYUJveCBbMCAwIDU5NS4zIDg0MS45XSAvUGFyZW50IDIgMCBSIC9UeXBlIC9QYWdlPj4NZW5kb2JqDTcgMCBvYmoNPDwvRmlsdGVyIC9GbGF0ZURlY29kZSAvTGVuZ3RoIDI1Pj4NCnN0cmVhbQ0KeJwzVDAAQl1DIGFhYqhnqZCcywsAJBsDwQ0KZW5kc3RyZWFtDWVuZG9iag0yIDAgb2JqDTw8L0NvdW50IDEgL0tpZHMgWzYgMCBSXSAvVHlwZSAvUGFnZXM+Pg1lbmRvYmoNNCAwIG9iag08PC9OYW1lcyBbXT4+DWVuZG9iag01IDAgb2JqDTw8Pj4NZW5kb2JqDXhyZWYNCjAgOA0KMDAwMDAwMDAwMCA2NTUzNSBmDQowMDAwMDAwMDE2IDAwMDAwIG4NCjAwMDAwMDA1MzYgMDAwMDAgbg0KMDAwMDAwMDEwMyAwMDAwMCBuDQowMDAwMDAwNTkxIDAwMDAwIG4NCjAwMDAwMDA2MjAgMDAwMDAgbg0KMDAwMDAwMDM1MCAwMDAwMCBuDQowMDAwMDAwNDM5IDAwMDAwIG4NCnRyYWlsZXI8PC9TaXplIDggL1Jvb3QgMSAwIFIgL0luZm8gMyAwIFIgL0lEIFs8ZWUwNWMxNGYwNWE5NDE5ZThmNTRlMTM2NDljZjc1Yjk+PDhkNzk4OTE1YmZkZjQ3YmU4Mjg1OGM3YjgyMzA4NmI1Pl0+Pg1zdGFydHhyZWYNNjQwDSUlRU9GDQ==\",\r\n\"duration\":\"\",\r\n\"notifySet\":\"\"\r\n}]\r\n}\r\n"

headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "31964b88-c3a8-7892-fe78-9645c91ccf73"
    }

conn.request("POST", "/JustSignService/v1/doc/addDocList", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))