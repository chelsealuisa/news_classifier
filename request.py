import requests

url = 'http://localhost:5000/api'
r = requests.post(url,json={'title':'Hurricane Makes Landfall in North Carolina',})
print(r.json())