import requests

url = "http://127.0.0.1:8000/prediction"
files = {"file": open("empty_gender_name_liste.xlsx", "rb")}
response = requests.post(url, files=files)

with open("result.xlsx", "wb") as f:
    f.write(response.content)

print("✅ File saved as result.xlsx")