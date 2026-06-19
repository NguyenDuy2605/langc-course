# import requests
# from bs4 import BeautifulSoup
# import json
# import re

# url = "https://www.udemy.com/course/learn-mcp-model-context-protocol-course-and-a2a-bootcamphands-hands-on/?couponCode=KEEPLEARNING"

# html = requests.get(url).text

# # tìm object lớn trong html
# match = re.search(
#     r'("courseId":6554191.*?)</script>',
#     html,
#     re.S
# )

# print(match.group(1)[:500])
import requests

url = "https://www.udemy.com/course/learn-mcp-model-context-protocol-course-and-a2a-bootcamphands-hands-on/?couponCode=KEEPLEARNING"

r = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0"
    }
)

print(r.status_code)
print(r.text[:1000])