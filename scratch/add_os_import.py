with open("src/origin_classifier.py", "rb") as f:
    content = f.read().decode("utf-8")

content = content.replace("import hashlib\nimport logging", "import os\nimport hashlib\nimport logging")
content = content.replace("import hashlib\r\nimport logging", "import os\r\nimport hashlib\r\nimport logging")

with open("src/origin_classifier.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done")

