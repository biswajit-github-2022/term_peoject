import requests

url = "https://brain.predis.ai/predis_api/v1/create_content/"

payload = {
  "brand_id": "665e4a8423b79f50ac852c43",
  "text": "3 tips for a healthy morning breakfast",
  "media_type": "single_image"
}

headers = {"Authorization": "O6oTs2mJc6Rn6VYmkypAMVmlyHxeyOS5"}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)
# {
#     "post_ids": [
#         "CREATED_POST_ID"
#     ],
#     "post_status": "inProgress",
#     "errors": []
# }
