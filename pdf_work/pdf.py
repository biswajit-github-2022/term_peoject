import fitz  # PyMuPDF
import pandas as pd
import re

# Define a function to categorize the text
def categorize_text(text):
    if re.match(r'^\s*Title\s*:\s*', text, re.IGNORECASE):
        return 'Title'
    elif re.match(r'^\s*Abstract\s*:\s*', text, re.IGNORECASE):
        return 'Abstract'
    elif re.match(r'^\s*Introduction\s*:\s*', text, re.IGNORECASE):
        return 'Introduction'
    else:
        return 'Body'

# Path to the PDF file
pdf_path = 'cv.pdf'

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Initialize a list to store the extracted and categorized text
data = []

# Iterate through each page
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    text = page.get_text("text")

    # Split the text into lines and categorize
    for line in text.split('\n'):
        category = categorize_text(line)
        data.append({'Page': page_num + 1, 'Category': category, 'Text': line.strip()})

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_path = 'categorized_text.csv'
df.to_csv(csv_path, index=False)

print(f'Text has been extracted and saved to {csv_path}')
