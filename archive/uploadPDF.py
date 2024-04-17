import requests
import argparse

def upload_pdf(api_key, pdf_file_path):
    url = "https://api.langchain.com/upload"  # This is a placeholder URL. Replace with the actual API endpoint.
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    files = {
        'file': open(pdf_file_path, 'rb')
    }
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        print("PDF successfully uploaded and indexed.")
        print("Response:", response.json())
    else:
        print("Failed to upload PDF.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload and index a PDF file using the LangChain API.')
    parser.add_argument('--pdf_file', type=str, help='Path to the PDF file to upload.', required=True)
    parser.add_argument('--api_key', type=str, help='Your LangChain API key.', required=True)

    args = parser.parse_args()

    upload_pdf(args.api_key, args.pdf_file)
