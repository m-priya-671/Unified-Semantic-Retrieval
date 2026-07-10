import time
from playwright.sync_api import sync_playwright

def main():
    print("Connecting to browser remote debugger over CDP...")
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            print("Successfully connected over CDP.")
            
            streamlit_page = None
            for context in browser.contexts:
                for page in context.pages:
                    print(f"Found page: {page.url}")
                    if "8501" in page.url:
                        streamlit_page = page
                        break
                if streamlit_page:
                    break
            
            if not streamlit_page:
                print("Error: Could not locate the Streamlit app tab (port 8501) in browser contexts.")
                return
                
            print(f"Targeting Streamlit page: {streamlit_page.url}")
            
            # Locate file uploader input
            file_input = streamlit_page.locator("input[type='file']")
            file_input.wait_for(state="attached", timeout=5000)
            
            target_file_path = r"C:\xampp\htdocs\Offline-RAG\data\uploads\ocr_test_mixed.png"
            print(f"Setting input files to: {target_file_path}")
            file_input.set_input_files(target_file_path)
            
            print("File set action successfully dispatched.")
            # Give it a moment to transfer the file and trigger the upload action
            time.sleep(3)
            
        except Exception as e:
            print(f"Failed to connect or upload file: {e}")

if __name__ == "__main__":
    main()
