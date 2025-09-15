import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Global WebDriver instance
_driver = None

def get_driver():
    global _driver
    if _driver is None:
        print("Setting up browser...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service(ChromeDriverManager().install())
        _driver = webdriver.Chrome(service=service, options=chrome_options)
    return _driver

def close_driver():
    global _driver
    if _driver:
        print("Closing browser...")
        _driver.quit()
        _driver = None

def get_thread_links():
    """Fetches all thread links from the main forum page using Selenium."""
    url = "https://www.agnoshealth.com/forums"
    driver = get_driver()
    
    try:
        print(f"Fetching main forum page: {url}")
        driver.get(url)
        
        # Wait for the dynamic content to load
        print("Waiting for page to load completely...")
        time.sleep(10)  # Wait 10 seconds for JS to load all content
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        links = set()
        # Find all <a> tags whose href starts with /forums/ and ends with a number (post ID)
        for a_tag in soup.find_all("a", href=lambda href: href and href.startswith("/forums/") and len(href.split('/')) > 2 and href.split('/')[-1].isdigit()):
            full_link = "https://www.agnoshealth.com" + a_tag["href"]
            links.add(full_link)
            
        return list(links)
    except Exception as e:
        print(f"An error occurred while fetching main forum page with Selenium: {e}")
        return []

def scrape_thread(url):
    """Scrapes the question and answers from a single thread URL using Selenium."""
    driver = get_driver()
    try:
        print(f"Fetching thread page with Selenium: {url}")
        driver.get(url)
        time.sleep(5) # Wait for content to load

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        question = ""
        answers = []

        # Find the question (span with font-bold text-lg)
        question_element = soup.find("span", class_="font-bold text-lg")
        if question_element:
            question = question_element.get_text(strip=True)

        # Find answers
        # Answers are within <p class="mt-4"> tags, which are inside <li> elements.
        # These <li> elements are part of a <ul> with class="space-y-4".
        
        answer_ul = soup.find("ul", class_="space-y-4")
        
        if answer_ul:
            answer_lis = answer_ul.find_all("li")
            for li in answer_lis:
                answer_p = li.find("p", class_="mt-4")
                if answer_p:
                    answers.append(answer_p.get_text(strip=True))
        
        return {
            "url": url,
            "question": question,
            "answers": answers
        }
    except Exception as e:
        print(f"An unexpected error occurred while scraping thread {url} with Selenium: {e}")
        return None

def main():
    """Main function to run the scraper."""
    print("Starting scraper with Selenium...")
    try:
        thread_links = get_thread_links()
        
        if not thread_links:
            print("No thread links found after using Selenium. There might be an issue with the selectors or page structure. Exiting.")
            return

        print(f"Found {len(thread_links)} unique threads to scrape.")
        
        all_data = []
        for link in tqdm(thread_links, desc="Scraping Threads"):
            scraped_data = scrape_thread(link)
            if scraped_data and (scraped_data['question'] or scraped_data['answers']):
                all_data.append(scraped_data)
            time.sleep(0.1)

        output_filename = "agnos_forum_data.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
            
        print(f"\nScraping complete. Data saved to {output_filename}")
        print(f"Successfully scraped {len(all_data)} threads.")
    finally:
        close_driver()

if __name__ == "__main__":
    main()
