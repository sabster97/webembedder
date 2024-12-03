import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlparse
import os

class WebsiteScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        self.raw_data_dir = 'raw_data'
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def clean_text(self, text):
        lines = (line.strip() for line in text.splitlines())
        text = ' '.join(line for line in lines if line)
        
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s-]', '', text)
        return text

    def scrape_urls_from_sitemap(self, sitemap_path):
        with open(sitemap_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
            urls = [url.text for url in soup.find_all('loc')]
            print(f"Found {len(urls)} URLs in sitemap")
            return urls

    def scrape_page(self, url):
        try:
            time.sleep(1)
            print(f"Scraping: {url}")
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                
                if main_content:
                    page_data = {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'content': self.clean_text(main_content.get_text()),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    
                    filename = f"{self.raw_data_dir}/{urlparse(url).path.replace('/', '_')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(page_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"Successfully scraped: {url}")
                    return page_data
                    
            print(f"Failed to scrape {url}: No main content found")
            return None
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def scrape_all(self, sitemap_path):
        urls = self.scrape_urls_from_sitemap(sitemap_path)
        results = []
        
        for url in urls:
            result = self.scrape_page(url)
            if result:
                results.append(result)
        
        print(f"Scraping completed. Successfully scraped {len(results)} pages")
        return results

if __name__ == "__main__":
    scraper = WebsiteScraper()
    results = scraper.scrape_all('sitemap.xml')