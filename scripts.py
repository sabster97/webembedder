import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlparse, urljoin
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
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Main content extraction
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                
                if main_content:
                    # Title
                    title = soup.title.string if soup.title else ''
                    
                    # Meta description and keywords
                    meta_description = ''
                    meta_keywords = []
                    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
                    if meta_desc_tag:
                        meta_description = meta_desc_tag.get('content', '')
                    
                    # meta_keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
                    # if meta_keywords_tag:
                    #     meta_keywords = [kw.strip() for kw in meta_keywords_tag.get('content', '').split(',')]
                    
                    meta_keywords = []
                    # Internal and external links
                    internal_links = []
                    external_links = []
                    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(url))
                    
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        full_url = urljoin(base_url, href)
                        if urlparse(full_url).netloc == urlparse(url).netloc:
                            internal_links.append(full_url)
                        else:
                            external_links.append(full_url)
                    
                    # Outline (Headings)
                    outline = []
                    
                    # Personas (simple heuristic-based example)
                    personas = []
                    
                    # Additional keywords (from content and hyperlinks)
                    content_text = main_content.get_text()
                    keywords = []  # Top 10 words as keywords
                    additional_keywords = []
                    
                    # Link structure
                    link_structure = {
                        'internal': internal_links,
                        'external': external_links
                    }
                    
                    # Intent (basic heuristic for intent classification)
                    intent = ""
                    
                    # Page data structure
                    page_data = {
                        'url': url,
                        'title': title,
                        'intent': intent,
                        'keywords': keywords,
                        'internal_links': internal_links,
                        'meta_description': meta_description,
                        'meta_keywords': meta_keywords,
                        'content': self.clean_text(main_content.get_text()),
                        'pillar_keywords': keywords,
                        'personas': personas,
                        'additional_keywords': additional_keywords,
                        'outline': outline,
                        'link_structure': link_structure,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save to file
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
