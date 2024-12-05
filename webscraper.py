import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import json
import os
from typing import List, Dict, Tuple

class SEOWebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.base_url = None

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s-]', '', text)
        return text.strip()

    def get_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        try:
            response = self.session.get(sitemap_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'xml')
            self.base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(sitemap_url))
            
            urls = []
            for url in soup.find_all('loc'):
                urls.append(url.text)
            
            print(f"Found {len(urls)} URLs in sitemap")
            return urls
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
            return []

    def extract_meta_data(self, soup: BeautifulSoup, url: str) -> Tuple[str, str, str]:
        meta_title = ""
        meta_description = ""
        title = ""

        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
            meta_title = title

        meta_title_tag = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'title'})
        if meta_title_tag:
            meta_title = meta_title_tag.get('content', '')

        meta_desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', property='og:description')
        if meta_desc_tag:
            meta_description = meta_desc_tag.get('content', '')

        return title, meta_title, meta_description

    def extract_headings(self, soup: BeautifulSoup) -> List[str]:
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append(f"H{i}: {heading.get_text().strip()}")
        return headings

    def extract_internal_links(self, soup: BeautifulSoup, base_url: str) -> Tuple[List[str], Dict[str, str]]:
        internal_links = []
        keywords_in_links = {}
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            full_url = urljoin(base_url, href)
            
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                internal_links.append(full_url)
                keywords_in_links[full_url] = link.get_text().strip()
        
        return internal_links, keywords_in_links

    def calculate_keyword_density(self, content: str, keyword: str) -> str:
        return ""

    def extract_top_keywords(self, content: str, num_keywords: int = 5) -> List[str]:
        return [""]

    def infer_page_intent(self, content: str, title: str) -> str:
        return ""

    def infer_persona(self, content: str, title: str) -> str:
        return ""

    def scrape_page(self, url: str, original_sitemap: str) -> Dict:
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: Status code {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title, meta_title, meta_description = self.extract_meta_data(soup, url)
            
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            content_text = main_content.get_text() if main_content else ""
            clean_content = self.clean_text(content_text)
            
            internal_links, keywords_in_links = self.extract_internal_links(soup, url)
            headings = self.extract_headings(soup)
            
            try:
                parsed_url = urlparse(url)
                path_parts = [p for p in parsed_url.path.split('/') if p]
                if path_parts:
                    user_path = 'Home > ' + ' > '.join(path_parts)
                else:
                    user_path = 'Home'
            except Exception as e:
                print(f"Error creating user path for {url}: {e}")
                user_path = 'Home'
            
            page_struct = {
                "title": title or "",
                "url": url,
                "meta_title": meta_title or "",
                "meta_description": meta_description or "",
                "original_sitemap": original_sitemap,
                "bounce_rate": "",
                "visitors": "",
                "page_intent": self.infer_page_intent(clean_content, title),
                "persona": self.infer_persona(clean_content, title),
                "top_keywords": "",
                "internal_links": ", ".join(internal_links) if internal_links else "",
                "keywords_in_links": ", ".join(set(keywords_in_links.values())) if keywords_in_links else "",
                "pillar_keyword": "",
                "target_keyword": "",
                "word_count": len(clean_content.split()) if clean_content else 0,
                "title_length": len(title) if title else 0,
                "meta_title_length": len(meta_title) if meta_title else 0,
                "meta_description_length": len(meta_description) if meta_description else 0,
                "content_type": "",
                "h1_h6_headings": ", ".join(headings) if headings else "",
                "keyword_density": self.calculate_keyword_density(clean_content, ""),
                "user_path": user_path,
                "raw_content": clean_content
            }
            
            print(f"Successfully scraped: {url}")
            return page_struct
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def scrape_all(self, sitemap_url: str) -> List[Dict]:
        urls = self.get_urls_from_sitemap(sitemap_url)
        results = []
        
        for url in urls:
            result = self.scrape_page(url, sitemap_url)
            if result:
                results.append(result)
            time.sleep(1)
        
        return results

def main():
    scraper = SEOWebScraper()
    results = scraper.scrape_all("https://www.getclientell.com/sitemap.xml")
    
    output_dir = 'scraped_web_data'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/pages_data.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Scraping completed. Processed {len(results)} pages.")

if __name__ == "__main__":
    main()
