import json

def format_for_slack(input_text):
    sections = input_text.split("\n\n")
    blocks = []

    for section in sections:
        # Add section header
        if section.startswith("###"):
            header = section.split("\n")[0].replace("### ", "")
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header
                }
            })
            # Extract the remaining part
            section = "\n".join(section.split("\n")[1:])

        # Add text block
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": section
            }
        })

    # Format for Slack JSON payload
    slack_payload = {
        "blocks": blocks
    }

    return json.dumps(slack_payload, indent=2)

if __name__ == "__main__":
    # Example usage
    input_text = """### Current SEO Strengths and Weaknesses:
    **Strengths:**
    - The content is rich in relevant keywords related to Salesforce best practices across various industries.
    - The website has a good amount of content, which can help in targeting a wide range of search queries.
    - The content provides detailed information about Salesforce best practices, implementation checklists, and benefits for different sectors.

    **Weaknesses:**
    - Lack of unique meta descriptions and title tags for each page, which can affect click-through rates in search results.
    - Limited use of internal linking to connect related content and improve website structure.
    - The content could be more organized and structured for better user experience and SEO.

    ### Keyword Optimization Suggestions:
    - Conduct keyword research to identify high-volume and relevant keywords related to Salesforce best practices in different industries.
    - Optimize meta titles, headings, and content with targeted keywords to improve visibility in search results.
    - Use long-tail keywords specific to each industry or service to attract more qualified traffic.
    - Include keywords in image alt text, URLs, and meta descriptions for better optimization.

    ### Content Structure Improvements:
    - Create a clear hierarchy of content with headings, subheadings, and bullet points for easy readability.
    - Use internal linking to connect related content and guide users to explore more pages on the website.
    - Consider creating pillar pages for each industry or service category to consolidate related content and improve SEO.

    ### Meta Description and Title Tag Recommendations:
    - Craft unique and compelling meta descriptions that accurately describe the content of each page and include relevant keywords.
    - Optimize title tags to be concise, descriptive, and include primary keywords to improve click-through rates and search visibility.
    - Ensure meta descriptions and title tags are within recommended character limits for better display in search results.

    ### Specific Actionable Improvements:
    1. Implement a consistent internal linking strategy to connect related content and improve website structure.
    2. Optimize meta descriptions and title tags for each page with relevant keywords and unique descriptions.
    3. Consider creating industry-specific landing pages with targeted content and keywords to attract more organic traffic.
    4. Improve content organization by grouping related topics together and creating clear navigation paths for users.
    5. Regularly update and refresh content to keep it relevant and engaging for users and search engines.

    By implementing these improvements, you can enhance the website's SEO performance, increase visibility in search results, and attract more qualified traffic from organic search."""

    formatted_text = format_for_slack(input_text)
    print(formatted_text)
