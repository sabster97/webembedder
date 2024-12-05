import re

def convert_markdown_to_slack(markdown_text):
    """
    Converts markdown-formatted text to Slack-compatible formatting.

    Parameters:
    markdown_text (str): The markdown-formatted text to convert.

    Returns:
    str: The text formatted for Slack.
    """
    # Remove any time stamps or artifacts like '9:14'
    slack_text = re.sub(r'\n\s*\d{1,2}:\d{2}\n', '\n', markdown_text)

    # Convert markdown headings (lines starting with #) to bold uppercase in Slack
    def replace_heading(match):
        heading = match.group(1).strip()
        return f'*{heading.upper()}*'

    slack_text = re.sub(r'^(#+)\s*(.*)', replace_heading, slack_text, flags=re.MULTILINE)

    # Convert bold (**text**) to Slack bold (*text*)
    slack_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', slack_text)

    # Convert italic (*text* or _text_) to Slack italic (_text_)
    # Avoid converting bold asterisks that have been converted to single asterisks
    slack_text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*(?!\*)', r'_\1_', slack_text)
    slack_text = re.sub(r'_(.*?)_', r'_\1_', slack_text)

    # Convert markdown links [text](url) to Slack links <url|text>
    slack_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', slack_text)

    # Convert code blocks ```code``` to Slack code blocks
    slack_text = re.sub(r'```(.*?)```', r'```\1```', slack_text, flags=re.DOTALL)

    # Convert inline code `code` to Slack inline code
    slack_text = re.sub(r'`([^`]+)`', r'`\1`', slack_text)

    return slack_text

def format_sources(sources):
    """
    Formats a list of sources into a Slack-formatted string.

    Parameters:
    sources (list): A list of dictionaries containing source information.

    Returns:
    str: The formatted sources string.
    """
    if not sources:
        return ''
    sources_text = '*Sources:*\n'
    for idx, source in enumerate(sources, 1):
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        # Slack link format: <url|title>
        sources_text += f'{idx}. <{url}|{title}>\n'
    return sources_text

def format_llm_output_for_slack(llm_output):
    """
    Formats the LLM output for professional readability in Slack.

    Parameters:
    llm_output (dict): The output from the LLM containing 'answer' and optionally 'sources'.

    Returns:
    str: The formatted Slack message.
    """
    answer_text = llm_output.get('answer', '')
    processed_answer = convert_markdown_to_slack(answer_text)

    sources = llm_output.get('sources', [])
    processed_sources = format_sources(sources)

    # Combine the answer and sources
    final_output = processed_answer.strip()
    if processed_sources:
        final_output += '\n\n' + processed_sources.strip()

    return final_output

# Example usage with the provided LLM output:
llm_output = {
    'answer': '**Comprehensive Content Analysis:**\n\n1. **Clarity, Accuracy, Relevance, and Depth:** The current content on services for clientele appears to be concise but lacks depth. It briefly mentions the services offered without providing detailed information that potential clients may seek.\n\n2. **Key Content Gaps and Areas for Improvement:**\n   - Detailed descriptions of each service offered, including benefits and outcomes.\n   - Testimonials or case studies showcasing successful client outcomes.\n   - Comparison charts or tables highlighting the differences between various service packages.\n   - Information on the expertise and qualifications of the service providers.\n   - Pricing information or a clear call-to-action for clients to inquire about pricing.\n\n3. **Tone, Readability, and Structure:** The tone is professional but could benefit from a more engaging and client-centric approach. The content is readable but lacks visual elements to break up the text and improve user experience.\n\n**Actionable Content Suggestions:**\n\n1. **Fresh Content Ideas and Topics:**\n   - "How Our Services Can Benefit You: A Comprehensive Guide"\n   - "Client Success Stories: Real-Life Examples of Our Impact"\n   - "Choosing the Right Service Package for Your Needs: A Step-by-Step Guide"\n   - "Meet Our Team: The Experts Behind Our Services"\n\n2. **Content Outlines:**\n   - **Service Descriptions:**\n     - Introduction to each service\n     - Key features and benefits\n     - Client testimonials (if available)\n   - **Client Success Stories:**\n     - Case study 1: Problem, solution, results\n     - Case study 2: Problem, solution, results\n   - **Team Introduction:**\n     - Brief bio of each team member\n     - Areas of expertise and qualifications\n\n3. **Drafts:**\n   - **Service Description (Sample):**\n     - Introduction: Brief overview of the service\n     - Benefits: How clients can benefit from the service\n     - Testimonials: Quotes from satisfied clients\n\n**Internal Linking Strategy for SEO & User Navigation:**\n\n1. **Relevant Internal Links:**\n   - Link service descriptions to related case studies or testimonials.\n   - Link team member bios to their respective service descriptions.\n   - Link pricing information to service descriptions.\n\n2. **New Internal Link Opportunities:**\n   - Link client success stories to related services for a deeper understanding.\n   - Link team member bios to relevant blog posts or articles they have written.\n   - Link service comparison charts to individual service descriptions.\n\n**User Engagement and Content Enhancement:**\n\n1. **Multimedia Elements:**\n   - Include client testimonial videos for authenticity.\n   - Add infographics showcasing service benefits or success metrics.\n   - Include images of the team members to personalize the content.\n\n2. **Calls-to-Action (CTAs):**\n   - "Contact Us for a Free Consultation Today!"\n   - "Subscribe to Our Newsletter for Exclusive Insights!"\n   - "Explore Our Services in Detail: Book a Discovery Call!"\n\n**Content and SEO Alignment:**\n\n1. **SEO Best Practices:**\n   - Incorporate relevant keywords naturally into the content.\n   - Optimize meta descriptions with compelling summaries of each service.\n   - Use proper header tags to structure content for readability and SEO.\n\n2. **Optimization Opportunities:**\n   - Target long-tail keywords related to specific services or client needs.\n   - Optimize content for voice search by including FAQs and conversational language.\n   - Utilize semantic search by answering common client queries within the content.\n\n**Continuous Improvement & Adaptability:**\n\n1. **Optimizing Older Content:**\n   - Update existing service descriptions with more detailed information.\n   - Refresh client testimonials with recent success stories.\n   - Add internal links to older content to improve navigation and SEO.\n\n2. **Evolving Trends:**\n\n    9:14\n    - Stay updated on industry trends and incorporate them into content.\n   - Monitor user feedback and analytics to adapt content strategies accordingly.\n   - Test new content formats or topics to gauge user interest and engagement.\n\nBy implementing these actionable insights, the content on services for clientele can be enhanced to better engage users, improve SEO performance, and drive conversions effectively.',
    'confidence_score': 0.0,
    'metadata': {
        'num_sources': 10,
        'query_timestamp': '2024-12-04 21:14:40',
        'token_usage': {
            'input_tokens': 4394,
            'output_tokens': 845,
            'total_tokens': 5239
        }
    },
    'sources': [
        {
            'relevance_score': 0.014448306357144114,
            'title': 'Salesforce Best Practices for Professional Services - Clientell',
            'url': 'https://www.getclientell.com/best-practices/salesforce-best-practices-professional-services'
        },
        {
            'relevance_score': -0.07082289741066172,
            'title': 'Boost Sales with Clientell + Salesforce Integration - Clientell',
            'url': 'https://www.getclientell.com/resources/blogs/transforming-sales-outcomes-with-clientell-and-salesforce-integration'
        },
        # ... additional sources ...
    ]
}

# Generate the formatted Slack message
formatted_message = format_llm_output_for_slack(llm_output)

# Output the formatted message (in an actual application, this would be sent to Slack)
print(formatted_message)
