from flask import Flask, request, jsonify
from db import DocumentQA
import os
from dotenv import load_dotenv
import requests
import asyncio
from threading import Thread

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
qa = DocumentQA()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
API_URL = os.getenv("API_URL")

@app.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.get_json()

    # Slack event challenge verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Handle message events
    if data['event']['type'] == 'message' and 'subtype' not in data['event'] and not data['event']['text'].startswith("Bot:"):
        user_query = data['event']['text']
        channel_id = data['event']['channel']
        thread_ts = data['event'].get('ts')

        # Run the task in a separate thread
        if not (data['event'].get('bot_id') or data['event'].get('subtype') == "message_changed"):
            thread = Thread(target=run_async_task, args=(user_query, channel_id, thread_ts))
            thread.start()

    return jsonify({"ok": True})

def run_async_task(user_query, channel_id, thread_ts):
    """Run async task in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task = loop.create_task(execute(user_query, channel_id, thread_ts))
        loop.run_until_complete(task)
    finally:
        loop.close()

async def execute(user_query, channel_id, thread_ts):
    """Execute the task."""
    # Call your API with the user's query
    api_response = requests.post(API_URL, json={"query": user_query})
    api_data = api_response.json()

    # Format the response based on the API response structure
    response = api_data.get('response', {})
    intent = api_data.get('intent', 'general')

    if isinstance(response, dict) and 'search_results' in response:
        # Format for general questions with search results
        message = "*Question:*\n" + user_query + "\n\n"
        message += "*AI Response:*\n" + str(response.get('ai_response', 'No response')) + "\n\n"
        
        if response.get('search_results'):
            message += "*Related Documents:*\n"
            for idx, result in enumerate(response['search_results'], 1):
                message += f"#{idx}. {result}\n"
    else:
        # Format for CTA or SEO improvements
        message = "*Query:*\n" + user_query + "\n\n"
        message += "*Suggestions:*\n" + str(response)

    # Send the formatted message to Slack
    #Temp code for testing
    msg = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Current SEO Strengths and Weaknesses:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "**Strengths:**\n- The content is rich in relevant keywords related to Salesforce best practices across various industries.\n- The website has a good amount of content, which can help in targeting a wide range of search queries.\n- The content provides detailed information about Salesforce best practices, implementation checklists, and benefits for different sectors."
                }
    },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "**Weaknesses:**\n- Lack of unique meta descriptions and title tags for each page, which can affect click-through rates in search results.\n- Limited use of internal linking to connect related content and improve website structure.\n- The content could be more organized and structured for better user experience and SEO."
                }
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Keyword Optimization Suggestions:"
                }
    },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "- Conduct keyword research to identify high-volume and relevant keywords related to Salesforce best practices in different industries.\n- Optimize meta titles, headings, and content with targeted keywords to improve visibility in search results.\n- Use long-tail keywords specific to each industry or service to attract more qualified traffic.\n- Include keywords in image alt text, URLs, and meta descriptions for better optimization."
                }
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Content Structure Improvements:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "- Create a clear hierarchy of content with headings, subheadings, and bullet points for easy readability.\n- Use internal linking to connect related content and guide users to explore more pages on the website.\n- Consider creating pillar pages for each industry or service category to consolidate related content and improve SEO."
                }
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Meta Description and Title Tag Recommendations:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "- Craft unique and compelling meta descriptions that accurately describe the content of each page and include relevant keywords.\n- Optimize title tags to be concise, descriptive, and include primary keywords to improve click-through rates and search visibility.\n- Ensure meta descriptions and title tags are within recommended character limits for better display in search results."
                }
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Specific Actionable Improvements:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "1. Implement a consistent internal linking strategy to connect related content and improve website structure.\n2. Optimize meta descriptions and title tags for each page with relevant keywords and unique descriptions.\n3. Consider creating industry-specific landing pages with targeted content and keywords to attract more organic traffic.\n4. Improve content organization by grouping related topics together and creating clear navigation paths for users.\n5. Regularly update and refresh content to keep it relevant and engaging for users and search engines."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "By implementing these improvements, you can enhance the website's SEO performance, increase visibility in search results, and attract more qualified traffic from organic search."
                }
            }
        ]
    }
    send_slack_message(channel_id, msg, thread_ts)

async def cancel_task_after(task, delay):
    """Cancel the task after a delay."""
    await asyncio.sleep(delay)
    if not task.done():
        task.cancel()

def send_slack_message(channel, text, thread_ts=None):
    """Send a message to Slack."""
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": channel,
        # "text": "Bot: " +text,
        "blocks": text if isinstance(text, list) else text["blocks"]
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query_text = data['query']
        
        # Classify the query intent
        intent = qa.classify_query(query_text)
        
        # Process based on intent
        if intent == "improve_cta":
            response = qa.improve_cta(query_text)
        elif intent == "improve_seo":
            response = qa.improve_seo(query_text)
        else:
            # For general questions, include both search results and AI response
            search_results = qa.search_documents(query_text)
            ai_response = qa.ask(query_text)
            response = {
                "search_results": search_results,
                "ai_response": ai_response
            }
        
        return jsonify({
            "intent": intent,
            "response": response
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 