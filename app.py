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
    send_slack_message(channel_id, message, thread_ts)

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
        "text": "Bot: " + text,
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