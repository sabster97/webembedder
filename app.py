from flask import Flask, request, jsonify
from db import DocumentQA
import os
from dotenv import load_dotenv
import requests
import asyncio
from threading import Thread
from slack_format import format_for_slack
import json
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
    print("response ---> ", response)
    answer = response.get('answer', '')
    formatted_answer = format_for_slack(answer)
    # Convert string to JSON if it's a string
    if isinstance(formatted_answer, str):
        formatted_answer = json.loads(formatted_answer)
    # Send the formatted message to Slack
    send_slack_message(channel_id, formatted_answer, thread_ts)

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
        "blocks": text["blocks"],
        "text": "Message from Bot"
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)


@app.route('/query', methods=['POST'])
def query():
    try:
        print("Query received")
        data = request.get_json()
        print(data)
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query_text = data['query']
        print(query_text)
        # Classify the query intent
        sys_prompt = qa.classify_query(query_text)
        
        print("sys_prompt ---> ", sys_prompt)
        print("query_text ---> ", query_text)
        response = qa.ai_magic(sys_prompt, query_text)
        
        return jsonify({
            "intent": sys_prompt,
            "response": response
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 