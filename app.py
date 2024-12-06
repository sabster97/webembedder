from flask import Flask, request, jsonify
from retriever import DocumentQA
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
print("API_URL ---> ", API_URL)

@app.route('/slack/events', methods=['POST'])
def slack_events():
    print("Slack event received")
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
    print("Slack event processed")
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
    try:
        print("Executing task...")
        # Call your API with the user's query
        api_response = requests.post(API_URL, json={"query": user_query})
        print("api_response", api_response)
        api_response.raise_for_status()  # Raise an exception for bad status codes
        
        try:
            api_data = api_response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Failed to decode API response: {api_response.text}")
            error_message = {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, I received an invalid response from the API."}}]}
            send_slack_message(channel_id, error_message, thread_ts)
            return

        # Format the response based on the API response structure
        response = api_data.get('response', {})
        answer = response.get('answer', '')
        formatted_answer = format_for_slack(answer)
        # Convert string to JSON if it's a string
        if isinstance(formatted_answer, str):
            try:
                formatted_answer = json.loads(formatted_answer)
            except json.JSONDecodeError as e:
                print(f"Failed to decode formatted answer: {formatted_answer}")
                error_message = {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, I encountered an error formatting the response."}}]}
                send_slack_message(channel_id, error_message, thread_ts)
                return

        print("Task completed")
        # Send the formatted message to Slack
        send_slack_message(channel_id, formatted_answer, thread_ts)
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        error_message = {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "Sorry, I couldn't reach the API at this time."}}]}
        send_slack_message(channel_id, error_message, thread_ts)

async def cancel_task_after(task, delay):
    """Cancel the task after a delay."""
    await asyncio.sleep(delay)
    if not task.done():
        task.cancel()

def filter_valid_blocks(blocks):
    """
    Filters out blocks with empty or invalid text fields.

    Parameters:
    blocks (list): The list of blocks to validate.

    Returns:
    list: A list of valid blocks.
    """
    valid_blocks = []
    for block in blocks:
        if block.get("type") == "section":
            text = block.get("text", {})
            if text.get("text", "").strip():  # Check if text is non-empty
                valid_blocks.append(block)
            else:
                print(f"Skipping invalid block: {block}")
        else:
            valid_blocks.append(block)
    return valid_blocks


def send_slack_message(channel, text, thread_ts=None):
    """Send a message to Slack."""
    # print("Entered send_slack_message")
    # print("format_for_slack ---> ", len(text["blocks"]))
    valid_blocks = filter_valid_blocks(text["blocks"])

    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": channel,
        "blocks": valid_blocks,
        "text": "Message from Bot"
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts

    response = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)
    print("response ---> ", response.json())


@app.route('/query', methods=['POST'])
def query():
    try:
        print("Query API called")
        data = request.get_json()
        # print(data)
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query_text = data['query']

        # Classify the query intent
        sys_prompt = qa.classify_query(query_text)
        
        print("sys_prompt ---> ", sys_prompt)
        print("query_text ---> ", query_text)
        response = qa.ai_magic(sys_prompt, query_text)
        print("Response from Query API ---> ", response)
        print("Query API executed")
        
        return jsonify({
            "intent": sys_prompt,
            "response": response
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("App running on port", port)
    app.debug = os.environ.get('FLASK_DEBUG', 'True') != 'False'
    app.run(host='127.0.0.1', port=port) 