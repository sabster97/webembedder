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
    print("data['event']", data['event'])
    # return jsonify({"ok": True})

    # Slack event challenge verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Handle message events
    if data['event']['type'] == 'message' and 'subtype' not in data['event'] and not data['event']['text'].startswith("Bot:"):
        user_id = data['event']['user']
        user_query = data['event']['text']
        channel_id = data['event']['channel']
        ts = data['event'].get('ts')
        thread_ts = data['event'].get('thread_ts')
        conversation_id = f"{user_id}-{ts}"

        # Run the task in a separate thread
        if not (data['event'].get('bot_id') or data['event'].get('subtype') == "message_changed"):
            thread = Thread(target=run_async_task, args=(
                user_query, channel_id, thread_ts, user_id, conversation_id, ts))
            thread.start()
    print("Slack event processed")
    return jsonify({"ok": True})


def run_async_task(user_query, channel_id, thread_ts, user_id, conversation_id, ts):
    print("Params received for run_async_task:")
    print("User Query:", user_query)
    print("Channel ID:", channel_id)
    print("Thread Timestamp:", thread_ts)
    print("User ID:", user_id)
    """Run async task in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task = loop.create_task(
            execute(user_query, channel_id, thread_ts, user_id, conversation_id, ts))
        loop.run_until_complete(task)
    finally:
        loop.close()


async def execute(user_query: str, channel_id: str, thread_ts: str, user_id: str, conversation_id: str, ts: str):
    """Execute the task with both ts values"""
    try:
        print(f"Starting execute with user_query: {
              user_query}, channel_id: {channel_id}, thread_ts: {thread_ts}")
        # check conversation id
        if conversation_id is not f"{user_id}-{thread_ts}":
            pass
        else:
            conversation_id = f"{user_id}-{thread_ts}"
        # Store user query in chat history
        qa.store_chat_message(user_query, user_id, thread_ts,
                              "user", conversation_id)
        print("User query stored in chat history")

        # Get system prompt based on query classification
        sys_prompt = qa.classify_query(user_query)
        print(f"System prompt for user_query: {sys_prompt}")

        # Get response using both chat history and content
        response = qa.ai_magic(sys_prompt, user_query, conversation_id)
        print(f"Response received: {response}")

        if "error" in response:
            error_message = {"blocks": [{"type": "section", "text": {
                "type": "mrkdwn", "text": "Sorry, I encountered an error processing your request."}}]}
            send_slack_message(channel_id, error_message, thread_ts, ts)
            print("Error response sent to Slack")
            return

        # Store assistant's response in chat history
        qa.store_chat_message(
            response["answer"], user_id, thread_ts, "assistant", conversation_id)
        print("Assistant's response stored in chat history")

        # Format and send response to Slack
        formatted_answer = format_for_slack(response["answer"])
        # # Convert string to JSON if it's a string
        # if isinstance(formatted_answer, str):
        #     try:
        #         formatted_answer = json.loads(formatted_answer)
        #     except json.JSONDecodeError as e:
        #         print(f"Failed to decode formatted answer: {formatted_answer}")
        #         error_message = {"blocks": [{"type": "section", "text": {
        #             "type": "mrkdwn", "text": "Sorry, I encountered an error formatting the response."}}]}
        #         send_slack_message(channel_id, error_message, thread_ts, ts)
        #         return
        send_slack_message(channel_id, formatted_answer, thread_ts, ts)
        print("Formatted answer sent to Slack")

    except Exception as e:
        print(f"Error in execute: {e}")
        error_message = {"blocks": [{"type": "section", "text": {
            "type": "mrkdwn", "text": "Sorry, I encountered an error processing your request."}}]}
        send_slack_message(channel_id, error_message, thread_ts, ts)


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


def send_slack_message(channel, text, thread_ts: str, ts: str):
    """Send a message to Slack."""
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

    payload["thread_ts"] = ts

    if thread_ts is not None:
        payload["thread_ts"] = thread_ts

    response = requests.post(
        "https://slack.com/api/chat.postMessage", headers=headers, json=payload)
    print("response ---> ", response.json())


# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         data = request.get_json()
#         if not data or 'query' not in data:
#             return jsonify({"error": "No query provided"}), 400

#         query_text = data['query']
#         thread_ts = data.get('thread_ts')  # Get thread_ts if provided

#         # Classify the query intent
#         sys_prompt = qa.classify_query(query_text)

#         # Pass thread_ts to ai_magic for conversation context
#         response = qa.ai_magic(sys_prompt, query_text, thread_ts)

#         if "error" in response:
#             return jsonify({"error": response["error"]}), 500

#         # Store the query and response in chat history
#         qa.store_chat_message(query_text, thread_ts, "user")
#         qa.store_chat_message(response["answer"], thread_ts, "assistant")

#         return jsonify({
#             "intent": sys_prompt,
#             "response": response
#         }), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("App running on port", port)
    app.debug = os.environ.get('FLASK_DEBUG', 'True') != 'False'
    app.run(host='127.0.0.1', port=port)
