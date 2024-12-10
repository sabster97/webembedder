import json


def chunk_text(text, max_length=2900):
    """Split text into chunks that respect Slack's character limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Add space before word except at start
        word_length = len(word) + (1 if current_length > 0 else 0)

        if current_length + word_length > max_length:
            # Join current chunk and start new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add remaining chunk if any
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def format_for_slack(input_text):
    """Format text for Slack with proper block structure and length limits."""
    blocks = []

    # Split into sections by double newline
    sections = input_text.split('\n\n')

    for section in sections:
        if not section.strip():
            continue

        # Handle headers (lines starting with ###)
        lines = section.split('\n')
        if lines[0].startswith('###'):
            header = lines[0].replace('### ', '').strip()
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header[:150]  # Slack header length limit
                }
            })

            # Remove header from section text
            section = '\n'.join(lines[1:])

        if section.strip():
            # Split long sections into chunks
            text_chunks = chunk_text(section)

            for chunk in text_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": chunk
                        }
                    })

    # Slack has a limit of 50 blocks per message
    if len(blocks) > 50:
        blocks = blocks[:50]
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "_(Message truncated due to length)_"
            }
        })

    return {
        "blocks": blocks
    }


if __name__ == "__main__":
    # Test input
    test_input = """### Example Header
This is a test section with some content.

### Another Header
This is another section with different content.
It has multiple lines.

### Final Section
- Bullet point 1
- Bullet point 2
- Bullet point 3"""

    formatted = format_for_slack(test_input)
    print(json.dumps(formatted, indent=2))
