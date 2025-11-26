"""Chat functions for Q&A about artworks"""

from openai import OpenAI
import config


def chat_with_artwork(question, artwork_description, metadata, chat_history=None):
    """
    Answer questions about the artwork using GPT with context
    
    Args:
        question (str): User's question
        artwork_description (str): Description of the artwork
        metadata (dict): Metadata about the artwork
        chat_history (list): Previous chat messages (optional)
    
    Returns:
        str: Answer to the question
    """
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Build context
        context = f"""You are an art history expert and museum guide.
        
Information about the current artwork:

DESCRIPTION:
{artwork_description}

METADATA:
- Artist: {metadata.get('artist', 'Unknown')}
- Title: {metadata.get('title', 'Unknown')}
- Year: {metadata.get('year', 'Unknown')}
- Period: {metadata.get('period', 'Unknown')}

Answer questions about this artwork in an accessible, friendly manner.
Reference specific details of the work and explain art-historical concepts in an understandable way."""

        # Build messages
        messages = [
            {"role": "system", "content": context}
        ]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Call OpenAI Chat API
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        return f"Chat error: {str(e)}"
