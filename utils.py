# utils.py

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list:
    """
    Splits the text into chunks of a specified size with a given overlap.
    
    Args:
        text (str): The text to be split.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
