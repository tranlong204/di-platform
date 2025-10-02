"""
Document processing service for handling various file formats
"""

import hashlib
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import markdown
from loguru import logger

from app.core.config import settings


class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.txt': 'text',
        '.md': 'markdown',
        '.html': 'html'
    }
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract text content"""
        try:
            file_path = Path(file_path)
            file_type = self.SUPPORTED_FORMATS.get(file_path.suffix.lower())
            
            if not file_type:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Extract text content
            content = self._extract_text(file_path, file_type)
            
            # Generate metadata
            metadata = self._generate_metadata(file_path, content)
            
            # Split into chunks
            chunks = self._split_into_chunks(content)
            
            return {
                'content': content,
                'chunks': chunks,
                'metadata': metadata,
                'file_type': file_type
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _extract_text(self, file_path: Path, file_type: str) -> str:
        """Extract text content based on file type"""
        if file_type == 'pdf':
            return self._extract_pdf_text(file_path)
        elif file_type == 'docx':
            return self._extract_docx_text(file_path)
        elif file_type == 'text':
            return self._extract_text_file(file_path)
        elif file_type == 'markdown':
            return self._extract_markdown_text(file_path)
        elif file_type == 'html':
            return self._extract_html_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text().strip()
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text().strip()
    
    def _generate_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Generate metadata for the document"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        return {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'content_hash': content_hash,
            'word_count': len(content.split()),
            'character_count': len(content),
            'file_extension': file_path.suffix
        }
    
    def _split_into_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Split content into overlapping chunks"""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                chunks.append({
                    'content': chunk_text,
                    'chunk_index': len(chunks),
                    'content_hash': chunk_hash,
                    'token_count': len(chunk_text.split()),
                    'start_word': i,
                    'end_word': min(i + self.chunk_size, len(words))
                })
        
        return chunks
