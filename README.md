# Doc_Search
##A document search and retrieval system.

Queries over a corpus consisting of different document types(e.g., .docx, .pdf, .csv, .jpeg) when the user gives a text input and returns the most relevant documents.
- Used PyMuPDF library for text extraction from documents.
- Used PunktSentenceTokenizer from NLTK Package to tokenize documents.
- To make it space and computation efficient, removed stop words given in NLTK library.
- Used Porter's Stemming Algorithm to perform stemming
- Used TF-IDF technique to rank documents in their order of relevance to user query
