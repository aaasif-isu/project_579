# project_579: RAG System
Welcome to our RAG (Retrieval-Augmented Generation) system repository! This system leverages the power of LangChain to create a sophisticated pipeline that integrates with the Weaviate vector database for efficient data retrieval, enhanced by the intuitive and responsive interface provided by Funix.io.

## Features
- **LangChain Integration:** Utilizes LangChain for creating complex, context-aware AI applications with an emphasis on combining retrieval with language model outputs.
- **Weaviate Vector Database:** Employs Weaviate, a vector search database, for fast and scalable retrieval of contextually relevant information, improving the accuracy and relevance of generated content.
- **Funix.io UI:** Features a user interface developed with Funix.io, a modern UI library designed for creating high-performance web applications with a focus on user experience.

**First Part:**

1. Install packages:
   
    pip install -r requirements.txt

2. Run the following in the terminal:
   
    python main_final.py --pdf_file "paper.pdf"

3. If the script runs successfully, then you will see the following messages:
   
    Connected to database successfully
   
    Indexing successfully

You can see the 1st video demo here: 
https://iastate.box.com/s/0befg4wzngas8i3x45hg6jif2zmzedih


**Second part:**

1. Run the following in the terminal:
   
    python main.py --pdf_file "paper.pdf" --query='what is attention?'

2. Sample output:
   Connected to database
   
   Inserted Successfully
   
   Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.32it/s]


   attention is a mechanism that relates different positions of a single sequence....and so on

You can see the short video for the second milestone here:
https://iastate.box.com/s/leunf877sgdd7t0zcb54nhc526otkw08

**Final Part:**
1. Install packages:
   
    pip install -r requirements.txt
   
2. Run the following in the terminal:
   
    funix rag.py

