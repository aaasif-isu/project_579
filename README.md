# ComS_579: RAG System
Welcome to our RAG (Retrieval-Augmented Generation) system repository! This system leverages the power of LangChain to create a sophisticated pipeline that integrates with the Weaviate vector database for efficient data retrieval, enhanced by the intuitive and responsive interface provided by Funix.io.

## System Overview
- **LangChain Pipeline:** Utilizes LangChain to orchestrate the flow of data and queries, enabling efficient text processing and interaction.
- **Weaviate Vector Database:** Integrates with Weaviate for robust vector storage and retrieval, ensuring scalable and precise search functionalities.
- **Funix.io UI Library:** Employs Funix.io to create an intuitive and responsive user interface, enhancing the user experience and interaction with the system.

## Live Demo
Experience our application firsthand by accessing our hosted version. Feel free to test different functionalities and see how our system handles real-world data and queries. 

[Access the Hosted Application](https://huggingface.co/spaces/arafspn/rag-project)

## Final/ Third Milestone
1. Clone this repository to your local machine:  
   ```bash
   git clone https://github.com/aaasif-isu/project_579.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file in the root folder, put the line below and replace with your _API_KEY_ and _DB_CLUSTER__
   
   ```OPENAI_API_KEY=YOUR_API_KEY```

   ```DB_URL=YOUR_DB_CLUSTER```
   
4. Run the main file in the terminal and there will be a pop up window and you need to upload pdf file and upload it and in query section write your query:
   
   ```bash
   funix rag.py
   ```


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



