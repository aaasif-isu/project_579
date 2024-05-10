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
To get started with the final stage of the project, follow these steps:

1. **Clone the Repository:**
   Begin by cloning the repository to your local machine using the following command
   ```bash
   git clone https://github.com/aaasif-isu/project_579.git
   ```
2. **Install Dependencies:**
   Install the required Python packages by executing
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables:**
   Create a .env file in the root directory and include the following lines, replacing _YOUR_API_KEY_ and _YOUR_DB_CLUSTER_ with your actual API key and database cluster URL respectively
   
   ```OPENAI_API_KEY=YOUR_API_KEY```
   
   ```DB_URL=YOUR_DB_CLUSTER```
   
4. **Run the Application:**
   Launch the application by running the main file. A popup window will appear where you need to upload a PDF file. After uploading, use the query section to input your queries
   ```bash
   funix rag.py
   ```
By following these steps, you will have the project set up and running on your system. This setup is designed to ensure a smooth start and easy testing of the project functionalities.

**Final version of the demo video-**

[*Click here*](https://iastate.box.com/s/wth54wixapd58k49hyv6s5i2ouxiu773)



## Second Milestone

1. Run the following command in the terminal:
   ```bash
   python main.py --pdf_file "paper.pdf" --query='what is attention?'
   ```   
2. Sample output:
   
   ```Connected to database
   
   Inserted Successfully
   
   Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.32it/s]


   attention is a mechanism that relates different positions of a single sequence....and so on
   ```

You can see the short video for the second milestone here:
[Demo Video 2](https://iastate.box.com/s/leunf877sgdd7t0zcb54nhc526otkw08)


## First Milestone

1. Run the following command in the terminal:

    ```bash
   python main_final.py --pdf_file "paper.pdf"
   ```     

2. If the script runs successfully, then you will see the following messages:
   
    ```
    Connected to database successfully
   
    Indexing successfully
    ```

You can see the short video for the first milestone here: 
[Demo Video 1](https://iastate.box.com/s/0befg4wzngas8i3x45hg6jif2zmzedih)







